"""
Player prop projections, built the way the user described it: volume/usage
share first, matchup and game-script second, hot-streak efficiency regressed
toward a baseline rather than extrapolated.

Formula, per player per market:
    projected_volume   = team_implied_plays_of_type * player's recency-weighted
                          usage share (targets/carries), nudged by opponent
                          matchup strength
    projected_yards     = projected_volume * recency-weighted efficiency
                          (yds/target, yds/carry), shrunk toward the
                          position-group average for small samples
    projected_anytime_td = recency-weighted TD rate/game, shrunk toward
                          position baseline, nudged by matchup and by the
                          team's projected scoring (more plays -> more scores)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import HISTORY_SEASONS
from data.nflverse_loader import load_pbp, load_rosters, season_list
from data.player_stats import derive_weekly_player_stats

RECENCY_GAMES = 17  # ~1 season; caps how far back we look, not how it's weighted
USAGE_PRIOR_GAMES = 3  # shrinkage strength for small samples

# pipeline/props_backtest.py compared exponential recency-decay against a
# plain equal-weighted average (and "just use last game") on 4 seasons of
# real player data: decay-weighting was a statistical wash against a plain
# average (occasionally slightly worse), and both clearly beat "last game."
# So weighting here is uniform across the window — the extra complexity of
# decay wasn't earning its keep. Re-run that backtest before changing this.

POSITION_PRIORS = {
    # league-average efficiency by position, used as the shrinkage target
    "RB": {"yds_per_carry": 4.1, "yds_per_target": 6.8, "catch_rate": 0.75, "td_per_game": 0.35},
    "WR": {"yds_per_carry": 7.0, "yds_per_target": 8.2, "catch_rate": 0.63, "td_per_game": 0.30},
    "TE": {"yds_per_carry": 5.0, "yds_per_target": 7.4, "catch_rate": 0.68, "td_per_game": 0.28},
    "QB": {"yds_per_attempt": 6.9, "completion_pct": 0.64, "td_per_game": 1.4, "int_per_game": 0.7},
}


@dataclass
class TeamPace:
    pass_plays: float
    rush_plays: float


def _recency_weights(n: int) -> np.ndarray:
    # Uniform weights over the trailing RECENCY_GAMES window — see the note
    # above RECENCY_GAMES for why this isn't decay-weighted.
    return np.ones(n) / n


def compute_team_pace(pbp: pd.DataFrame, n_games: int = RECENCY_GAMES) -> dict[str, TeamPace]:
    plays = pbp[
        pbp["posteam"].notna() & ((pbp["pass"] == 1) | (pbp["rush"] == 1)) & (pbp["season_type"] == "REG")
    ]
    counts = plays.groupby(["posteam", "game_id", "season", "week"]).agg(
        pass_plays=("pass", "sum"), rush_plays=("rush", "sum")
    ).reset_index()

    pace = {}
    for team, g in counts.groupby("posteam"):
        g = g.sort_values(["season", "week"], ascending=False).head(n_games)
        w = _recency_weights(len(g))
        pace[team] = TeamPace(
            pass_plays=float(np.average(g["pass_plays"], weights=w)),
            rush_plays=float(np.average(g["rush_plays"], weights=w)),
        )
    return pace


def _shrink(value: float, n: float, prior: float, prior_n: float = USAGE_PRIOR_GAMES) -> float:
    if n <= 0:
        return prior
    w = n / (n + prior_n)
    return w * value + (1 - w) * prior


def _shrink_ratio(numerator: float, denominator: float, prior: float, prior_denominator: float) -> float:
    """Pooled ratio (e.g. total yards / total carries) shrunk toward a prior, in
    units of opportunities. Beats averaging per-game ratios, where a 1-carry
    game counts as much as a 20-carry game (pipeline/props_volume_check.py)."""
    return (numerator + prior_denominator * prior) / (denominator + prior_denominator)


# Volume (carries/targets) is a coaching decision and persists, so it's shrunk far
# less than efficiency — and toward the average for players who actually have a
# role (the ones that get prop lines), not a generic 3/game. A single game of 10
# carries used to be pulled to ~5 because of that 3/game prior. Values below come
# from pipeline/props_volume_check.py (walk-forward, validated on held-out seasons).
VOLUME_PRIORS = {  # (stat, position) -> (prior per game, pseudo-games of shrinkage)
    ("carries", "RB"): (10.8, 0.5),
    ("targets", "WR"): (5.5, 1.0),
    ("targets", "TE"): (5.5, 1.0),
    ("targets", "RB"): (2.9, 2.0),
}
DEFAULT_VOLUME_PRIOR = (3.0, 1.0)
# A weak defense changes how many yards a player gets per touch, not how many touches
# he gets (volume follows role and game script). Matchup therefore scales yardage and
# TDs only, and gently: a model that swung volume +/-25% projected Mahomes for 45 attempts.
MATCHUP_CLIP = 0.10
EFFICIENCY_PRIOR_OPPS = {"carries": 60, "targets": 40, "attempts": 150}
STARTER_MIN_ATTEMPTS = 15  # a QB game with fewer attempts was a relief appearance, not a start


def _volume(g: pd.DataFrame, col: str, stat: str, position: str, w: np.ndarray) -> float:
    prior, k = VOLUME_PRIORS.get((stat, position), DEFAULT_VOLUME_PRIOR)
    return _shrink(np.average(g[col], weights=w), len(g), prior, k)


def _player_recent_games(weekly: pd.DataFrame, player_name: str, n_games: int = RECENCY_GAMES) -> pd.DataFrame:
    g = weekly[weekly["player_display_name"] == player_name].sort_values(
        ["season", "week"], ascending=False
    ).head(n_games)
    return g


def project_receiving_rushing(
    player_name: str,
    position: str,
    team_pace: TeamPace,
    opponent_def_epa_pass: float,
    opponent_def_epa_rush: float,
    league_def_epa_pass: float,
    league_def_epa_rush: float,
    weekly: pd.DataFrame,
) -> dict | None:
    g = _player_recent_games(weekly, player_name)
    if g.empty:
        return None
    if position == "QB":
        # Same relief-appearance problem as passing: a QB's carries in 1-5 attempt
        # cameo games say nothing about his rushing when he starts.
        starts = g[g["attempts"] >= STARTER_MIN_ATTEMPTS]
        if len(starts):
            g = starts

    n = len(g)
    w = _recency_weights(n)
    # Receiving/rushing priors only exist for RB/WR/TE — a QB (or any other
    # position) showing up on a rush_yds/receptions market falls back to RB
    # shape, since that's closer to a scrambling QB's rushing profile than
    # the passing-shaped QB prior.
    prior = POSITION_PRIORS.get(position, POSITION_PRIORS["RB"])
    if "yds_per_carry" not in prior:
        prior = POSITION_PRIORS["RB"]

    target_share = _shrink(np.average(g["target_share"].fillna(0), weights=w), n, 0.12)

    targets_pg = _volume(g, "targets", "targets", position, w)
    carries_pg = _volume(g, "carries", "carries", position, w)

    opps = EFFICIENCY_PRIOR_OPPS
    catch_rate = _shrink_ratio(g["receptions"].sum(), g["targets"].sum(), prior["catch_rate"], opps["targets"])
    yds_per_target = _shrink_ratio(g["receiving_yards"].sum(), g["targets"].sum(), prior["yds_per_target"], opps["targets"])
    yds_per_carry = _shrink_ratio(g["rushing_yards"].sum(), g["carries"].sum(), prior["yds_per_carry"], opps["carries"])
    td_per_game = _shrink(
        np.average((g["receiving_tds"] + g["rushing_tds"]), weights=w), n, prior["td_per_game"]
    )

    # Matchup multiplier: defense's EPA/play allowed vs league average.
    # Positive def_epa_* means defense allows MORE value than average (bad defense) -> multiplier > 1.
    pass_matchup_mult = 1.0 + np.clip((opponent_def_epa_pass - league_def_epa_pass) * 3.0, -MATCHUP_CLIP, MATCHUP_CLIP)
    rush_matchup_mult = 1.0 + np.clip((opponent_def_epa_rush - league_def_epa_rush) * 3.0, -MATCHUP_CLIP, MATCHUP_CLIP)

    proj_targets = targets_pg
    proj_carries = carries_pg
    proj_receptions = proj_targets * catch_rate
    proj_rec_yards = proj_targets * yds_per_target * pass_matchup_mult
    proj_rush_yards = proj_carries * yds_per_carry * rush_matchup_mult
    proj_td = td_per_game * ((pass_matchup_mult + rush_matchup_mult) / 2)

    return {
        "player_name": player_name,
        "position": position,
        "proj_targets": round(proj_targets, 1),
        "proj_receptions": round(proj_receptions, 1),
        "proj_reception_yds": round(proj_rec_yards, 1),
        "proj_carries": round(proj_carries, 1),
        "proj_rush_yds": round(proj_rush_yards, 1),
        "proj_anytime_td": round(proj_td, 2),
        "sample_games": n,
        "target_share": round(target_share, 3),
    }


def project_passing(
    player_name: str,
    team_pace: TeamPace,
    opponent_def_epa_pass: float,
    league_def_epa_pass: float,
    weekly: pd.DataFrame,
) -> dict | None:
    g = _player_recent_games(weekly, player_name)
    if g.empty:
        return None

    prior = POSITION_PRIORS["QB"]

    # Only games he actually started count. A backup's history is mostly 1-6 attempt
    # relief appearances; averaging those in made a QB who just took over (32
    # attempts in his one start) look like a ~10-attempt passer.
    starts = g[g["attempts"] >= STARTER_MIN_ATTEMPTS]
    n_starts = len(starts)
    base = starts if n_starts else g
    w = _recency_weights(len(base))

    attempts_pg = (
        _shrink(np.average(starts["attempts"], weights=_recency_weights(n_starts)), n_starts, team_pace.pass_plays, 1.0)
        if n_starts else team_pace.pass_plays
    )
    attempts = base["attempts"].sum()
    completion_pct = _shrink_ratio(base["completions"].sum(), attempts, prior["completion_pct"], EFFICIENCY_PRIOR_OPPS["attempts"])
    yds_per_attempt = _shrink_ratio(base["passing_yards"].sum(), attempts, prior["yds_per_attempt"], EFFICIENCY_PRIOR_OPPS["attempts"])
    td_pg = _shrink(np.average(base["passing_tds"], weights=w), len(base), prior["td_per_game"])
    int_pg = _shrink(np.average(base["interceptions"], weights=w), len(base), prior["int_per_game"])

    matchup_mult = 1.0 + np.clip((opponent_def_epa_pass - league_def_epa_pass) * 3.0, -MATCHUP_CLIP, MATCHUP_CLIP)

    proj_attempts = attempts_pg
    proj_completions = proj_attempts * completion_pct
    proj_pass_yards = proj_attempts * yds_per_attempt * matchup_mult
    proj_pass_tds = td_pg * matchup_mult
    proj_ints = int_pg / matchup_mult  # tougher defense -> more INT risk, inverse of yardage mult

    return {
        "player_name": player_name,
        "position": "QB",
        "proj_attempts": round(proj_attempts, 1),
        "proj_completions": round(proj_completions, 1),
        "proj_pass_yds": round(proj_pass_yards, 1),
        "proj_pass_tds": round(proj_pass_tds, 2),
        "proj_interceptions": round(proj_ints, 2),
        "sample_games": n_starts,
    }


MARKET_TO_FIELD = {
    "player_pass_yds": "proj_pass_yds",
    "player_pass_tds": "proj_pass_tds",
    "player_pass_interceptions": "proj_interceptions",
    "player_pass_completions": "proj_completions",
    "player_pass_attempts": "proj_attempts",
    "player_rush_yds": "proj_rush_yds",
    "player_rush_attempts": "proj_carries",
    "player_reception_yds": "proj_reception_yds",
    "player_receptions": "proj_receptions",
}

# Market key -> the ACTUAL (realized) stat column in derive_weekly_player_stats
# output, used for grading a locked pick once the game's final — distinct from
# MARKET_TO_FIELD, which maps to the *projected* field name.
MARKET_TO_ACTUAL_FIELD = {
    "player_pass_yds": "passing_yards",
    "player_pass_tds": "passing_tds",
    "player_pass_interceptions": "interceptions",
    "player_pass_completions": "completions",
    "player_pass_attempts": "attempts",
    "player_rush_yds": "rushing_yards",
    "player_rush_attempts": "carries",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}


# A %-only edge threshold blows up on tiny lines (a 0.5-INT prop off by one
# is a "200% edge" but is not actually a strong signal). Require a minimum
# ABSOLUTE edge in the market's own units too, so low-volume markets can't
# dominate purely on percentage.
MARKET_MIN_ABS_EDGE = {
    "player_pass_yds": 15.0,
    "player_pass_tds": 0.4,
    "player_pass_interceptions": 0.3,
    "player_pass_completions": 2.5,
    "player_pass_attempts": 3.0,
    "player_rush_yds": 8.0,
    "player_rush_attempts": 2.0,
    "player_reception_yds": 8.0,
    "player_receptions": 1.0,
}


# Guardrails. In an efficient market a prop line is right on average, so when the
# model is wildly off it's far more often the MODEL missing something (a role
# change, an injury, game script) than a real edge — the same pattern the game
# backtest found, where the biggest model-vs-market gaps did worst. There's no
# free history of prop lines to validate the exact cutoff, so treat 40% as a
# conservative sanity limit rather than a tuned number. The model also can't see
# role changes (a teammate injured, a backup promoted), which is exactly when the
# market moves a line far from a player's history.
MAX_PLAUSIBLE_EDGE_PCT = 0.40

# 'strong' requires both a large percentage edge AND a large absolute edge (in
# units of the market's own minimum meaningful edge, MARKET_MIN_ABS_EDGE) —
# tightened from 2.2x/1.5x after a full week's worth of props showed ~100
# 'strong' picks, too many to mean anything as a curated top tier. At these
# multiples, roughly the top 3-4% of a week's props qualify (~20 out of 550+)
# rather than ~1 in 6. There's no historical prop-line backtest to validate an
# exact cutoff (see README), so this is a judgment call, not a tuned number —
# revisit once the Track Record has enough graded strong props to say whether
# this bar is actually the right one.
STRONG_EDGE_PCT_MULT = 3.0
STRONG_ABS_MULT = 2.0
MIN_GAMES_FOR_STRONG = 3  # with 1-2 games of data a model has no basis for a top-tier call


def evaluate_prop_edge(
    projection: float, market_line: float, market: str | None = None, min_edge_pct: float = 0.06,
    sample_games: int | None = None,
) -> dict:
    if market_line in (None, 0) or projection is None:
        return {"edge_pct": None, "pick": None, "confidence": None}

    abs_edge = projection - market_line
    edge_pct = abs_edge / market_line
    min_abs = MARKET_MIN_ABS_EDGE.get(market, 0.0)

    if abs(edge_pct) < min_edge_pct or abs(abs_edge) < min_abs:
        return {"edge_pct": round(edge_pct, 3), "pick": None, "confidence": None}
    if abs(edge_pct) > MAX_PLAUSIBLE_EDGE_PCT:
        return {"edge_pct": round(edge_pct, 3), "pick": None, "confidence": None}

    pick = "OVER" if edge_pct > 0 else "UNDER"
    strong = abs(edge_pct) >= min_edge_pct * STRONG_EDGE_PCT_MULT and abs(abs_edge) >= min_abs * STRONG_ABS_MULT
    if strong and sample_games is not None and sample_games < MIN_GAMES_FOR_STRONG:
        strong = False
    confidence = "strong" if strong else "lean"
    return {"edge_pct": round(edge_pct, 3), "pick": pick, "confidence": confidence}
