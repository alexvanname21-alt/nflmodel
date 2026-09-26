"""
Opponent-adjusted EPA/play team ratings — offense and defense, split by
pass/rush, with recency weighting across weeks and seasons.

This is the core "power rating" the game model projects scores from. It is
deliberately NOT built from the betting market — it has to be an independent
read on team quality, or comparing it to the market line to find edges is
circular.

Method: iterative opponent adjustment (a handful of passes removing opponent
strength from each team's raw efficiency, similar in spirit to SRS/simple
ridge-regression power ratings), on top of a recency-weighted sample and a
shrinkage-to-league-average prior for small samples (bye weeks, early season).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HISTORY_SEASONS, INTRA_SEASON_DECAY, PRIOR_GAMES, SEASON_WEIGHTS
from data.nflverse_loader import load_pbp, season_list

N_ITERATIONS = 6


def _game_team_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """Collapse play-by-play into one row per (game, team, side) with mean
    EPA/play, split pass vs rush, for both offense and defense."""
    plays = pbp[
        pbp["epa"].notna()
        & pbp["posteam"].notna()
        & pbp["defteam"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
        & (pbp["season_type"] == "REG")
    ].copy()
    plays["play_kind"] = np.where(plays["pass"] == 1, "pass", "rush")

    off = (
        plays.groupby(["game_id", "season", "week", "posteam", "defteam", "play_kind"])["epa"]
        .mean()
        .rename("off_epa")
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opponent"})
    )
    deff = (
        plays.groupby(["game_id", "season", "week", "defteam", "posteam", "play_kind"])["epa"]
        .mean()
        .rename("def_epa")
        .reset_index()
        .rename(columns={"defteam": "team", "posteam": "opponent"})
    )
    return off, deff


def _weight(season: int, week: int, latest_season: int, max_week_in_season: dict) -> float:
    season_index = latest_season - season
    season_w = SEASON_WEIGHTS.get(season_index, 0.0)
    if season_w == 0.0:
        return 0.0
    weeks_from_end = max_week_in_season.get(season, week) - week
    intra_w = INTRA_SEASON_DECAY ** max(weeks_from_end, 0)
    return season_w * intra_w


def _iterative_adjust(game_df: pd.DataFrame, value_col: str, teams: list[str]) -> tuple[dict, dict, dict]:
    """
    game_df must have columns: team, opponent, {value_col}, opp_col, weight
    where opp_col is the *other* side's rating we adjust against (def for
    offense passes, off for defense passes). Returns (team_rating,
    games_sampled, effective_n) — effective_n is the SUM of recency weights
    per team, used for shrinkage instead of raw game count. A team with 69
    games on the books but most of them decayed to near-zero weight has much
    less real evidence than 69 full-weight games, and shrinkage needs to see
    that or it barely shrinks anything.
    """
    league_avg = np.average(game_df[value_col], weights=game_df["weight"]) if len(game_df) else 0.0
    rating = {t: league_avg for t in teams}

    for _ in range(N_ITERATIONS):
        opp_rating = game_df["opponent"].map(rating).fillna(league_avg)
        adjusted = game_df[value_col] - opp_rating + league_avg
        tmp = game_df.assign(adjusted=adjusted)
        grouped = tmp.groupby("team").apply(
            lambda g: np.average(g["adjusted"], weights=g["weight"]) if g["weight"].sum() > 0 else league_avg,
            include_groups=False,
        )
        new_rating = grouped.to_dict()
        for t in teams:
            if t not in new_rating:
                new_rating[t] = league_avg
        rating = new_rating

    games_sampled = game_df.groupby("team")["game_id"].nunique().to_dict()
    effective_n = game_df.groupby("team")["weight"].sum().to_dict()
    return rating, games_sampled, effective_n


def _shrink(rating: float, effective_n: float, league_avg: float, prior_games: float = PRIOR_GAMES) -> float:
    if effective_n <= 0:
        return league_avg
    weight = effective_n / (effective_n + prior_games)
    return weight * rating + (1 - weight) * league_avg


def compute_team_ratings_from_pbp(pbp: pd.DataFrame, prior_games: float = PRIOR_GAMES) -> pd.DataFrame:
    """
    Core rating computation over whatever play-by-play is handed in. Callers
    control recency by what they pass — the live pipeline passes the last
    N seasons; the backtester passes only plays strictly before the game
    being tested, so ratings never see the future. prior_games controls
    shrinkage strength (in effective, recency-weighted games) toward league
    average — exposed for the backtest to sweep and validate.
    """
    off, deff = _game_team_epa(pbp)
    if pbp.empty:
        return pd.DataFrame(columns=[
            "team", "off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush",
            "off_epa_total", "def_epa_total", "net_epa", "games_sampled",
        ])

    latest_season = int(pbp["season"].max())
    max_week_in_season = pbp.groupby("season")["week"].max().to_dict()

    off["weight"] = off.apply(
        lambda r: _weight(r["season"], r["week"], latest_season, max_week_in_season), axis=1
    )
    deff["weight"] = deff.apply(
        lambda r: _weight(r["season"], r["week"], latest_season, max_week_in_season), axis=1
    )
    off = off[off["weight"] > 0]
    deff = deff[deff["weight"] > 0]

    teams = sorted(set(off["team"]) | set(deff["team"]))

    off_pass = off[off["play_kind"] == "pass"].rename(columns={"off_epa": "value"})
    off_rush = off[off["play_kind"] == "rush"].rename(columns={"off_epa": "value"})
    def_pass = deff[deff["play_kind"] == "pass"].rename(columns={"def_epa": "value"})
    def_rush = deff[deff["play_kind"] == "rush"].rename(columns={"def_epa": "value"})

    off_pass_rating, off_pass_games, off_pass_n = _iterative_adjust(off_pass, "value", teams)
    off_rush_rating, off_rush_games, off_rush_n = _iterative_adjust(off_rush, "value", teams)
    def_pass_rating, def_pass_games, def_pass_n = _iterative_adjust(def_pass, "value", teams)
    def_rush_rating, def_rush_games, def_rush_n = _iterative_adjust(def_rush, "value", teams)

    league_off_pass = np.mean(list(off_pass_rating.values())) if off_pass_rating else 0.0
    league_off_rush = np.mean(list(off_rush_rating.values())) if off_rush_rating else 0.0
    league_def_pass = np.mean(list(def_pass_rating.values())) if def_pass_rating else 0.0
    league_def_rush = np.mean(list(def_rush_rating.values())) if def_rush_rating else 0.0

    rows = []
    for t in teams:
        n_games = off_pass_games.get(t, 0) + off_rush_games.get(t, 0)
        n_games = n_games / 2 if n_games else 0  # pass+rush counted separately, avg back to games; display only
        off_pass_v = _shrink(off_pass_rating.get(t, league_off_pass), off_pass_n.get(t, 0), league_off_pass, prior_games)
        off_rush_v = _shrink(off_rush_rating.get(t, league_off_rush), off_rush_n.get(t, 0), league_off_rush, prior_games)
        def_pass_v = _shrink(def_pass_rating.get(t, league_def_pass), def_pass_n.get(t, 0), league_def_pass, prior_games)
        def_rush_v = _shrink(def_rush_rating.get(t, league_def_rush), def_rush_n.get(t, 0), league_def_rush, prior_games)

        off_total = (off_pass_v + off_rush_v) / 2
        def_total = (def_pass_v + def_rush_v) / 2
        rows.append(
            {
                "team": t,
                "off_epa_pass": round(off_pass_v, 4),
                "off_epa_rush": round(off_rush_v, 4),
                "def_epa_pass": round(def_pass_v, 4),
                "def_epa_rush": round(def_rush_v, 4),
                "off_epa_total": round(off_total, 4),
                "def_epa_total": round(def_total, 4),
                "net_epa": round(off_total - def_total, 4),
                "games_sampled": int(round(n_games)),
            }
        )

    return pd.DataFrame(rows).sort_values("net_epa", ascending=False).reset_index(drop=True)


def compute_team_ratings(n_seasons: int = HISTORY_SEASONS, prior_games: float = PRIOR_GAMES) -> pd.DataFrame:
    seasons = season_list(n_seasons)
    pbp = load_pbp(seasons)
    return compute_team_ratings_from_pbp(pbp, prior_games)
