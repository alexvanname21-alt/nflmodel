"""
The live "is the model actually good" ledger — separate from pipeline/backtest.py's
historical walk-forward test. This tracks real picks made in real time:

1. lock_strong_picks() — once a game has kicked off, its active strong-confidence
   picks (spread/total/props) can no longer change (generate_picks skips games
   that have already started), so that's the right moment to snapshot them
   permanently into tracked_picks. Only strong-confidence picks are tracked,
   per how this dashboard is meant to be used — a small number of high-bar
   plays with a real record, not everything the model ever considered.
2. grade_picks() — for locked-but-pending picks whose games are now final,
   fetch the actual result (scores via The Odds API for spread/total, actual
   player stats via fresh play-by-play for props) and grade win/loss/push.

Usage: python -m pipeline.track_picks
Called automatically at the end of pipeline/generate_picks.py too.
"""
from __future__ import annotations

from datetime import datetime, timezone

from config import HISTORY_SEASONS
from data.cfb_data import cfb_current_season, load_results as load_cfb_results
from data.nflverse_loader import load_pbp, load_rosters, load_schedules, season_list
from data.odds_client import OddsClient
from data.player_stats import derive_weekly_player_stats
from models.props_model import MARKET_TO_ACTUAL_FIELD
from storage.db import get_conn, get_pending_tracked_picks, grade_tracked_pick, lock_pick, now_iso


def _opening_spread_line(conn, game_id: str, side: str) -> float | None:
    """The team's own signed line from the EARLIEST row where this game's
    spread pick was already strong — i.e. the number the market offered when
    we first would have bet this, not the one right before kickoff."""
    row = conn.execute(
        """SELECT market_spread_home FROM game_projections
           WHERE game_id = ? AND spread_confidence = 'strong' AND market_spread_home IS NOT NULL
           ORDER BY generated_at ASC LIMIT 1""",
        (game_id,),
    ).fetchone()
    if row is None or row["market_spread_home"] is None:
        return None
    return row["market_spread_home"] if side == "home" else -row["market_spread_home"]


def _opening_total_line(conn, game_id: str) -> float | None:
    row = conn.execute(
        """SELECT market_total FROM game_projections
           WHERE game_id = ? AND total_confidence = 'strong' AND market_total IS NOT NULL
           ORDER BY generated_at ASC LIMIT 1""",
        (game_id,),
    ).fetchone()
    return row["market_total"] if row else None


def _opening_prop_line(conn, game_id: str, player_name: str, market: str) -> float | None:
    row = conn.execute(
        """SELECT market_line FROM player_prop_projections
           WHERE game_id = ? AND player_name = ? AND market = ? AND confidence = 'strong'
             AND market_line IS NOT NULL
           ORDER BY generated_at ASC LIMIT 1""",
        (game_id, player_name, market),
    ).fetchone()
    return row["market_line"] if row else None


def compute_clv(pick_type: str, side: str, open_line: float | None, close_line: float | None) -> float | None:
    """Positive = betting at the OPENING line would have beaten the closing
    line — the classic sharp-money signal, independent of whether the pick
    itself wins or loses. For spreads the team's own signed line already
    encodes direction (bigger signed number is always better for whichever
    team we picked), so open-minus-close works directly. For totals/props,
    which side of the number we're on flips which direction is favorable:
    a total moving UP after we bet the OVER means our earlier number was the
    easier one to clear (positive CLV); moving up after an UNDER bet makes
    our earlier number the harder one to have stayed below (negative CLV)."""
    if open_line is None or close_line is None:
        return None
    if pick_type == "spread":
        return open_line - close_line
    return (close_line - open_line) if side.lower() == "over" else (open_line - close_line)


def lock_strong_picks(conn) -> tuple[int, int]:
    now = datetime.now(timezone.utc)
    n_game = n_prop = 0

    games = conn.execute(
        """SELECT gp.*, g.commence_time, g.season, g.week, g.sport
           FROM game_projections gp JOIN games g ON gp.game_id = g.game_id
           WHERE gp.is_active = 1"""
    ).fetchall()

    for g in games:
        kickoff = datetime.fromisoformat(g["commence_time"].replace("Z", "+00:00"))
        if kickoff > now:
            continue  # game hasn't started — pick could still change, don't lock yet

        if g["spread_confidence"] == "strong" and g["spread_pick"]:
            side = "home" if (g["spread_edge"] or 0) > 0 else "away"
            close_line = g["market_spread_home"] if side == "home" else -g["market_spread_home"]
            open_line = _opening_spread_line(conn, g["game_id"], side)
            locked = lock_pick(conn, {
                "sport": g["sport"] or "nfl", "pick_type": "spread", "game_id": g["game_id"], "season": g["season"],
                "week": g["week"], "kickoff": g["commence_time"], "description": g["spread_pick"],
                "player_name": "", "market": "", "side": side,
                "open_line": open_line, "close_line": close_line,
                "clv": compute_clv("spread", side, open_line, close_line),
                "model_projection": g["proj_margin"], "confidence": g["spread_confidence"],
                "locked_at": now_iso(),
            })
            n_game += locked

        if g["total_confidence"] == "strong" and g["total_pick"]:
            side = "over" if (g["total_edge"] or 0) > 0 else "under"
            close_line = g["market_total"]
            open_line = _opening_total_line(conn, g["game_id"])
            locked = lock_pick(conn, {
                "sport": g["sport"] or "nfl", "pick_type": "total", "game_id": g["game_id"], "season": g["season"],
                "week": g["week"], "kickoff": g["commence_time"], "description": g["total_pick"],
                "player_name": "", "market": "", "side": side,
                "open_line": open_line, "close_line": close_line,
                "clv": compute_clv("total", side, open_line, close_line),
                "model_projection": g["proj_total"], "confidence": g["total_confidence"],
                "locked_at": now_iso(),
            })
            n_game += locked

    props = conn.execute(
        """SELECT pp.*, g.commence_time, g.season, g.week
           FROM player_prop_projections pp JOIN games g ON pp.game_id = g.game_id
           WHERE pp.is_active = 1 AND pp.confidence = 'strong'"""
    ).fetchall()

    for p in props:
        kickoff = datetime.fromisoformat(p["commence_time"].replace("Z", "+00:00"))
        if kickoff > now:
            continue

        market_label = p["market"].replace("player_", "").replace("_", " ").title()
        description = f"{p['player_name']} {p['pick']} {p['market_line']} {market_label}"
        close_line = p["market_line"]
        open_line = _opening_prop_line(conn, p["game_id"], p["player_name"], p["market"])
        locked = lock_pick(conn, {
            "pick_type": "prop", "game_id": p["game_id"], "season": p["season"],
            "week": p["week"], "kickoff": p["commence_time"], "description": description,
            "player_name": p["player_name"], "market": p["market"], "side": p["pick"],
            "open_line": open_line, "close_line": close_line,
            "clv": compute_clv("prop", p["pick"], open_line, close_line),
            "model_projection": p["projection"], "confidence": p["confidence"], "locked_at": now_iso(),
        })
        n_prop += locked

    return n_game, n_prop


def _grade_spread_total(pick, home_score: float, away_score: float) -> tuple[str, float]:
    if pick["pick_type"] == "spread":
        team_margin = (home_score - away_score) if pick["side"] == "home" else (away_score - home_score)
        cover_diff = team_margin + pick["close_line"]
        actual_value = team_margin
    else:  # total
        actual_total = home_score + away_score
        cover_diff = (actual_total - pick["close_line"]) if pick["side"] == "over" else (pick["close_line"] - actual_total)
        actual_value = actual_total

    if abs(cover_diff) < 1e-9:
        return "push", actual_value
    return ("win" if cover_diff > 0 else "loss"), actual_value


def _grade_prop(pick, actual_stat: float) -> tuple[str, float]:
    diff = actual_stat - pick["close_line"]
    if abs(diff) < 1e-9:
        return "push", actual_stat
    went_over = diff > 0
    correct = went_over if pick["side"] == "OVER" else not went_over
    return ("win" if correct else "loss"), actual_stat


def _nfl_final_scores(conn, game_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Final (home, away) scores keyed by our own game_id, from nflverse
    schedules — no lookback window, unlike the Odds API's scores endpoint
    (which only covers the last 3 days and misses anything graded late)."""
    placeholders = ",".join("?" * len(game_ids))
    rows = conn.execute(
        f"SELECT game_id, season, home_team, away_team FROM games WHERE game_id IN ({placeholders})",
        game_ids,
    ).fetchall()
    if not rows:
        return {}
    seasons = sorted({r["season"] for r in rows if r["season"]})
    schedules = load_schedules(seasons or season_list(1))
    done = schedules.dropna(subset=["home_score", "away_score"])
    sched_idx = {
        (r.season, r.home_team, r.away_team): (float(r.home_score), float(r.away_score))
        for r in done.itertuples()
    }
    out = {}
    for r in rows:
        key = (r["season"], r["home_team"], r["away_team"])
        if key in sched_idx:
            out[r["game_id"]] = sched_idx[key]
    return out


def _cfb_final_scores(conn, game_ids: list[str]) -> dict[str, tuple[float, float]]:
    """Same idea as _nfl_final_scores, via ESPN results and the college
    ESPN team ids we already store on each game (no name-matching needed)."""
    placeholders = ",".join("?" * len(game_ids))
    rows = conn.execute(
        f"SELECT game_id, season, home_espn_id, away_espn_id FROM games WHERE game_id IN ({placeholders})",
        game_ids,
    ).fetchall()
    if not rows:
        return {}
    seasons = sorted({r["season"] for r in rows if r["season"]}) or [cfb_current_season()]
    results = load_cfb_results(seasons)
    done = results.dropna(subset=["home_points", "away_points"])
    res_idx = {
        (r.season, r.home_id, r.away_id): (float(r.home_points), float(r.away_points))
        for r in done.itertuples()
    }
    out = {}
    for r in rows:
        key = (r["season"], r["home_espn_id"], r["away_espn_id"])
        if key in res_idx:
            out[r["game_id"]] = res_idx[key]
    return out


def grade_picks(conn, client: OddsClient | None = None) -> int:
    pending = get_pending_tracked_picks(conn)
    if not pending:
        return 0

    n_graded = 0
    game_picks = [p for p in pending if p["pick_type"] in ("spread", "total")]
    prop_picks = [p for p in pending if p["pick_type"] == "prop"]

    if game_picks:
        nfl_ids = [p["game_id"] for p in game_picks if (p["sport"] or "nfl") == "nfl"]
        cfb_ids = [p["game_id"] for p in game_picks if p["sport"] == "cfb"]
        scores_by_id: dict[str, tuple[float, float]] = {}
        if nfl_ids:
            scores_by_id.update(_nfl_final_scores(conn, nfl_ids))
        if cfb_ids:
            scores_by_id.update(_cfb_final_scores(conn, cfb_ids))

        for pick in game_picks:
            final = scores_by_id.get(pick["game_id"])
            if final is None:
                continue
            result, actual_value = _grade_spread_total(pick, final[0], final[1])
            grade_tracked_pick(conn, pick["id"], result, actual_value)
            n_graded += 1

    if prop_picks:
        seasons = season_list(HISTORY_SEASONS)
        pbp = load_pbp(seasons)
        rosters = load_rosters(seasons)
        weekly = derive_weekly_player_stats(pbp, rosters)
        final_games = _nfl_final_scores(conn, [p["game_id"] for p in prop_picks])

        for pick in prop_picks:
            field = MARKET_TO_ACTUAL_FIELD.get(pick["market"])
            if field is None:
                continue
            row = weekly[
                (weekly["player_display_name"] == pick["player_name"])
                & (weekly["season"] == pick["season"])
                & (weekly["week"] == pick["week"])
            ]
            if row.empty:
                if pick["game_id"] not in final_games:
                    continue  # game/week not in fresh data yet (pbp release lag)
                # Game's over and he never shows up in the play-by-play at all —
                # he didn't see the field (inactive/injury scratch), so his stat
                # was 0. A real sportsbook would void this instead of grading it,
                # but there's no "void" state here, so grade it as 0 and it's on
                # the ledger either way rather than stuck pending forever.
                actual_stat = 0.0
            else:
                actual_stat = float(row.iloc[0][field])
            result, actual_value = _grade_prop(pick, actual_stat)
            grade_tracked_pick(conn, pick["id"], result, actual_value)
            n_graded += 1

    return n_graded


def main():
    client = OddsClient()
    with get_conn() as conn:
        n_game, n_prop = lock_strong_picks(conn)
        print(f"Locked {n_game} new game picks and {n_prop} new prop picks.")
        n_graded = grade_picks(conn, client)
        print(f"Graded {n_graded} pending picks.")


if __name__ == "__main__":
    main()
