"""
The main orchestrator: pulls live odds, projects every game in the current
NFL week (and its player props), and writes picks to the DB with the
pick-stability rule applied (won't flip a published pick just because the
line twitched — see storage/db.py upsert_game_projection). Players on IR/
reserve/cut, or manually excluded in data/excluded_players.txt, are skipped.

Usage: python -m pipeline.generate_picks
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from config import DB_PATH, HISTORY_SEASONS
from data.nflverse_loader import current_season, current_week, load_injuries, load_pbp, load_rosters, load_schedules, season_list
from data.odds_client import OddsClient
from data.player_availability import load_manual_exclusions, load_unavailable_players
from data.player_stats import derive_weekly_player_stats
from data.team_map import fmt_spread, nickname, to_abbr
from data.weather import get_game_weather
from models.game_model import calibrate_points_model, evaluate_spread_edge, evaluate_total_edge, project_game
from models.props_model import (
    MARKET_TO_FIELD,
    compute_team_pace,
    evaluate_prop_edge,
    project_passing,
    project_receiving_rushing,
)
from pipeline.track_picks import grade_picks, lock_strong_picks
from storage.db import (
    deactivate_props_for_players,
    deactivate_stale_projections,
    deactivate_stale_props,
    get_conn,
    init_db,
    upsert_game,
    upsert_game_projection,
    upsert_prop_projection,
)

PASS_MARKETS = {"player_pass_yds", "player_pass_tds", "player_pass_interceptions", "player_pass_completions", "player_pass_attempts"}
RUSH_REC_MARKETS = {"player_rush_yds", "player_rush_attempts", "player_reception_yds", "player_receptions"}


BOOK_PREFERENCE = ["fanduel", "draftkings"]


def _first_book(bookmakers: list[dict], market_key: str) -> tuple[str, list[dict]] | None:
    """(book key, outcomes) from the most-preferred book that offers this market."""
    by_key = {bm["key"]: bm for bm in bookmakers}
    for book in BOOK_PREFERENCE + [k for k in by_key if k not in BOOK_PREFERENCE]:
        bm = by_key.get(book)
        if not bm:
            continue
        for m in bm.get("markets", []):
            if m["key"] == market_key and m.get("outcomes"):
                return book, m["outcomes"]
    return None


def extract_market_lines(event: dict) -> dict:
    """Real, bettable numbers from ONE book (FanDuel, else DraftKings) per market.
    Averaging the two books produced lines like 2.8 that nobody actually offers."""
    home, away = event["home_team"], event["away_team"]
    bms = event.get("bookmakers", [])

    market_spread_home = market_total = market_ml_home = market_ml_away = None
    line_book = None

    sp = _first_book(bms, "spreads")
    if sp:
        line_book, outcomes = sp
        market_spread_home = next((o["point"] for o in outcomes if o["name"] == home and o.get("point") is not None), None)
    tot = _first_book(bms, "totals")
    if tot:
        line_book = line_book or tot[0]
        market_total = next((o["point"] for o in tot[1] if o.get("point") is not None), None)
    ml = _first_book(bms, "h2h")
    if ml:
        market_ml_home = next((o["price"] for o in ml[1] if o["name"] == home), None)
        market_ml_away = next((o["price"] for o in ml[1] if o["name"] == away), None)

    return {
        "market_spread_home": market_spread_home,
        "market_total": market_total,
        "market_ml_home": market_ml_home,
        "market_ml_away": market_ml_away,
        "line_book": line_book,
    }


def get_injury_note(injuries: pd.DataFrame, team: str, season: int) -> str | None:
    recent_week = injuries[injuries["season"] == season]["week"].max()
    if pd.isna(recent_week):
        return None
    rows = injuries[
        (injuries["season"] == season)
        & (injuries["week"] == recent_week)
        & (injuries["team"] == team)
        & (injuries["report_status"].isin(["Out", "Doubtful"]))
    ]
    if rows.empty:
        return None
    names = rows["full_name"].tolist()
    return f"{team} ({recent_week}): " + ", ".join(names)


def find_schedule_row(schedules: pd.DataFrame, home_abbr: str, away_abbr: str) -> pd.Series | None:
    rows = schedules[(schedules["home_team"] == home_abbr) & (schedules["away_team"] == away_abbr)]
    if rows.empty:
        return None
    return rows.iloc[-1]


def run_game_projections(
    client: OddsClient, ratings_df: pd.DataFrame, schedules: pd.DataFrame, injuries: pd.DataFrame, target_week: int
) -> list[dict]:
    calib = calibrate_points_model(HISTORY_SEASONS)
    events = client.get_game_odds()
    now = datetime.now(timezone.utc)
    season = current_season()

    processed = []
    with get_conn() as conn:
        for event in events:
            commence = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
            if commence < now:
                continue

            home_abbr = to_abbr(event["home_team"])
            away_abbr = to_abbr(event["away_team"])
            if not home_abbr or not away_abbr:
                continue

            sched_row = find_schedule_row(schedules, home_abbr, away_abbr)
            if sched_row is None or int(sched_row["week"]) != target_week:
                continue  # only the current NFL week
            home_rest = int(sched_row["home_rest"]) if pd.notna(sched_row["home_rest"]) else None
            away_rest = int(sched_row["away_rest"]) if pd.notna(sched_row["away_rest"]) else None
            is_div = bool(sched_row["div_game"])
            week = int(sched_row["week"])

            weather = get_game_weather(home_abbr, commence)
            market = extract_market_lines(event)

            try:
                proj = project_game(
                    home_abbr, away_abbr, ratings_df, calib,
                    home_rest_days=home_rest, away_rest_days=away_rest,
                    is_divisional=is_div, weather=weather,
                    market_spread_home=market["market_spread_home"],
                    market_total=market["market_total"],
                )
            except ValueError:
                continue  # team not in ratings (bye-week bug, mid-season trade, etc.)

            spread_eval = evaluate_spread_edge(proj["proj_margin"], market["market_spread_home"])
            total_eval = evaluate_total_edge(proj["proj_total"], market["market_total"])

            home_injury = get_injury_note(injuries, home_abbr, season)
            away_injury = get_injury_note(injuries, away_abbr, season)
            injury_note = " | ".join(n for n in [home_injury, away_injury] if n) or None

            spread_pick = None
            if spread_eval["pick"] and market["market_spread_home"] is not None:
                if spread_eval["pick"] == "home":
                    side_abbr, team_line = home_abbr, market["market_spread_home"]
                else:
                    side_abbr, team_line = away_abbr, -market["market_spread_home"]
                spread_pick = f"{nickname(side_abbr)} {fmt_spread(team_line)}"

            total_pick = None
            if total_eval["pick"]:
                total_pick = f"{total_eval['pick'].upper()} {market['market_total']}"

            upsert_game(conn, {
                "game_id": event["id"], "season": season, "week": week,
                "home_team": home_abbr, "away_team": away_abbr,
                "commence_time": event["commence_time"],
                "home_score": None, "away_score": None, "completed": 0,
            })

            result = upsert_game_projection(conn, {
                "game_id": event["id"],
                "proj_home_score": proj["proj_home_score"],
                "proj_away_score": proj["proj_away_score"],
                "proj_margin": proj["proj_margin"],
                "proj_total": proj["proj_total"],
                "market_spread_home": market["market_spread_home"],
                "market_total": market["market_total"],
                "market_ml_home": market["market_ml_home"],
                "market_ml_away": market["market_ml_away"],
                "spread_edge": spread_eval["edge"],
                "total_edge": total_eval["edge"],
                "spread_pick": spread_pick,
                "total_pick": total_pick,
                "spread_confidence": spread_eval["confidence"],
                "total_confidence": total_eval["confidence"],
                "weather_json": json.dumps(weather) if weather else None,
                "injury_note": injury_note,
                "line_book": market["line_book"],
            })
            print(f"  [{result}] {away_abbr} @ {home_abbr}: proj {proj['proj_away_score']}-{proj['proj_home_score']} "
                  f"(spread edge {spread_eval['edge']}, total edge {total_eval['edge']})")
            processed.append({"event": event, "home_abbr": home_abbr, "away_abbr": away_abbr, "commence": commence})

    return processed


def run_player_props(
    client: OddsClient, processed_games: list[dict], ratings_df: pd.DataFrame, pace: dict,
    weekly: pd.DataFrame, rosters: pd.DataFrame, unavailable_players: set[str],
) -> None:
    league_def_pass = ratings_df["def_epa_pass"].mean()
    league_def_rush = ratings_df["def_epa_rush"].mean()
    r = ratings_df.set_index("team")

    with get_conn() as conn:
        for g in processed_games:
            event_id = g["event"]["id"]
            home_abbr, away_abbr = g["home_abbr"], g["away_abbr"]
            try:
                event_props = client.get_player_props(event_id)
            except Exception as e:
                print(f"  [props skip] {event_id}: {e}")
                continue

            lines_by_book: dict[tuple, dict[str, float]] = {}
            for bm in event_props.get("bookmakers", []):
                for m in bm.get("markets", []):
                    for o in m.get("outcomes", []):
                        player, point = o.get("description"), o.get("point")
                        if player and point is not None:
                            lines_by_book.setdefault((m["key"], player), {}).setdefault(bm["key"], point)

            for (market_key, player_name), by_book in lines_by_book.items():
                try:
                    if player_name in unavailable_players:
                        continue
                    # a real line from one book (FanDuel first), not an average of two
                    book = next((b for b in BOOK_PREFERENCE if b in by_book), next(iter(by_book)))
                    market_line = by_book[book]
                    proster = rosters[rosters["player_name"] == player_name]
                    if proster.empty:
                        continue
                    position = proster.iloc[-1]["position"]
                    player_team = proster.iloc[-1]["team"]
                    opponent = away_abbr if player_team == home_abbr else home_abbr if player_team == away_abbr else None
                    if opponent is None or opponent not in r.index:
                        continue

                    if market_key in PASS_MARKETS:
                        proj = project_passing(
                            player_name, pace.get(player_team), r.loc[opponent, "def_epa_pass"],
                            league_def_pass, weekly,
                        )
                    elif market_key in RUSH_REC_MARKETS:
                        proj = project_receiving_rushing(
                            player_name, position, pace.get(player_team),
                            r.loc[opponent, "def_epa_pass"], r.loc[opponent, "def_epa_rush"],
                            league_def_pass, league_def_rush, weekly,
                        )
                    else:
                        continue

                    if proj is None:
                        continue
                    field = MARKET_TO_FIELD.get(market_key)
                    projection_value = proj.get(field)
                    if projection_value is None:
                        continue

                    edge = evaluate_prop_edge(projection_value, market_line, market=market_key, sample_games=proj.get("sample_games"))
                    rationale = f"proj {projection_value} vs line {market_line} (sample {proj['sample_games']} games)"

                    result = upsert_prop_projection(conn, {
                        "game_id": event_id, "player_name": player_name, "team": player_team,
                        "opponent": opponent, "market": market_key, "projection": projection_value,
                        "market_line": market_line, "edge_pct": edge["edge_pct"], "pick": edge["pick"],
                        "confidence": edge["confidence"], "rationale": rationale,
                    })
                    if edge["pick"]:
                        print(f"    [{result}] {player_name} {market_key}: {edge['pick']} {market_line} "
                              f"(proj {projection_value}, {edge['confidence']})")
                except Exception as e:
                    print(f"    [prop skip] {player_name} {market_key}: {e}")


def main():
    init_db()
    print("Loading historical data (cached after first run)...")
    seasons = season_list(HISTORY_SEASONS)
    pbp = load_pbp(seasons)
    rosters = load_rosters(seasons)
    schedules = load_schedules(seasons)
    injuries = load_injuries([current_season()])
    weekly = derive_weekly_player_stats(pbp, rosters)
    pace = compute_team_pace(pbp)

    season = current_season()
    week = current_week(schedules, season)
    if week is None:
        print(f"No upcoming games found for the {season} season — nothing to project.")
        return
    print(f"Targeting {season} week {week} only.")

    unavailable = load_unavailable_players(season) | load_manual_exclusions()
    print(f"Excluding {len(unavailable)} inactive/IR/manually-excluded players from props.")

    with get_conn() as conn:
        n_stale_proj = deactivate_stale_projections(conn, season, week)
        n_stale_week_props = deactivate_stale_props(conn, season, week)
        n_stale_props = deactivate_props_for_players(conn, unavailable)
    if n_stale_proj or n_stale_week_props or n_stale_props:
        print(f"Cleared {n_stale_proj} stale projections (wrong week), {n_stale_week_props} stale props (wrong week), "
              f"and {n_stale_props} stale props (excluded players).")

    with get_conn() as conn:
        from storage.db import get_latest_ratings
        rows = get_latest_ratings(conn, season)
    if not rows:
        print("No ratings found — run `python -m pipeline.build_ratings` first.")
        return
    ratings_df = pd.DataFrame([dict(r) for r in rows])

    client = OddsClient()

    print("\nProjecting games...")
    processed = run_game_projections(client, ratings_df, schedules, injuries, week)

    print("\nProjecting player props...")
    run_player_props(client, processed, ratings_df, pace, weekly, rosters, unavailable)

    print("\nTracking strong picks (locking kicked-off games, grading completed ones)...")
    with get_conn() as conn:
        n_locked_game, n_locked_prop = lock_strong_picks(conn)
        n_graded = grade_picks(conn, client)
    print(f"Locked {n_locked_game} new game picks and {n_locked_prop} new prop picks. Graded {n_graded} pending picks.")

    print(f"\nDone. Quota remaining: {client.last_quota_remaining}")


if __name__ == "__main__":
    main()
