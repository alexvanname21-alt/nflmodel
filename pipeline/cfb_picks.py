"""
College football: score predictions + spread/total picks for the current college
week, written to the same tables as the NFL picks (tagged sport='cfb') so the
pick-stability logic, kickoff-locking, grading and CLV tracking all apply.

Spreads and totals only — no player props for college.

Usage: python -m pipeline.cfb_picks
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from config import CFB_SPORT_KEY
from data.cfb_data import (
    build_name_map, cfb_current_season, current_week_window, fbs_team_ids,
    load_results, load_teams, odds_name_to_id,
)
from data.odds_client import OddsClient
from data.team_map import fmt_spread
from models.cfb_model import evaluate_edges, fit_ratings, project
from pipeline.generate_picks import extract_market_lines
from pipeline.track_picks import grade_picks, lock_strong_picks
from storage.db import (
    deactivate_stale_projections, deactivate_stale_props, get_conn, init_db, upsert_game, upsert_game_projection,
)


def run() -> None:
    init_db()
    season = cfb_current_season()
    print(f"Loading college results ({season - 3}-{season})...")
    results = load_results(list(range(season - 3, season + 1)))
    fbs = fbs_team_ids(results)
    teams = load_teams()
    name_map = build_name_map(teams)
    loc_by_id = dict(zip(teams["team_id"], teams["location"]))

    now = datetime.now(timezone.utc)
    ratings = fit_ratings(results, fbs, pd.Timestamp(now))
    if ratings is None:
        print("Not enough completed college games to fit ratings.")
        return
    n_done = int((results["season"] == season).sum() and results[(results.season == season)]["completed"].sum())
    print(f"Fit ratings on {int(results['completed'].sum())} games ({n_done} from {season}); "
          f"hfa {ratings.hfa:+.1f}, avg team score {ratings.mu:.1f}.")

    week, week_end = current_week_window(now)
    print(f"Targeting college week {week} (through {week_end:%a %m/%d}).")

    client = OddsClient()
    events = client.get_game_odds(sport_key=CFB_SPORT_KEY)
    n = skipped = 0
    with get_conn() as conn:
        if week is not None:
            cleared = deactivate_stale_projections(conn, season, week, sport="cfb")
            cleared_props = deactivate_stale_props(conn, season, week, sport="cfb")
            if cleared or cleared_props:
                print(f"Cleared {cleared} stale college projections and {cleared_props} stale props (other weeks).")

        for ev in events:
            commence = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if commence < now or commence >= week_end:
                continue

            hid = odds_name_to_id(ev["home_team"], name_map)
            aid = odds_name_to_id(ev["away_team"], name_map)
            if hid is None or aid is None:
                unmapped = [n for n, i in ((ev["home_team"], hid), (ev["away_team"], aid)) if i is None]
                print(f"  [skip] can't map team name(s) to a known school: {unmapped}")
                skipped += 1
                continue  # never guess — a wrong mapping would project the game as a generic team
            if hid not in fbs and aid not in fbs:
                skipped += 1
                continue

            market = extract_market_lines(ev)
            proj = project(ratings, hid, aid, neutral=False)
            edges = evaluate_edges(proj["proj_margin"], proj["proj_total"],
                                   market["market_spread_home"], market["market_total"])

            home_name = loc_by_id.get(hid, ev["home_team"])
            away_name = loc_by_id.get(aid, ev["away_team"])

            spread_pick = None
            if edges["spread_conf"] and market["market_spread_home"] is not None:
                if edges["spread_side"] == "home":
                    spread_pick = f"{home_name} {fmt_spread(market['market_spread_home'])}"
                else:
                    spread_pick = f"{away_name} {fmt_spread(-market['market_spread_home'])}"
            total_pick = None
            if edges["total_conf"] and market["market_total"] is not None:
                total_pick = f"{edges['total_side'].upper()} {market['market_total']}"

            upsert_game(conn, {
                "game_id": ev["id"], "season": season, "week": week,
                "home_team": home_name, "away_team": away_name,
                "commence_time": ev["commence_time"], "home_score": None, "away_score": None,
                "completed": 0, "sport": "cfb",
                "home_espn_id": hid, "away_espn_id": aid,
            })
            result = upsert_game_projection(conn, {
                "game_id": ev["id"],
                "proj_home_score": proj["proj_home_score"], "proj_away_score": proj["proj_away_score"],
                "proj_margin": proj["proj_margin"], "proj_total": proj["proj_total"],
                "market_spread_home": market["market_spread_home"], "market_total": market["market_total"],
                "market_ml_home": market["market_ml_home"], "market_ml_away": market["market_ml_away"],
                "spread_edge": edges["spread_edge"], "total_edge": edges["total_edge"],
                "spread_pick": spread_pick, "total_pick": total_pick,
                "spread_confidence": edges["spread_conf"], "total_confidence": edges["total_conf"],
                "weather_json": None, "injury_note": None,
                "line_book": market["line_book"],
            })
            n += 1
            print(f"  [{result}] {away_name} @ {home_name}: proj {proj['proj_away_score']}-{proj['proj_home_score']}"
                  f" (spread edge {edges['spread_edge']}, total edge {edges['total_edge']})")

        n_game, _ = lock_strong_picks(conn)
        n_graded = grade_picks(conn, client)
    print(f"\nProjected {n} games ({skipped} skipped: no FBS team / unmapped). "
          f"Locked {n_game} new picks, graded {n_graded}. Quota remaining: {client.last_quota_remaining}")


if __name__ == "__main__":
    run()
