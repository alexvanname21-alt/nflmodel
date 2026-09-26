"""
Recompute team EPA ratings and persist them to the DB. Run this once before
generate_picks.py, and again periodically (e.g. daily) as new games add data.

Usage: python -m pipeline.build_ratings
"""
from __future__ import annotations

from config import HISTORY_SEASONS
from data.nflverse_loader import current_season
from models.ratings import compute_team_ratings
from storage.db import get_conn, init_db, save_team_ratings


def main():
    init_db()
    print(f"Computing ratings from the last {HISTORY_SEASONS} seasons...")
    ratings = compute_team_ratings(HISTORY_SEASONS)
    season = current_season()

    with get_conn() as conn:
        save_team_ratings(conn, season, ratings)

    print(f"Saved ratings for {len(ratings)} teams (season={season}).")
    print(ratings.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
