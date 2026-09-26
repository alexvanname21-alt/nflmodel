"""
Historical NFL data via nfl_data_py (wraps the free public nflverse releases).
Everything is cached to local parquet in data/cache/ since a multi-season
play-by-play pull is tens of thousands of rows and slow to re-download.

Completed seasons are cached forever (they never change). The CURRENT season
is always re-fetched — caching it would freeze the model on whatever week it
happened to be when the cache file was first written, which is a real bug
this file used to have: it silently kept treating a season as "week 2" for
days after week 2 had finished. See pipeline/cfb_data.py's load_results for
the same pattern done first, on the college side.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Columns we actually need from pbp — the full table has 370+ columns per play.
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "home_team", "away_team",
    "posteam", "defteam", "play_type", "epa", "success", "pass", "rush",
    "yards_gained", "air_yards", "yards_after_catch", "sack", "interception",
    "fumble_lost", "penalty", "touchdown", "field_goal_result",
    "posteam_score", "defteam_score", "wp", "qb_epa", "cpoe",
    # player attribution, for deriving weekly stats ourselves (nflverse's
    # separate player_stats release lags and is missing the current season —
    # pbp is always current, so we build weekly stats from it directly).
    "passer_player_id", "passer_player_name", "receiver_player_id",
    "receiver_player_name", "rusher_player_id", "rusher_player_name",
    "complete_pass", "pass_attempt", "rush_attempt", "pass_touchdown",
    "rush_touchdown", "passing_yards", "receiving_yards", "rushing_yards",
]


def current_season() -> int:
    """NFL season 'year' — season runs Sep->Feb, so Jan/Feb still belongs to
    the season that started the previous September."""
    today = pd.Timestamp.today()
    return today.year if today.month >= 3 else today.year - 1


def current_week(schedules: pd.DataFrame, season: int) -> int | None:
    """The NFL week currently being played: the lowest week number among
    this season's games that haven't finished yet. None once the season's
    fully complete (or hasn't started, before any schedule data exists)."""
    upcoming = schedules[(schedules["season"] == season) & schedules["home_score"].isna()]
    if upcoming.empty:
        return None
    return int(upcoming["week"].min())


def season_list(n_seasons: int) -> list[int]:
    latest = current_season()
    return list(range(latest - n_seasons + 1, latest + 1))


def _is_stale(season: int, force_refresh: bool) -> bool:
    return force_refresh or season == current_season()


def load_pbp(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    frames = []
    for season in seasons:
        cache_file = CACHE_DIR / f"pbp_{season}.parquet"
        if cache_file.exists() and not _is_stale(season, force_refresh):
            frames.append(pd.read_parquet(cache_file))
            continue
        try:
            df = nfl.import_pbp_data(
                [season], downcast=True, cache=False, include_participation=False
            )
        except Exception as e:
            # Current in-progress season can 404 mid-week before nflverse
            # publishes that week's file; skip it rather than fail the whole load.
            print(f"  [skip] pbp {season} unavailable yet: {e}")
            if cache_file.exists():
                frames.append(pd.read_parquet(cache_file))  # fall back to last-known-good
            continue
        keep_cols = [c for c in PBP_COLUMNS if c in df.columns]
        df = df[keep_cols]
        df.to_parquet(cache_file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PBP_COLUMNS)


def load_schedules(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    frames = []
    for season in seasons:
        cache_file = CACHE_DIR / f"schedules_{season}.parquet"
        if cache_file.exists() and not _is_stale(season, force_refresh):
            frames.append(pd.read_parquet(cache_file))
            continue
        try:
            df = nfl.import_schedules([season])
        except Exception as e:
            print(f"  [skip] schedules {season} unavailable yet: {e}")
            if cache_file.exists():
                frames.append(pd.read_parquet(cache_file))
            continue
        df.to_parquet(cache_file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rosters(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    frames = []
    for season in seasons:
        cache_file = CACHE_DIR / f"rosters_{season}.parquet"
        if cache_file.exists() and not _is_stale(season, force_refresh):
            frames.append(pd.read_parquet(cache_file))
            continue
        try:
            df = nfl.import_seasonal_rosters([season])
        except Exception as e:
            print(f"  [skip] rosters {season} unavailable yet: {e}")
            if cache_file.exists():
                frames.append(pd.read_parquet(cache_file))
            continue
        df.to_parquet(cache_file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_injuries(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    frames = []
    for season in seasons:
        cache_file = CACHE_DIR / f"injuries_{season}.parquet"
        if cache_file.exists() and not _is_stale(season, force_refresh):
            frames.append(pd.read_parquet(cache_file))
            continue
        try:
            df = nfl.import_injuries([season])
        except Exception as e:
            print(f"  [skip] injuries {season} unavailable yet: {e}")
            if cache_file.exists():
                frames.append(pd.read_parquet(cache_file))
            continue
        df.to_parquet(cache_file)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
