"""
Weekly player stats derived directly from play-by-play.

nfl_data_py's separate `player_stats` release lags nflverse's pbp release and
is missing the current in-progress season entirely as of this build — pbp
always has the latest week, so we aggregate it ourselves instead of trusting
the weekly release to be current.
"""
from __future__ import annotations

import pandas as pd


def _passing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[pbp["pass_attempt"] == 1].copy()
    g = p.groupby(["season", "week", "posteam", "passer_player_id"]).agg(
        attempts=("pass_attempt", "sum"),
        completions=("complete_pass", "sum"),
        passing_yards=("passing_yards", "sum"),
        passing_tds=("pass_touchdown", "sum"),
        interceptions=("interception", "sum"),
    ).reset_index().rename(columns={"passer_player_id": "player_id", "posteam": "team"})
    return g


def _rushing_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    r = pbp[pbp["rush_attempt"] == 1].copy()
    g = r.groupby(["season", "week", "posteam", "rusher_player_id"]).agg(
        carries=("rush_attempt", "sum"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id", "posteam": "team"})
    return g


def _receiving_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    rec = pbp[(pbp["pass_attempt"] == 1) & pbp["receiver_player_id"].notna()].copy()
    g = rec.groupby(["season", "week", "posteam", "receiver_player_id"]).agg(
        targets=("pass_attempt", "sum"),
        receptions=("complete_pass", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id", "posteam": "team"})

    team_targets = rec.groupby(["season", "week", "posteam"])["pass_attempt"].sum().rename(
        "team_targets"
    ).reset_index().rename(columns={"posteam": "team"})
    g = g.merge(team_targets, on=["season", "week", "team"], how="left")
    g["target_share"] = (g["targets"] / g["team_targets"]).fillna(0)
    return g.drop(columns=["team_targets"])


def derive_weekly_player_stats(pbp: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (season, week, player_id) with passing/rushing/
    receiving volume + efficiency stats, plus player_display_name/position
    joined in from seasonal rosters.
    """
    reg = pbp[pbp["season_type"] == "REG"]

    passing = _passing_stats(reg)
    rushing = _rushing_stats(reg)
    receiving = _receiving_stats(reg)

    merged = passing.merge(
        rushing, on=["season", "week", "player_id"], how="outer", suffixes=("", "_rush")
    ).merge(
        receiving, on=["season", "week", "player_id"], how="outer", suffixes=("", "_rec")
    )

    # Consolidate the team column (each source has its own copy).
    team_col = merged["team"].combine_first(merged.get("team_rush")).combine_first(
        merged.get("team_rec")
    )
    merged["team"] = team_col
    merged = merged.drop(columns=[c for c in ["team_rush", "team_rec"] if c in merged.columns])

    numeric_cols = [
        "attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds", "target_share",
    ]
    for c in numeric_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)

    roster_slim = (
        rosters[["player_id", "player_name", "position", "season"]]
        .drop_duplicates(subset=["player_id", "season"])
    )
    merged = merged.merge(roster_slim, on=["player_id", "season"], how="left")
    merged = merged.rename(columns={"player_name": "player_display_name"})
    merged = merged.dropna(subset=["player_id"])
    return merged
