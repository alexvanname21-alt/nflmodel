"""
Turns opponent-adjusted EPA ratings (models/ratings.py) into a projected final
score for a specific matchup, independent of the betting market.

The conversion from "EPA/play" to "points" is calibrated from real historical
games (regressing actual final scores on each game's realized EPA), not a
guessed constant — see calibrate_points_model().
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import BLEND_WEIGHT, EDGE_THRESHOLDS, HISTORY_SEASONS
from data.nflverse_loader import load_pbp, load_schedules, season_list
from data.weather import weather_total_adjustment

HOME_REST_PT_PER_DAY = 0.05
MAX_REST_ADJ = 1.5
DIVISIONAL_MARGIN_DAMPEN = 0.90  # divisional games trend a bit closer to the number


@dataclass
class Calibration:
    margin_slope: float
    margin_intercept: float  # ~ league-average home field edge, in points
    total_slope: float
    total_intercept: float
    avg_pass_plays: float
    avg_rush_plays: float


def _game_epa_sums(pbp: pd.DataFrame) -> pd.DataFrame:
    plays = pbp[
        pbp["epa"].notna()
        & pbp["posteam"].notna()
        & ((pbp["pass"] == 1) | (pbp["rush"] == 1))
        & (pbp["season_type"] == "REG")
    ]
    per_team_game = plays.groupby(["game_id", "posteam"])["epa"].sum().reset_index()
    per_team_game_plays = plays.groupby(["game_id", "posteam"]).size().reset_index(name="n_plays")
    return per_team_game.merge(per_team_game_plays, on=["game_id", "posteam"])


# Fallback if a backtest cutoff has too little prior data to fit a stable
# regression (early in the very first test season). Roughly league-average
# NFL values: ~1.5-2pt home field, ~44-45pt average total.
FALLBACK_CALIBRATION = Calibration(
    margin_slope=1.0, margin_intercept=1.5,
    total_slope=0.5, total_intercept=44.5,
    avg_pass_plays=35.0, avg_rush_plays=25.0,
)
MIN_CALIBRATION_GAMES = 50


def calibrate_points_model_from_data(pbp: pd.DataFrame, schedules: pd.DataFrame) -> Calibration:
    """Core calibration over whatever pbp/schedules are handed in — see
    compute_team_ratings_from_pbp for why callers control the data window."""
    schedules = schedules[schedules["game_type"] == "REG"].dropna(subset=["home_score", "away_score"])
    if pbp.empty or schedules.empty:
        return FALLBACK_CALIBRATION

    epa_sums = _game_epa_sums(pbp)

    merged = schedules.merge(
        epa_sums.rename(columns={"posteam": "home_team", "epa": "home_epa_sum", "n_plays": "home_plays"}),
        on=["game_id", "home_team"],
        how="inner",
    ).merge(
        epa_sums.rename(columns={"posteam": "away_team", "epa": "away_epa_sum", "n_plays": "away_plays"}),
        on=["game_id", "away_team"],
        how="inner",
    )

    if len(merged) < MIN_CALIBRATION_GAMES:
        return FALLBACK_CALIBRATION

    margin_actual = merged["home_score"] - merged["away_score"]
    margin_x = merged["home_epa_sum"] - merged["away_epa_sum"]
    margin_slope, margin_intercept = np.polyfit(margin_x, margin_actual, 1)

    total_actual = merged["home_score"] + merged["away_score"]
    total_x = merged["home_epa_sum"] + merged["away_epa_sum"]
    total_slope, total_intercept = np.polyfit(total_x, total_actual, 1)

    plays = pbp[
        pbp["epa"].notna() & pbp["posteam"].notna() & (pbp["season_type"] == "REG")
    ]
    pass_plays_per_game = plays[plays["pass"] == 1].groupby(["game_id", "posteam"]).size()
    rush_plays_per_game = plays[plays["rush"] == 1].groupby(["game_id", "posteam"]).size()

    return Calibration(
        margin_slope=float(margin_slope),
        margin_intercept=float(margin_intercept),
        total_slope=float(total_slope),
        total_intercept=float(total_intercept),
        avg_pass_plays=float(pass_plays_per_game.mean()),
        avg_rush_plays=float(rush_plays_per_game.mean()),
    )


def calibrate_points_model(n_seasons: int = HISTORY_SEASONS) -> Calibration:
    seasons = season_list(n_seasons)
    pbp = load_pbp(seasons)
    schedules = load_schedules(seasons)
    return calibrate_points_model_from_data(pbp, schedules)


def _team_row(ratings_df: pd.DataFrame, team: str) -> pd.Series:
    row = ratings_df[ratings_df["team"] == team]
    if row.empty:
        raise ValueError(f"No ratings for team {team}")
    return row.iloc[0]


def _predicted_epa_sum(offense: pd.Series, defense: pd.Series, ratings_df: pd.DataFrame, calib: Calibration) -> float:
    league_def_pass = ratings_df["def_epa_pass"].mean()
    league_def_rush = ratings_df["def_epa_rush"].mean()

    matchup_pass = offense["off_epa_pass"] + defense["def_epa_pass"] - league_def_pass
    matchup_rush = offense["off_epa_rush"] + defense["def_epa_rush"] - league_def_rush

    return matchup_pass * calib.avg_pass_plays + matchup_rush * calib.avg_rush_plays


def project_game(
    home_team: str,
    away_team: str,
    ratings_df: pd.DataFrame,
    calib: Calibration,
    home_rest_days: int | None = None,
    away_rest_days: int | None = None,
    is_divisional: bool = False,
    weather: dict | None = None,
    market_spread_home: float | None = None,
    market_total: float | None = None,
    blend_weight: float = BLEND_WEIGHT,
) -> dict:
    """
    market_spread_home/market_total, when given, blend the model's raw
    projection toward the market's own number: final = blend_weight * model
    + (1 - blend_weight) * market. This exists because a backtest showed the
    raw model's edge is anti-predictive on the games it disagrees with the
    market on the most — a classic sign of an uncalibrated, too-volatile
    model. Shrinking toward market consensus is the standard fix; blend_weight
    was chosen by sweeping values against the backtest (see pipeline/backtest.py).
    A blend_weight of 1.0 disables shrinkage entirely (pure raw model).
    """
    home = _team_row(ratings_df, home_team)
    away = _team_row(ratings_df, away_team)

    home_epa_sum = _predicted_epa_sum(home, away, ratings_df, calib)
    away_epa_sum = _predicted_epa_sum(away, home, ratings_df, calib)

    margin = calib.margin_slope * (home_epa_sum - away_epa_sum) + calib.margin_intercept
    total = calib.total_slope * (home_epa_sum + away_epa_sum) + calib.total_intercept

    if home_rest_days is not None and away_rest_days is not None:
        rest_delta = home_rest_days - away_rest_days
        margin += max(-MAX_REST_ADJ, min(MAX_REST_ADJ, rest_delta * HOME_REST_PT_PER_DAY))

    if is_divisional:
        margin *= DIVISIONAL_MARGIN_DAMPEN

    total += weather_total_adjustment(weather)

    raw_margin, raw_total = margin, total

    if market_spread_home is not None:
        market_margin = -market_spread_home
        margin = blend_weight * margin + (1 - blend_weight) * market_margin
    if market_total is not None:
        margin_total_blend = blend_weight * total + (1 - blend_weight) * market_total
        total = margin_total_blend

    home_score = (total + margin) / 2
    away_score = (total - margin) / 2

    return {
        "home_team": home_team,
        "away_team": away_team,
        "proj_home_score": round(home_score, 1),
        "proj_away_score": round(away_score, 1),
        "proj_margin": round(margin, 1),
        "proj_total": round(total, 1),
        "raw_model_margin": round(raw_margin, 1),
        "raw_model_total": round(raw_total, 1),
    }


def evaluate_spread_edge(proj_margin: float, market_spread_home: float | None) -> dict:
    """market_spread_home is the home team's spread in standard notation
    (negative = home favored). proj_margin > 0 means home favored by model."""
    if market_spread_home is None:
        return {"edge": None, "pick": None, "confidence": None}

    market_margin = -market_spread_home  # convert spread to implied margin
    edge = proj_margin - market_margin

    if abs(edge) < EDGE_THRESHOLDS["lean"]:
        pick = None
        confidence = None
    else:
        favored_side = "home" if edge > 0 else "away"
        confidence = "strong" if abs(edge) >= EDGE_THRESHOLDS["strong"] else "lean"
        pick = favored_side

    return {"edge": round(edge, 1), "pick": pick, "confidence": confidence}


def evaluate_total_edge(proj_total: float, market_total: float | None) -> dict:
    if market_total is None:
        return {"edge": None, "pick": None, "confidence": None}

    edge = proj_total - market_total
    if abs(edge) < EDGE_THRESHOLDS["lean"]:
        pick = None
        confidence = None
    else:
        pick = "over" if edge > 0 else "under"
        confidence = "strong" if abs(edge) >= EDGE_THRESHOLDS["strong"] else "lean"

    return {"edge": round(edge, 1), "pick": pick, "confidence": confidence}
