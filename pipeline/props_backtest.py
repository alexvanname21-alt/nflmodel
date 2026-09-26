"""
Lightweight accuracy check for the player-props projection logic.

There's no free historical player-prop odds feed (The Odds API's historical
snapshots are a paid add-on), so this can't grade actual over/under bets the
way pipeline/backtest.py does for game spreads/totals. What it CAN test:
does the props model's core method — recency-weighted average with shrinkage
— actually predict a player's next game better than two naive baselines?
If it doesn't beat "just use last game's number," the shrinkage/weighting
isn't earning its complexity.

Walk-forward within each player-season: for game i, only games 0..i-1 (same
season) are used to build each method's prediction for game i.

Usage: python -m pipeline.props_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HISTORY_SEASONS
from data.nflverse_loader import load_pbp, load_rosters, season_list
from data.player_stats import derive_weekly_player_stats

# Comparison baseline only — the live model no longer uses decay weighting
# (see models/props_model.py), this constant exists purely so this script can
# keep showing "what if we had used decay" for future reference.
RECENCY_DECAY = 0.85
MIN_PRIOR_GAMES = 3
STATS = {
    "rushing_yards": {"position": {"RB"}, "min_opportunity_col": "carries", "min_opportunity": 5},
    "receiving_yards": {"position": {"WR", "TE", "RB"}, "min_opportunity_col": "targets", "min_opportunity": 3},
    "receptions": {"position": {"WR", "TE", "RB"}, "min_opportunity_col": "targets", "min_opportunity": 3},
    "passing_yards": {"position": {"QB"}, "min_opportunity_col": "attempts", "min_opportunity": 10},
}


def _recency_weighted(values: np.ndarray) -> float:
    n = len(values)
    w = np.array([RECENCY_DECAY ** i for i in range(n)])[::-1]  # most recent = highest weight
    return float(np.average(values, weights=w))


def evaluate_stat(weekly: pd.DataFrame, stat: str, cfg: dict) -> dict:
    pool = weekly[
        weekly["position"].isin(cfg["position"])
        & (weekly[cfg["min_opportunity_col"]] >= cfg["min_opportunity"])
    ].sort_values(["player_id", "season", "week"])

    naive_last_errs, naive_avg_errs, weighted_errs, trailing8_errs = [], [], [], []

    for (_, _season), g in pool.groupby(["player_id", "season"]):
        vals = g[stat].to_numpy()
        for i in range(MIN_PRIOR_GAMES, len(vals)):
            history = vals[:i]
            actual = vals[i]
            naive_last_errs.append(abs(history[-1] - actual))
            naive_avg_errs.append(abs(history.mean() - actual))
            weighted_errs.append(abs(_recency_weighted(history[-8:]) - actual))
            trailing8_errs.append(abs(history[-8:].mean() - actual))

    n = len(naive_last_errs)
    return {
        "stat": stat,
        "n_samples": n,
        "mae_last_game": round(float(np.mean(naive_last_errs)), 2) if n else None,
        "mae_season_avg": round(float(np.mean(naive_avg_errs)), 2) if n else None,
        "mae_recency_weighted": round(float(np.mean(weighted_errs)), 2) if n else None,
        "mae_trailing8_equal": round(float(np.mean(trailing8_errs)), 2) if n else None,
    }


def main():
    seasons = season_list(HISTORY_SEASONS)
    print(f"Loading {seasons[0]}-{seasons[-1]} data...")
    pbp = load_pbp(seasons)
    rosters = load_rosters(seasons)
    weekly = derive_weekly_player_stats(pbp, rosters)

    print(f"\n{'='*88}\nPROPS MODEL METHOD CHECK — does recency-weighting beat naive guesses?\n{'='*88}\n")
    print(f"{'Stat':<18}{'N':>7}{'MAE: last game':>16}{'MAE: season avg':>17}{'MAE: decay-wtd':>16}{'MAE: trail8-eq':>16}")
    for stat, cfg in STATS.items():
        r = evaluate_stat(weekly, stat, cfg)
        if r["n_samples"] == 0:
            print(f"{stat:<18}  no samples")
            continue
        print(f"{r['stat']:<18}{r['n_samples']:>7}{r['mae_last_game']:>16}{r['mae_season_avg']:>17}{r['mae_recency_weighted']:>16}{r['mae_trailing8_equal']:>16}")

    print(
        "\nLower MAE is better. If 'weighted' isn't clearly the lowest of the three, the "
        "props model's recency-weighting + shrinkage isn't adding value over a naive guess "
        "and should be simplified or retuned rather than trusted as-is."
    )


if __name__ == "__main__":
    main()
