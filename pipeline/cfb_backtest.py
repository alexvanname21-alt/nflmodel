"""
Walk-forward backtest of the college ridge power rating against real closing
lines. For every week, ratings are fit only on games that kicked off before
that week, then FBS-vs-FBS games with a posted line are projected and graded.

Same honesty rules as the NFL backtest: hyperparameters are chosen on the tune
seasons only and reported on held-out seasons.

Usage: python -m pipeline.cfb_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.cfb_data import fbs_team_ids, load_closing_lines, load_results, load_teams
from models.cfb_model import DEFAULT_ALPHA, DEFAULT_HALF_LIFE_DAYS, DEFAULT_MAX_MARGIN, fit_ratings, project

BREAKEVEN = 0.5238
TUNE_SEASONS = [2022, 2023]
HOLDOUT_SEASONS = [2024, 2025]


def walk_forward(results, lines, fbs, seasons, alpha, half_life, max_margin) -> pd.DataFrame:
    games = results[
        results["season"].isin(seasons) & (results["seasontype"] == 2) & results["completed"]
        & results["home_id"].isin(fbs) & results["away_id"].isin(fbs)
    ].merge(lines, on="game_id", how="inner").dropna(subset=["spread_home", "total_line"])

    rows = []
    for (season, week), wk in games.groupby(["season", "week"]):
        r = fit_ratings(results, fbs, wk["start"].min(), alpha, half_life, max_margin)
        if r is None:
            continue
        for g in wk.itertuples():
            p = project(r, g.home_id, g.away_id, g.neutral)
            rows.append({
                "season": season, "week": week, "game": f"{g.away_name} @ {g.home_name}",
                **p, "spread_home": g.spread_home, "total_line": g.total_line,
                "actual_margin": g.home_points - g.away_points, "actual_total": g.home_points + g.away_points,
            })
    return pd.DataFrame(rows)


def grade(df: pd.DataFrame, min_edge: float) -> dict:
    """ATS and O/U record on picks where model disagrees with the line by >= min_edge pts."""
    out = {}
    market_margin = -df["spread_home"]
    s_edge = df["proj_margin"] - market_margin
    picks = df[s_edge.abs() >= min_edge]
    pe = s_edge[picks.index]
    cover = picks["actual_margin"] - (-picks["spread_home"])
    decided = cover != 0
    won = ((pe > 0) & (cover > 0)) | ((pe < 0) & (cover < 0))
    out["spread"] = (int(won[decided].sum()), int((~won[decided]).sum()))

    t_edge = df["proj_total"] - df["total_line"]
    picks = df[t_edge.abs() >= min_edge]
    te = t_edge[picks.index]
    diff = picks["actual_total"] - picks["total_line"]
    decided = diff != 0
    won = ((te > 0) & (diff > 0)) | ((te < 0) & (diff < 0))
    out["total"] = (int(won[decided].sum()), int((~won[decided]).sum()))
    return out


def fmt(rec):
    w, l = rec
    return f"{w}-{l} ({w / (w + l):.1%})" if w + l else "n/a"


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label}: {len(df)} games ---")
    print(f"  Margin MAE  model {np.mean(np.abs(df.proj_margin - (-df.spread_home))*0 + np.abs(df.proj_margin - df.actual_margin)):.2f}"
          f"   market {np.mean(np.abs(-df.spread_home - df.actual_margin)):.2f}")
    print(f"  Total  MAE  model {np.mean(np.abs(df.proj_total - df.actual_total)):.2f}"
          f"   market {np.mean(np.abs(df.total_line - df.actual_total)):.2f}")
    print(f"  corr(spread_home, actual_margin) = {np.corrcoef(df.spread_home, df.actual_margin)[0,1]:+.2f}  (should be strongly negative)")
    print(f"  {'min edge':>9}  {'spread ATS':>18}  {'total O/U':>18}")
    for e in (0, 2, 3, 4, 5, 7, 10):
        g = grade(df, e)
        print(f"  {e:>9}  {fmt(g['spread']):>18}  {fmt(g['total']):>18}")


def main():
    seasons = [2021, 2022, 2023, 2024, 2025]
    print("Loading college results + closing lines...")
    results = load_results(seasons)
    fbs = fbs_team_ids(results)
    lines = load_closing_lines(seasons, load_teams())

    print(f"\nSweeping ridge alpha / recency half-life on tune seasons {TUNE_SEASONS} (2021 = warm-up)...")
    print(f"{'alpha':>7} {'half-life':>10} {'spread edge>=3':>18} {'all-games spread ATS':>22} {'margin MAE':>11}")
    best = None
    for alpha in (10, 30, 100):
        for hl in (150, 250, 400):
            df = walk_forward(results, lines, fbs, TUNE_SEASONS, alpha, hl, DEFAULT_MAX_MARGIN)
            mae = np.mean(np.abs(df.proj_margin - df.actual_margin))
            g3, g0 = grade(df, 3)["spread"], grade(df, 0)["spread"]
            print(f"{alpha:>7} {hl:>10} {fmt(g3):>18} {fmt(g0):>22} {mae:>11.2f}")
            if best is None or mae < best[0]:
                best = (mae, alpha, hl)
    _, alpha, hl = best
    print(f"\nBest on tune set by margin MAE: alpha={alpha}, half-life={hl}d")

    tune = walk_forward(results, lines, fbs, TUNE_SEASONS, alpha, hl, DEFAULT_MAX_MARGIN)
    hold = walk_forward(results, lines, fbs, HOLDOUT_SEASONS, alpha, hl, DEFAULT_MAX_MARGIN)
    report(tune, f"TUNE seasons {TUNE_SEASONS}")
    report(hold, f"HOLDOUT seasons {HOLDOUT_SEASONS} (never used to choose parameters)")
    hold.to_csv("cfb_backtest_holdout.csv", index=False)


if __name__ == "__main__":
    main()
