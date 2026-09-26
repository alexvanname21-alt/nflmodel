"""
How hard should a player's *volume* (carries / targets) be shrunk toward a
prior when he has only 1-3 games of data?

The props model used one shrinkage strength (3 pseudo-games toward a generic
3-per-game prior) for everything. For a player with a single game that put
75% of the weight on "3 carries" — e.g. a rookie RB with 10 carries in his only
game was projected for ~5. Usage is persistent (it reflects a coaching decision),
so volume should be shrunk far less than efficiency (yards per carry).

This walk-forward test predicts a player's NEXT-game volume from his prior
same-season games using pseudo-game strength k, on players who already have a
real role (the population that actually gets prop lines). The prior mean and k
are chosen on the tune seasons and reported on held-out seasons.

Usage: python -m pipeline.props_volume_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.nflverse_loader import load_pbp, load_rosters, season_list
from data.player_stats import derive_weekly_player_stats

TUNE, HOLDOUT = [2022, 2023], [2024, 2025]
KS = [0, 0.25, 0.5, 1, 2, 3]
SPECS = {
    "carries (RB)": {"positions": {"RB"}, "col": "carries", "min_avg": 5.0},
    "targets (WR/TE)": {"positions": {"WR", "TE"}, "col": "targets", "min_avg": 3.0},
    "targets (RB)": {"positions": {"RB"}, "col": "targets", "min_avg": 2.0},
}


def samples(weekly: pd.DataFrame, spec: dict, seasons: list[int], max_n: int = 3):
    """(n_prior_games, prior_avg, next_game_value) for role players."""
    pool = weekly[weekly["position"].isin(spec["positions"]) & weekly["season"].isin(seasons)]
    out = []
    for (_, _), g in pool.sort_values("week").groupby(["player_id", "season"]):
        v = g[spec["col"]].to_numpy()
        for i in range(1, min(len(v), max_n + 1)):
            hist = v[:i]
            if hist.mean() >= spec["min_avg"]:
                out.append((i, hist.mean(), v[i]))
    return pd.DataFrame(out, columns=["n", "avg", "next"])


def qb_check(weekly: pd.DataFrame) -> None:
    """Predict a QB's next-game pass attempts when he just started (prev game >= 20 att):
    average of ALL prior games vs average of only 'starter-like' games (>= 15 att)."""
    print("\nQB pass attempts: does counting relief appearances hurt? (QBs who started last game)")
    qb = weekly[(weekly["position"] == "QB") & (weekly["attempts"] > 0)].sort_values(["season", "week"])
    rows = []
    for _, g in qb.groupby("player_id"):
        a, seasons = g["attempts"].to_numpy(), g["season"].to_numpy()
        for i in range(1, len(a)):
            if a[i - 1] < 20:
                continue
            hist = a[max(0, i - 17):i]
            starts = hist[hist >= 15]
            rows.append((seasons[i], hist.mean(), starts.mean() if len(starts) else np.nan, a[i]))
    df = pd.DataFrame(rows, columns=["season", "all", "starts", "next"]).dropna()
    for label, d in (("tune", df[df.season.isin(TUNE)]), ("holdout", df[df.season.isin(HOLDOUT)])):
        print(f"  {label:8} n={len(d):4}  MAE all-games {np.mean(np.abs(d['all'] - d['next'])):.2f}   "
              f"starter-games-only {np.mean(np.abs(d['starts'] - d['next'])):.2f}")


def ratio_check(weekly: pd.DataFrame) -> None:
    """Rush yards = carries x yards/carry. Compare yards/carry estimated as the
    mean of per-game ratios (old) vs pooled total yards / total carries (new)."""
    print("\nYards per carry estimator (RBs, >= 2 prior same-season games; next-game rush yards MAE)")
    rb = weekly[(weekly["position"] == "RB") & weekly["season"].isin(TUNE + HOLDOUT)].sort_values("week")
    rows = []
    for (_, season), g in rb.groupby(["player_id", "season"]):
        c, y = g["carries"].to_numpy(float), g["rushing_yards"].to_numpy(float)
        for i in range(2, len(c)):
            if c[:i].mean() < 5:
                continue
            per_game = np.where(c[:i] > 0, y[:i] / np.maximum(c[:i], 1), 0).mean()
            pooled = y[:i].sum() / max(c[:i].sum(), 1)
            shrunk = (y[:i].sum() + 60 * 4.2) / (c[:i].sum() + 60)
            vol = c[:i].mean()
            rows.append((season, vol * per_game, vol * pooled, vol * shrunk, y[i]))
    df = pd.DataFrame(rows, columns=["season", "old", "pooled", "pooled_shrunk", "actual"])
    for label, d in (("tune", df[df.season.isin(TUNE)]), ("holdout", df[df.season.isin(HOLDOUT)])):
        print(f"  {label:8} n={len(d):4}  mean-of-ratios {np.mean(np.abs(d['old'] - d['actual'])):.2f}   "
              f"pooled {np.mean(np.abs(d['pooled'] - d['actual'])):.2f}   pooled+shrink(60 carries) {np.mean(np.abs(d['pooled_shrunk'] - d['actual'])):.2f}")


def main() -> None:
    seasons = season_list(5)
    pbp, ros = load_pbp(seasons), load_rosters(seasons)
    weekly = derive_weekly_player_stats(pbp, ros)
    qb_check(weekly)
    ratio_check(weekly)

    for label, spec in SPECS.items():
        tune, hold = samples(weekly, spec, TUNE), samples(weekly, spec, HOLDOUT)
        prior = tune["next"].mean()  # population mean of role players' next-game volume
        print(f"\n{label}: role players with >= {spec['min_avg']}/game so far; prior mean = {prior:.1f} "
              f"(tune n={len(tune)}, holdout n={len(hold)})")
        print(f"  {'k (pseudo-games)':>17} {'tune MAE':>10} {'holdout MAE':>12}   holdout MAE by prior games (n=1 / 2 / 3)")
        best = None
        for k in KS:
            def mae(df):
                pred = (df["avg"] * df["n"] + k * prior) / (df["n"] + k)
                return (pred - df["next"]).abs()
            t, h = mae(tune), mae(hold)
            by_n = " / ".join(f"{h[hold['n'] == n].mean():.2f}" for n in (1, 2, 3))
            print(f"  {k:>17} {t.mean():>10.2f} {h.mean():>12.2f}   {by_n}")
            if best is None or t.mean() < best[0]:
                best = (t.mean(), k)
        naive = (hold["avg"] - hold["next"]).abs().mean()
        old = ((hold["avg"] * hold["n"] + 3 * 3.0) / (hold["n"] + 3) - hold["next"]).abs().mean()
        print(f"  best k on tune = {best[1]}   |  OLD behavior (k=3 toward a prior of 3.0): holdout MAE {old:.2f}")


if __name__ == "__main__":
    main()
