"""
College football power ratings: a ridge regression on final scores.

Each game gives two observations — (home team's points) and (away team's
points) — modeled as

    points = mu + off[scoring team] + def[opponent] + hfa * (scoring team is home, non-neutral)

Ridge shrinkage pulls every team toward average (what you want for a team with
3 games of data), and each game is weighted by an exponential recency decay
so last season still informs September but fades as the new season fills in.
All non-FBS opponents are lumped into a single team — they play only a handful
of games and mostly show up as paycheck games.

Unlike the NFL model this uses scores, not EPA (no fresh free college
play-by-play exists), so it carries less signal per game. Whether it beats the
market is an empirical question — see pipeline/cfb_backtest.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

NONFBS = -1

DEFAULT_ALPHA = 1.0
DEFAULT_HALF_LIFE_DAYS = 400.0
DEFAULT_MAX_MARGIN = 40.0  # winsorize garbage-time blowouts when fitting


@dataclass
class CfbRatings:
    mu: float
    hfa: float
    off: dict
    dfn: dict
    fbs: set

    def key(self, team_id: int) -> int:
        return team_id if team_id in self.fbs else NONFBS


def fit_ratings(
    results: pd.DataFrame, fbs: set, asof: pd.Timestamp,
    alpha: float = DEFAULT_ALPHA, half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    max_margin: float = DEFAULT_MAX_MARGIN,
) -> CfbRatings | None:
    g = results[results["completed"] & (results["start"] < asof)].dropna(subset=["home_points", "away_points"])
    if len(g) < 200:
        return None

    home_k = g["home_id"].map(lambda t: t if t in fbs else NONFBS).to_numpy()
    away_k = g["away_id"].map(lambda t: t if t in fbs else NONFBS).to_numpy()
    teams = sorted(set(home_k) | set(away_k))
    idx = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(g)

    hp, ap = g["home_points"].to_numpy(float), g["away_points"].to_numpy(float)
    if max_margin:  # cap blowouts symmetrically around the midpoint
        excess = np.maximum(np.abs(hp - ap) - max_margin, 0) * np.sign(hp - ap)
        hp, ap = hp - excess / 2, ap + excess / 2

    X = np.zeros((2 * n, 2 * T + 1))
    y = np.empty(2 * n)
    neutral = g["neutral"].to_numpy()
    for r in range(n):
        h, a = idx[home_k[r]], idx[away_k[r]]
        X[2 * r, h] = 1; X[2 * r, T + a] = 1
        X[2 * r + 1, a] = 1; X[2 * r + 1, T + h] = 1
        if not neutral[r]:
            X[2 * r, 2 * T] = 1
        y[2 * r], y[2 * r + 1] = hp[r], ap[r]

    age_days = (asof - g["start"]).dt.total_seconds().to_numpy() / 86400
    w = np.repeat(0.5 ** (age_days / half_life_days), 2)

    model = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=w)
    coef = model.coef_
    return CfbRatings(
        mu=float(model.intercept_), hfa=float(coef[2 * T]),
        off={t: float(coef[idx[t]]) for t in teams},
        dfn={t: float(coef[T + idx[t]]) for t in teams},
        fbs=fbs,
    )


def project(r: CfbRatings, home_id: int, away_id: int, neutral: bool = False) -> dict:
    h, a = r.key(home_id), r.key(away_id)
    hoff, hdef = r.off.get(h, 0.0), r.dfn.get(h, 0.0)
    aoff, adef = r.off.get(a, 0.0), r.dfn.get(a, 0.0)
    home_pts = r.mu + hoff + adef + (0.0 if neutral else r.hfa)
    away_pts = r.mu + aoff + hdef
    return {
        "proj_home_score": round(home_pts, 1), "proj_away_score": round(away_pts, 1),
        "proj_margin": round(home_pts - away_pts, 1), "proj_total": round(home_pts + away_pts, 1),
    }


def evaluate_edges(proj_margin: float, proj_total: float, spread_home: float | None, total_line: float | None) -> dict:
    """Edge + tier for spread and total vs the market, using the (much wider)
    college thresholds. spread_home is standard notation (negative = home favored)."""
    from config import CFB_EDGE_THRESHOLDS as T

    def tier(edge, cfg):
        a = abs(edge)
        return "strong" if a >= cfg["strong"] else "lean" if a >= cfg["lean"] else None

    out = {"spread_edge": None, "spread_side": None, "spread_conf": None,
           "total_edge": None, "total_side": None, "total_conf": None}
    if spread_home is not None:
        edge = proj_margin - (-spread_home)
        out.update(spread_edge=round(edge, 1), spread_conf=tier(edge, T["spread"]),
                   spread_side="home" if edge > 0 else "away")
    if total_line is not None:
        edge = proj_total - total_line
        out.update(total_edge=round(edge, 1), total_conf=tier(edge, T["total"]),
                   total_side="over" if edge > 0 else "under")
    return out
