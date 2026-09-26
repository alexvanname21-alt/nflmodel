"""
Walk-forward backtest of the game model's spread/total picks against real
historical closing lines.

For every test-set game, ratings and the EPA->points calibration are rebuilt
using ONLY play-by-play from strictly before that game — same data the model
would have actually had at the time. No lookahead. That's what makes this an
honest test rather than curve-fitting: a model that "predicts" games it was
trained on will always look great, which is exactly the trap to avoid here.

Market lines come from nflverse's schedule data (spread_line/total_line),
which carries real historical closing lines for every game — no paid
historical-odds subscription required.

Usage: python -m pipeline.backtest
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import EDGE_THRESHOLDS, HISTORY_SEASONS
from config import PRIOR_GAMES as RATINGS_PRIOR_GAMES
from data.nflverse_loader import current_season, load_pbp, load_schedules
from models.game_model import calibrate_points_model_from_data, evaluate_spread_edge, evaluate_total_edge, project_game
from models.ratings import compute_team_ratings_from_pbp

# How many recent seasons to grade picks in. Each test season additionally
# needs HISTORY_SEASONS of prior data loaded so ratings aren't built on air.
TEST_SEASONS_COUNT = 3
LOAD_BUFFER_SEASONS = HISTORY_SEASONS + TEST_SEASONS_COUNT
BREAKEVEN_WIN_RATE = 0.5238  # what you need to beat -110 vig long-run

BLEND_SWEEP = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
# In effective (recency-weighted) games now, not raw game count — see
# models/ratings.py's _shrink(). RATINGS_PRIOR_GAMES is the current default.
PRIOR_GAMES_SWEEP = [RATINGS_PRIOR_GAMES, 8, 15, 25, 40, 60]


def _seasons_window(anchor_season: int, n: int) -> list[int]:
    return list(range(anchor_season - n + 1, anchor_season + 1))


def _standard_spread(nflverse_spread_line: float) -> float:
    """nflverse's spread_line convention is the OPPOSITE of standard
    sportsbook notation: positive means the HOME team is favored (verified
    empirically — spread_line correlates positively with actual home margin).
    Flip it to standard notation (negative = home favored)."""
    return -nflverse_spread_line


@dataclass
class TierResult:
    wins: int = 0
    losses: int = 0
    pushes: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.total if self.total else None


@dataclass
class BacktestReport:
    spread_by_tier: dict = field(default_factory=lambda: {"lean": TierResult(), "strong": TierResult()})
    total_by_tier: dict = field(default_factory=lambda: {"lean": TierResult(), "strong": TierResult()})
    model_margin_errors: list = field(default_factory=list)
    market_margin_errors: list = field(default_factory=list)
    model_total_errors: list = field(default_factory=list)
    market_total_errors: list = field(default_factory=list)
    games_graded: int = 0
    rows: list = field(default_factory=list)


def _grade_spread(proj_margin: float, nflverse_spread_line: float, actual_margin: float) -> tuple[str, str | None]:
    standard_spread_home = _standard_spread(nflverse_spread_line)
    market_margin = -standard_spread_home  # = nflverse_spread_line
    eval_ = evaluate_spread_edge(proj_margin, standard_spread_home)
    if eval_["pick"] is None:
        return "no_bet", None

    cover_diff = actual_margin - market_margin
    if abs(cover_diff) < 1e-9:
        return "push", eval_["confidence"]
    home_covered = cover_diff > 0
    correct = home_covered if eval_["pick"] == "home" else not home_covered
    return ("win" if correct else "loss"), eval_["confidence"]


def _grade_total(proj_total: float, total_line: float, actual_total: float) -> tuple[str, str | None]:
    eval_ = evaluate_total_edge(proj_total, total_line)
    if eval_["pick"] is None:
        return "no_bet", None

    diff = actual_total - total_line
    if abs(diff) < 1e-9:
        return "push", eval_["confidence"]
    went_over = diff > 0
    correct = went_over if eval_["pick"] == "over" else not went_over
    return ("win" if correct else "loss"), eval_["confidence"]


def run_backtest(
    test_seasons: list[int],
    blend_weight: float = 1.0,
    ratings_prior_games: float = RATINGS_PRIOR_GAMES,
    pbp: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
    verbose: bool = True,
) -> BacktestReport:
    """
    Grades the model (with the given blend_weight shrinkage and ratings
    shrinkage strength applied) against every completed game in test_seasons.
    Pass pbp/schedules in to reuse an already-loaded wide window across
    multiple calls (e.g. a sweep) instead of re-fetching from disk/cache
    each time.
    """
    if pbp is None or schedules is None:
        anchor = max(test_seasons)
        load_seasons = _seasons_window(anchor, HISTORY_SEASONS + (anchor - min(test_seasons) + 1))
        if verbose:
            print(f"Loading {load_seasons[0]}-{load_seasons[-1]} play-by-play and schedules...")
        pbp = load_pbp(load_seasons)
        schedules = load_schedules(load_seasons)

    test_games = schedules[
        (schedules["game_type"] == "REG")
        & (schedules["season"].isin(test_seasons))
        & schedules["home_score"].notna()
        & schedules["spread_line"].notna()
        & schedules["total_line"].notna()
    ].sort_values(["season", "week"])

    if verbose:
        print(f"Grading {len(test_games)} completed games across seasons {test_seasons}...\n")

    report = BacktestReport()
    cutoffs = test_games[["season", "week"]].drop_duplicates().itertuples(index=False)

    for season, week in cutoffs:
        prior_pbp = pbp[(pbp["season"] < season) | ((pbp["season"] == season) & (pbp["week"] < week))]
        if prior_pbp.empty:
            continue

        ratings_seasons = _seasons_window(season if week > 1 else season - 1, HISTORY_SEASONS)
        cal_pbp = prior_pbp[prior_pbp["season"].isin(ratings_seasons)]
        cal_schedules = schedules[
            (schedules["season"].isin(ratings_seasons))
            & ((schedules["season"] < season) | ((schedules["season"] == season) & (schedules["week"] < week)))
        ]

        ratings_df = compute_team_ratings_from_pbp(cal_pbp, ratings_prior_games)
        if ratings_df.empty:
            continue
        calib = calibrate_points_model_from_data(cal_pbp, cal_schedules)

        week_games = test_games[(test_games["season"] == season) & (test_games["week"] == week)]
        for g in week_games.itertuples():
            try:
                proj = project_game(
                    g.home_team, g.away_team, ratings_df, calib,
                    home_rest_days=int(g.home_rest) if pd.notna(g.home_rest) else None,
                    away_rest_days=int(g.away_rest) if pd.notna(g.away_rest) else None,
                    is_divisional=bool(g.div_game), weather=None,  # historical weather not reconstructed
                    market_spread_home=_standard_spread(g.spread_line),
                    market_total=g.total_line,
                    blend_weight=blend_weight,
                )
            except ValueError:
                continue  # team missing from ratings (rare early-window edge case)

            actual_margin = g.home_score - g.away_score
            actual_total = g.home_score + g.away_score

            report.model_margin_errors.append(abs(proj["proj_margin"] - actual_margin))
            report.market_margin_errors.append(abs(g.spread_line - actual_margin))  # nflverse: spread_line IS the home margin
            report.model_total_errors.append(abs(proj["proj_total"] - actual_total))
            report.market_total_errors.append(abs(g.total_line - actual_total))
            report.games_graded += 1

            spread_result, spread_conf = _grade_spread(proj["proj_margin"], g.spread_line, actual_margin)
            total_result, total_conf = _grade_total(proj["proj_total"], g.total_line, actual_total)

            if spread_result in ("win", "loss") and spread_conf:
                tier = report.spread_by_tier[spread_conf]
                setattr(tier, "wins" if spread_result == "win" else "losses",
                        getattr(tier, "wins" if spread_result == "win" else "losses") + 1)
            elif spread_result == "push" and spread_conf:
                report.spread_by_tier[spread_conf].pushes += 1

            if total_result in ("win", "loss") and total_conf:
                tier = report.total_by_tier[total_conf]
                setattr(tier, "wins" if total_result == "win" else "losses",
                        getattr(tier, "wins" if total_result == "win" else "losses") + 1)
            elif total_result == "push" and total_conf:
                report.total_by_tier[total_conf].pushes += 1

            report.rows.append({
                "season": season, "week": week, "matchup": f"{g.away_team}@{g.home_team}",
                "proj_margin": proj["proj_margin"], "spread_line": g.spread_line, "actual_margin": actual_margin,
                "spread_result": spread_result, "spread_conf": spread_conf,
                "proj_total": proj["proj_total"], "total_line": g.total_line, "actual_total": actual_total,
                "total_result": total_result, "total_conf": total_conf,
            })

        if verbose and week == 1:
            print(f"  ...season {season} done")

    return report


def overall_spread_win_rate(report: BacktestReport) -> tuple[int, int, float | None]:
    wins = sum(report.spread_by_tier[t].wins for t in ("strong", "lean"))
    losses = sum(report.spread_by_tier[t].losses for t in ("strong", "lean"))
    return wins, losses, (wins / (wins + losses) if (wins + losses) else None)


def overall_total_win_rate(report: BacktestReport) -> tuple[int, int, float | None]:
    wins = sum(report.total_by_tier[t].wins for t in ("strong", "lean"))
    losses = sum(report.total_by_tier[t].losses for t in ("strong", "lean"))
    return wins, losses, (wins / (wins + losses) if (wins + losses) else None)


def print_report(report: BacktestReport) -> None:
    print(f"\n{'='*60}\nBACKTEST RESULTS — {report.games_graded} games graded\n{'='*60}\n")

    print("SPREAD picks vs. closing line, by confidence tier:")
    for tier_name in ("strong", "lean"):
        t = report.spread_by_tier[tier_name]
        wr = f"{t.win_rate:.1%}" if t.win_rate is not None else "n/a"
        edge_vs_breakeven = f"({(t.win_rate - BREAKEVEN_WIN_RATE):+.1%} vs breakeven)" if t.win_rate is not None else ""
        print(f"  {tier_name:7s}: {t.wins}-{t.losses}-{t.pushes} ({t.total} decided)  win rate {wr} {edge_vs_breakeven}")

    print("\nTOTAL (O/U) picks vs. closing line, by confidence tier:")
    for tier_name in ("strong", "lean"):
        t = report.total_by_tier[tier_name]
        wr = f"{t.win_rate:.1%}" if t.win_rate is not None else "n/a"
        edge_vs_breakeven = f"({(t.win_rate - BREAKEVEN_WIN_RATE):+.1%} vs breakeven)" if t.win_rate is not None else ""
        print(f"  {tier_name:7s}: {t.wins}-{t.losses}-{t.pushes} ({t.total} decided)  win rate {wr} {edge_vs_breakeven}")

    print(f"\nPrediction error (lower is better), model vs. the market itself:")
    print(f"  Margin MAE  — model: {np.mean(report.model_margin_errors):.2f} pts   market: {np.mean(report.market_margin_errors):.2f} pts")
    print(f"  Total MAE   — model: {np.mean(report.model_total_errors):.2f} pts   market: {np.mean(report.market_total_errors):.2f} pts")
    print(
        "\n(Breakeven against standard -110 vig is 52.4% — a tier needs to clear that, "
        "with a real sample size, to be worth betting. The market MAE numbers show how hard "
        "the market is to beat on raw prediction error, independent of betting results.)"
    )


def _sweep_and_validate(
    label: str,
    param_name: str,
    sweep_values: list[float],
    baseline_value: float,
    tune_seasons: list[int],
    holdout_seasons: list[int],
    pbp: pd.DataFrame,
    schedules: pd.DataFrame,
    fixed_kwargs: dict,
) -> tuple[float, BacktestReport, BacktestReport]:
    """Sweeps one hyperparameter on tune_seasons, validates the winner on
    holdout_seasons against the baseline value. Returns (best_value,
    baseline_holdout_report, tuned_holdout_report)."""
    print(f"\nTuning {label} on seasons {tune_seasons} (holding out {holdout_seasons})...\n")
    print(f"{param_name:>16}  {'spread W-L':>12}  {'spread win%':>12}  {'total W-L':>12}  {'total win%':>11}")

    sweep_results = []
    for val in sweep_values:
        kwargs = {**fixed_kwargs, param_name: val}
        r = run_backtest(tune_seasons, pbp=pbp, schedules=schedules, verbose=False, **kwargs)
        sw, sl, swr = overall_spread_win_rate(r)
        tw, tl, twr = overall_total_win_rate(r)
        sweep_results.append((val, swr, r))
        swr_s = f"{swr:.1%}" if swr is not None else "n/a"
        twr_s = f"{twr:.1%}" if twr is not None else "n/a"
        print(f"{val:>16.1f}  {f'{sw}-{sl}':>12}  {swr_s:>12}  {f'{tw}-{tl}':>12}  {twr_s:>11}")

    best_val, best_swr, _ = max(sweep_results, key=lambda x: x[1] if x[1] is not None else -1)
    print(f"\nBest on tune set: {param_name}={best_val} (spread win rate {best_swr:.1%})")

    print(f"\n--- holdout baseline ({param_name}={baseline_value}) ---")
    baseline_report = run_backtest(
        holdout_seasons, pbp=pbp, schedules=schedules,
        **{**fixed_kwargs, param_name: baseline_value},
    )
    print_report(baseline_report)

    print(f"\n--- holdout tuned ({param_name}={best_val}) ---")
    tuned_report = run_backtest(
        holdout_seasons, pbp=pbp, schedules=schedules,
        **{**fixed_kwargs, param_name: best_val},
    )
    print_report(tuned_report)

    base_sw, base_sl, base_swr = overall_spread_win_rate(baseline_report)
    tuned_sw, tuned_sl, tuned_swr = overall_spread_win_rate(tuned_report)
    print(f"\nHoldout spread win rate — baseline ({param_name}={baseline_value}): "
          f"{f'{base_swr:.1%}' if base_swr is not None else 'n/a'} ({base_sw}-{base_sl})")
    print(f"Holdout spread win rate — tuned ({param_name}={best_val}): "
          f"{f'{tuned_swr:.1%}' if tuned_swr is not None else 'n/a'} ({tuned_sw}-{tuned_sl})")
    if tuned_swr is not None and base_swr is not None:
        if tuned_swr > base_swr:
            print(f"-> {label} HELPED on held-out data (+{(tuned_swr-base_swr):.1%}). "
                  f"Recommend setting config.{param_name if param_name != 'ratings_prior_games' else 'PRIOR_GAMES'} = {best_val}.")
        else:
            print(f"-> {label} did NOT clearly help on held-out data ({(tuned_swr-base_swr):+.1%}). Leave config as-is.")

    return best_val, baseline_report, tuned_report


def main():
    anchor = current_season()
    test_seasons_all = _seasons_window(anchor, TEST_SEASONS_COUNT)
    tune_seasons, holdout_seasons = test_seasons_all[:-1], test_seasons_all[-1:]

    load_seasons = _seasons_window(anchor, LOAD_BUFFER_SEASONS)
    print(f"Loading {load_seasons[0]}-{load_seasons[-1]} play-by-play and schedules (once, reused for the whole run)...")
    pbp = load_pbp(load_seasons)
    schedules = load_schedules(load_seasons)

    print(f"\n{'='*72}\nSWEEP 1: blend_weight (shrink model margin toward market)\n{'='*72}")
    _sweep_and_validate(
        "Market-blending", "blend_weight", BLEND_SWEEP, 1.0,
        tune_seasons, holdout_seasons, pbp, schedules,
        fixed_kwargs={"ratings_prior_games": RATINGS_PRIOR_GAMES},
    )

    print(f"\n{'='*72}\nSWEEP 2: ratings_prior_games (shrinkage strength on team ratings)\n{'='*72}")
    best_pg, baseline_report, tuned_report = _sweep_and_validate(
        "Stronger ratings shrinkage", "ratings_prior_games", PRIOR_GAMES_SWEEP, RATINGS_PRIOR_GAMES,
        tune_seasons, holdout_seasons, pbp, schedules,
        fixed_kwargs={"blend_weight": 1.0},
    )

    if tuned_report.rows:
        pd.DataFrame(tuned_report.rows).to_csv("backtest_results_holdout_tuned.csv", index=False)
    if baseline_report.rows:
        pd.DataFrame(baseline_report.rows).to_csv("backtest_results_holdout_baseline.csv", index=False)
    print("\nHoldout game-by-game results (ratings_prior_games sweep) saved to "
          "backtest_results_holdout_tuned.csv / _baseline.csv")


if __name__ == "__main__":
    main()
