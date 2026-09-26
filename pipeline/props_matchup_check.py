"""
Does splitting a defense's pass-EPA-allowed by the receiver's position (WR vs
TE vs RB) predict a player's next game better than one team-wide number?

Right now the props model adjusts every pass-catcher's efficiency by the same
team-wide "how good is this pass defense" multiplier — a #1 WR and a
checkdown RB get identical matchup treatment. This isolates just that one
question: holding the player's own volume/efficiency baseline fixed (the part
already validated in props_volume_check.py), which matchup signal — team-wide
or position-specific — better predicts what actually happens next game?

Both signals are computed the same simple way (cumulative EPA allowed per
target, same-season, prior weeks only — no opponent adjustment, to isolate
the position-split question from ratings sophistication) so the comparison
is apples to apples.

Usage: python -m pipeline.props_matchup_check
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import HISTORY_SEASONS
from data.nflverse_loader import load_pbp, load_rosters, season_list

TUNE, HOLDOUT = [2022, 2023], [2024, 2025]
MIN_PRIOR_GAMES = 3
POSITION_GROUPS = {"WR", "TE", "RB"}
PRIOR_TARGETS_FOR_SHRINK = 25  # pseudo-targets of league-average shrinkage for defense splits


def build_target_log(pbp: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per target: who, which defense, which of his own position group, EPA, yards."""
    tgt = pbp[
        pbp["receiver_player_id"].notna() & (pbp["pass_attempt"] == 1) & (pbp["season_type"] == "REG")
    ][["season", "week", "game_id", "posteam", "defteam", "receiver_player_id", "epa",
       "complete_pass", "receiving_yards"]].copy()

    ros_slim = rosters[["player_id", "season", "position"]].drop_duplicates(subset=["player_id", "season"])
    tgt = tgt.merge(ros_slim, left_on=["receiver_player_id", "season"], right_on=["player_id", "season"], how="left")
    return tgt[tgt["position"].isin(POSITION_GROUPS)]


def defense_allowed_tables(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (defteam, season, week): mean EPA/target allowed, overall and by position group."""
    team_wide = targets.groupby(["defteam", "season", "week"])["epa"].agg(["mean", "count"]).reset_index()
    by_pos = targets.groupby(["defteam", "season", "week", "position"])["epa"].agg(["mean", "count"]).reset_index()
    return team_wide, by_pos


def _cum_avg(group: pd.DataFrame, prior_week: int) -> tuple[float, int]:
    prior = group[group["week"] < prior_week]
    if prior.empty:
        return 0.0, 0
    n = int(prior["count"].sum())
    return float(np.average(prior["mean"], weights=prior["count"])), n


def evaluate(targets: pd.DataFrame, team_wide: pd.DataFrame, by_pos: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    league_avg_by_pos = by_pos.groupby("position").apply(
        lambda g: np.average(g["mean"], weights=g["count"]), include_groups=False
    ).to_dict()
    league_avg_overall = float(np.average(team_wide["mean"], weights=team_wide["count"]))

    pool = targets[targets["season"].isin(seasons)].sort_values(["receiver_player_id", "week"])
    rows = []
    for (pid, season), g in pool.groupby(["receiver_player_id", "season"]):
        weeks = sorted(g["week"].unique())
        for i, wk in enumerate(weeks):
            if i < MIN_PRIOR_GAMES:
                continue
            prior = g[g["week"] < wk]
            if len(prior) < MIN_PRIOR_GAMES:
                continue
            actual_yards = g[g["week"] == wk]["receiving_yards"].sum()
            n_targets_actual = len(g[g["week"] == wk])
            if n_targets_actual == 0:
                continue
            position = prior["position"].iloc[-1]
            defteam = g[g["week"] == wk]["defteam"].iloc[0]

            # baseline: player's own cumulative yards/target so far (the validated volume x efficiency piece)
            base_ypt = prior["receiving_yards"].sum() / max(len(prior), 1)

            tw = team_wide[team_wide["defteam"] == defteam]
            team_epa, team_n = _cum_avg(tw, wk)
            team_epa = (team_epa * team_n + league_avg_overall * 10) / (team_n + 10)

            bp = by_pos[(by_pos["defteam"] == defteam) & (by_pos["position"] == position)]
            pos_epa, pos_n = _cum_avg(bp, wk)
            prior_pos = league_avg_by_pos.get(position, league_avg_overall)
            pos_epa = (pos_epa * pos_n + prior_pos * PRIOR_TARGETS_FOR_SHRINK) / (pos_n + PRIOR_TARGETS_FOR_SHRINK)

            # same small +/-10% clip the live model uses, just fed a different signal
            team_mult = 1.0 + np.clip((team_epa - league_avg_overall) * 3.0, -0.10, 0.10)
            pos_mult = 1.0 + np.clip((pos_epa - prior_pos) * 3.0, -0.10, 0.10)

            rows.append({
                "season": season, "position": position,
                "pred_team": base_ypt * n_targets_actual * team_mult,
                "pred_position": base_ypt * n_targets_actual * pos_mult,
                "pred_none": base_ypt * n_targets_actual,
                "actual": actual_yards,
            })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n--- {label}: {len(df)} player-games ---")
    for col, name in [("pred_none", "no matchup adj."), ("pred_team", "team-wide matchup"), ("pred_position", "position-specific matchup")]:
        mae = np.mean(np.abs(df[col] - df["actual"]))
        print(f"  {name:<26} MAE {mae:.2f} yds")
    print("  by position:")
    for pos in sorted(df["position"].unique()):
        sub = df[df["position"] == pos]
        t = np.mean(np.abs(sub["pred_team"] - sub["actual"]))
        p = np.mean(np.abs(sub["pred_position"] - sub["actual"]))
        print(f"    {pos:<4} n={len(sub):4}  team-wide {t:.2f}  position-specific {p:.2f}  {'(position better)' if p < t else '(team-wide better)'}")


def main() -> None:
    seasons = season_list(HISTORY_SEASONS)
    print(f"Loading {seasons[0]}-{seasons[-1]} play-by-play + rosters...")
    pbp, rosters = load_pbp(seasons), load_rosters(seasons)
    targets = build_target_log(pbp, rosters)
    team_wide, by_pos = defense_allowed_tables(targets)

    tune_df = evaluate(targets, team_wide, by_pos, TUNE)
    hold_df = evaluate(targets, team_wide, by_pos, HOLDOUT)
    report(tune_df, f"TUNE seasons {TUNE}")
    report(hold_df, f"HOLDOUT seasons {HOLDOUT} (never used to decide anything above)")


if __name__ == "__main__":
    main()
