"""
Filters out players who shouldn't get a prop projection: on IR/reserve, cut,
retired, or ruled out — plus a manual override list for moves the free public
data hasn't caught up to yet (roster/injury releases update on their own
weekly cadence and can lag a real transaction by a few days).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

INACTIVE_ROSTER_STATUSES = {"RES", "CUT", "RET", "INA", "EXE"}
MANUAL_EXCLUSIONS_PATH = Path(__file__).parent / "excluded_players.txt"


def load_manual_exclusions() -> set[str]:
    if not MANUAL_EXCLUSIONS_PATH.exists():
        return set()
    names = set()
    for line in MANUAL_EXCLUSIONS_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names


def load_unavailable_players(current_season: int) -> set[str]:
    """
    Best-effort automated check: as of the most recently recorded week of the
    CURRENT season, is this player on an inactive roster status or ruled Out?
    Only the current season is queried — nfl_data_py's multi-season
    import_weekly_rosters call has a bug that throws on some historical data
    (duplicate-birthdate reindex error), and we only need the latest status
    anyway. This still lags same-week transactions (a player placed on IR
    today won't show up here until nflverse's next weekly release) — that's
    what excluded_players.txt is for.
    """
    import nfl_data_py as nfl

    names: set[str] = set()

    try:
        rosters = nfl.import_weekly_rosters([current_season])
        latest = rosters.sort_values(["week"]).groupby("player_id").tail(1)
        inactive = latest[latest["status"].isin(INACTIVE_ROSTER_STATUSES)]
        names |= set(inactive["player_name"].dropna())
    except Exception as e:
        print(f"  [warn] could not load weekly rosters for availability check: {e}")

    try:
        injuries = nfl.import_injuries([current_season])
        latest_inj = injuries.sort_values(["week"]).groupby(["gsis_id"], dropna=False).tail(1)
        out = latest_inj[latest_inj["report_status"] == "Out"]
        names |= set(out["full_name"].dropna())
    except Exception as e:
        print(f"  [warn] could not load injuries for availability check: {e}")

    return names
