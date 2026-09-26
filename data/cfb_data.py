"""
College football data: game results via ESPN's public scoreboard (same team/game
ids as the historical betting-line file), historical closing lines from the
sportsdataverse/cfbfastR-data repo (backtest only), and Odds API <-> ESPN team
name mapping.

There's no free, current college play-by-play (the cfbfastR pbp repo stops at
2021), so the college model is a points-based ridge power rating built from
final scores rather than EPA. That's a deliberate, honest downgrade in signal
vs. the NFL model — see models/cfb_model.py and pipeline/cfb_backtest.py.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache" / "cfb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
BETTING_URL = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/betting/parquet/cfb_line_odds.parquet"
REGULAR_WEEKS = range(1, 17)
POST_WEEKS = range(1, 3)


def cfb_current_season() -> int:
    """College season runs Aug->Jan, so Jan still belongs to the season that started the prior August."""
    today = pd.Timestamp.today()
    return today.year if today.month >= 7 else today.year - 1


def _scoreboard(season: int, seasontype: int, week: int) -> dict:
    r = requests.get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": season, "seasontype": seasontype, "week": week, "groups": 80, "limit": 400},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _parse_events(payload: dict, season: int, seasontype: int, week: int) -> list[dict]:
    rows = []
    for e in payload.get("events", []):
        c = e["competitions"][0]
        home = next((t for t in c["competitors"] if t["homeAway"] == "home"), None)
        away = next((t for t in c["competitors"] if t["homeAway"] == "away"), None)
        if home is None or away is None:
            continue
        completed = bool(c["status"]["type"]["completed"])
        rows.append({
            "game_id": int(e["id"]), "season": season, "seasontype": seasontype, "week": week,
            "start": e["date"], "neutral": bool(c.get("neutralSite")), "completed": completed,
            "home_id": int(home["team"]["id"]), "home_name": home["team"]["displayName"],
            "away_id": int(away["team"]["id"]), "away_name": away["team"]["displayName"],
            "home_points": float(home["score"]) if completed and home.get("score") not in (None, "") else None,
            "away_points": float(away["score"]) if completed and away.get("score") not in (None, "") else None,
        })
    return rows


def load_results(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """One row per FBS-involved game. Fully-completed weeks are cached forever;
    the in-progress current season's incomplete weeks are re-fetched each run."""
    current = cfb_current_season()
    frames = []
    for season in seasons:
        for seasontype, weeks in ((2, REGULAR_WEEKS), (3, POST_WEEKS)):
            for week in weeks:
                cache = CACHE_DIR / f"results_{season}_{seasontype}_{week}.parquet"
                if cache.exists() and not force_refresh:
                    df = pd.read_parquet(cache)
                    stale = season == current and len(df) and not df["completed"].all()
                    if not stale:
                        frames.append(df)
                        continue
                try:
                    rows = _parse_events(_scoreboard(season, seasontype, week), season, seasontype, week)
                except requests.RequestException as e:
                    print(f"  [skip] cfb {season} type{seasontype} wk{week}: {e}")
                    continue
                df = pd.DataFrame(rows)
                if len(df):
                    df.to_parquet(cache)
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates("game_id", keep="last")
    out["start"] = pd.to_datetime(out["start"], utc=True)
    return out.sort_values("start").reset_index(drop=True)


def fbs_team_ids(results: pd.DataFrame, min_games: int = 6) -> set[int]:
    """FBS teams play essentially every game inside ESPN's FBS group; FCS
    opponents show up only 1-3 times a year. Anyone with >= min_games in a
    season is treated as FBS (anyone else gets lumped into one 'non-FBS' team)."""
    long = pd.concat([
        results[["season", "home_id"]].rename(columns={"home_id": "team"}),
        results[["season", "away_id"]].rename(columns={"away_id": "team"}),
    ])
    counts = long.groupby(["season", "team"]).size()
    return set(counts[counts >= min_games].index.get_level_values("team"))


def load_teams() -> pd.DataFrame:
    cache = CACHE_DIR / "teams.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    r = requests.get(f"{ESPN_BASE}/teams", params={"limit": 1000}, timeout=30)
    r.raise_for_status()
    teams = r.json()["sports"][0]["leagues"][0]["teams"]
    df = pd.DataFrame([{
        "team_id": int(t["team"]["id"]), "display_name": t["team"]["displayName"],
        "location": t["team"]["location"], "abbr": t["team"].get("abbreviation", ""),
        "short": t["team"].get("shortDisplayName", ""),
    } for t in teams])
    df.to_parquet(cache)
    return df


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", "and").replace("st.", "state")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Odds API school names that differ from ESPN's short school names.
NAME_ALIASES = {
    "appalachian state": "app state", "nicholls state": "nicholls", "sam houston state": "sam houston",
    "southern mississippi": "southern miss", "connecticut": "uconn", "central florida": "ucf",
    "brigham young": "byu", "southern methodist": "smu", "texas christian": "tcu",
    "florida international": "fiu", "umass": "massachusetts", "louisiana monroe": "ul monroe",
    "texas san antonio": "ut san antonio", "san jose state": "san jose state",
    "houston baptist": "houston christian",
}


def build_name_map(teams: pd.DataFrame) -> dict[str, int]:
    """Odds API names look like 'Texas Longhorns' == ESPN displayName. Keys are
    normalized (accents/punctuation stripped); aliases cover the schools whose
    Odds API name differs from ESPN's ('Appalachian State' vs 'App State')."""
    m = {norm(t.display_name): t.team_id for t in teams.itertuples()}
    by_loc = {norm(t.location): (t.team_id, t.display_name) for t in teams.itertuples()}
    for odds_school, espn_school in NAME_ALIASES.items():
        if espn_school in by_loc:
            team_id, display = by_loc[espn_school]
            # 'Appalachian State Mountaineers' -> mascot is whatever follows the ESPN location
            mascot = norm(display)[len(espn_school):].strip()
            m[f"{odds_school} {mascot}".strip()] = team_id
    return m


def odds_name_to_id(name: str, name_map: dict[str, int]) -> int | None:
    """Exact normalized match, then a conservative fuzzy match. Returns None
    when unsure — callers should skip the game rather than guess."""
    key = norm(name)
    if key in name_map:
        return name_map[key]
    close = difflib.get_close_matches(key, list(name_map), n=2, cutoff=0.9)
    if len(close) == 1 or (len(close) == 2 and difflib.SequenceMatcher(None, key, close[0]).ratio()
                                                 - difflib.SequenceMatcher(None, key, close[1]).ratio() > 0.05):
        return name_map[close[0]]
    return None


# --- historical lines (backtest only) ---------------------------------------

def load_closing_lines(seasons: list[int], teams: pd.DataFrame) -> pd.DataFrame:
    """Consensus (mean across books) closing spread from the HOME team's view
    (standard notation: negative = home favored) and total, per game_id."""
    cache = CACHE_DIR / "betting.parquet"
    if not cache.exists():
        pd.read_parquet(BETTING_URL).to_parquet(cache)
    b = pd.read_parquet(cache)
    b = b[b["season"].isin(seasons) & b["market_type"].isin(["spread", "total"])].copy()
    b["game_id"] = b["game_id"].astype(int)

    # Total: 'over' rows carry the number.
    tot = b[(b.market_type == "total") & (b.abbr.str.lower() == "over")]
    tot = tot.groupby("game_id")["lines"].mean().rename("total_line")

    # Spread: pick the row belonging to the home team, matched by name/abbr against game_desc.
    sp = b[b.market_type == "spread"].copy()
    sp["home_desc"] = sp["game_desc"].str.split("@").str[-1].map(norm)
    sp["abbr_n"] = sp["abbr"].map(norm)
    abbr_by_norm = {}
    for t in teams.itertuples():
        for k in (t.location, t.abbr, t.short):
            abbr_by_norm.setdefault(norm(k), norm(t.location))
    sp["abbr_loc"] = sp["abbr_n"].map(lambda x: abbr_by_norm.get(x, x))
    home_rows = sp[(sp["abbr_n"] == sp["home_desc"]) | (sp["abbr_loc"] == sp["home_desc"])]
    spread = home_rows.groupby("game_id")["lines"].mean().rename("spread_home")
    return pd.concat([spread, tot], axis=1).reset_index()


def current_week_window(now: datetime | None = None) -> tuple[int | None, datetime]:
    """(ESPN week number, end of that week) for the college week containing
    `now`, from ESPN's calendar. Falls back to now + 7 days off-season/bowls."""
    now = now or datetime.now(timezone.utc)
    try:
        r = requests.get(f"{ESPN_BASE}/scoreboard", params={"groups": 80, "limit": 1}, timeout=30)
        r.raise_for_status()
        for block in r.json()["leagues"][0].get("calendar", []):
            for entry in block.get("entries", []):
                start = pd.Timestamp(entry["startDate"]).to_pydatetime()
                end = pd.Timestamp(entry["endDate"]).to_pydatetime()
                if start <= now < end:
                    return int(entry["value"]), end
    except (requests.RequestException, KeyError, ValueError, IndexError):
        pass
    return None, now + pd.Timedelta(days=7)
