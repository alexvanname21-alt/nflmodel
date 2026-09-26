"""
NFL + college football model dashboard.

Tabs: Top Plays (Play/Prop of the Week + every strong pick), Slate (every game
this week: projected score + spread/total picks), Player Props (NFL, with
ESPN headshots), Track Record.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from config import CFB_EDGE_THRESHOLDS, EDGE_THRESHOLDS, PICK_STABILITY_THRESHOLD
from data.team_map import fmt_spread, nickname
from models.props_model import MARKET_MIN_ABS_EDGE, evaluate_prop_edge
from storage.db import get_all_active_projections, get_all_active_props, get_conn, get_track_record, init_db

st.set_page_config(page_title="Edge Board", layout="wide", page_icon="🏈")
init_db()

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #060706; --surface: #131513; --surface-2: #0c0d0c; --raised: #1c1f1c;
        --border: #2a2d2a; --border-soft: rgba(255,255,255,0.06);
        --text: #f4f6f4; --text-dim: #9aa19a; --text-faint: #5c625c;
        --accent: #3ee674; --accent-ink: #06170c; --accent-soft: rgba(62,230,116,0.14);
        --green: #3ee674; --green-soft: rgba(62,230,116,0.14);
        --amber: #f0b23e; --amber-soft: rgba(240,178,62,0.15);
        --gold: #f0b23e; --red: #ef4b57; --red-soft: rgba(239,75,87,0.15);
        --blue: #5b9cf0; --blue-soft: rgba(91,156,240,0.14);
        --radius-lg: 16px; --radius: 12px; --radius-sm: 8px;
        --shadow: 0 8px 20px -10px rgba(0,0,0,0.7);
    }
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    h1, h2, h3, .app-title, .section-title, .spotlight-pick, .pc-pick, .bet-pick, .tscore, .stat-item b {
        font-family: 'Sora', 'Inter', sans-serif; }
    .stApp { background: var(--bg); }
    .block-container { padding-top: 1.6rem; max-width: 1540px; }
    h1, h2, h3 { letter-spacing: -0.015em; }
    [data-testid="stSidebar"] { background: var(--surface-2); border-right: 1px solid var(--border-soft); }

    .section-title { font-size: 1.1rem; font-weight: 800; letter-spacing: -0.01em; margin: 2rem 0 0.9rem;
        display: flex; align-items: center; gap: 10px; color: var(--text); }
    .section-title::before { content: ""; width: 4px; height: 18px; border-radius: 2px; background: var(--accent); }
    .section-count { color: var(--text-faint); font-weight: 600; font-size: 0.85rem; font-family: 'Inter', sans-serif; }

    .empty-note { padding: 16px 18px; border-radius: var(--radius); background: var(--surface-2);
        border: 1px dashed var(--border); color: var(--text-dim); font-size: 0.88rem; }

    .stat-row { display: flex; gap: 10px; margin: 0.3rem 0 1.3rem; flex-wrap: wrap; }
    .stat-item { background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius-sm);
        padding: 11px 18px; font-size: 0.78rem; color: var(--text-dim); min-width: 96px; }
    .stat-item b { font-size: 1.2rem; font-weight: 800; display: block; letter-spacing: -0.01em; margin-bottom: 1px; color: var(--text); }

    /* spotlight: play / prop of the week */
    .spotlight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 16px; margin-bottom: 0.4rem; }
    .spotlight-card { position: relative; overflow: hidden; border-radius: var(--radius-lg); padding: 24px 26px;
        background: var(--surface); border: 1px solid var(--border-soft); border-left: 3px solid var(--accent);
        box-shadow: var(--shadow); }
    .spotlight-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
    .spotlight-eyebrow { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em; color: var(--accent);
        text-transform: uppercase; display: flex; align-items: center; gap: 6px; }
    .spotlight-logos { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; }
    .spotlight-logos img { width: 44px; height: 44px; object-fit: contain; background: var(--raised);
        border: 1px solid var(--border-soft); border-radius: 10px; padding: 6px; box-sizing: border-box; }
    .spotlight-avatar { width: 60px; height: 60px; border-radius: 10px; background: var(--surface-2) center top / cover no-repeat;
        border: 1px solid var(--border-soft); margin-bottom: 14px; }
    .spotlight-pick { font-size: 1.65rem; font-weight: 800; color: #fff; line-height: 1.2; margin-bottom: 8px; }
    .spotlight-meta { font-size: 0.84rem; color: var(--text-dim); line-height: 1.6; }
    .spotlight-meta b { color: #d7dad7; }
    .spotlight-grade { position: absolute; top: 22px; right: 24px; }

    /* day header bar */
    .day-bar { display: flex; justify-content: space-between; align-items: center; padding: 11px 18px;
        margin: 1.8rem 0 1rem; border: 1px solid var(--border-soft); border-radius: var(--radius-sm);
        background: var(--surface); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.09em; color: #d3d6d3; }
    .day-bar span:last-child { color: var(--text-faint); }

    /* game cards */
    .slate { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 14px; }
    .game-card { background: var(--surface); border: 1px solid var(--border-soft); border-top: 3px solid var(--border);
        border-radius: var(--radius); padding: 14px; transition: border-color .15s; }
    .game-card.has-strong { border-top-color: var(--green); }
    .gc-head { display: flex; justify-content: space-between; font-size: 0.66rem; font-weight: 700;
        letter-spacing: 0.06em; color: var(--text-faint); margin: 0 2px 10px; }
    .gc-head .tag { color: var(--green); }
    .team-row { display: flex; align-items: center; gap: 10px; padding: 5px 2px; border-radius: 8px; }
    .team-row img { width: 26px; height: 26px; object-fit: contain; flex: none; background: var(--raised);
        border-radius: 7px; padding: 3px; box-sizing: border-box; }
    .tname { flex: 1; font-size: 0.98rem; color: var(--text-dim); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tscore { font-size: 1.3rem; font-weight: 800; color: var(--text-dim); }
    .team-row.win .tname, .team-row.win .tscore { color: var(--text); }

    .bets { margin-top: 12px; border: 1px solid var(--border-soft); border-radius: var(--radius-sm); background: var(--surface-2); overflow: hidden; }
    .bet { display: flex; align-items: center; gap: 10px; padding: 11px 12px; border-top: 1px solid var(--border-soft); }
    .bet:first-child { border-top: none; }
    .chip { min-width: 52px; text-align: center; font-size: 0.6rem; font-weight: 800; letter-spacing: 0.06em;
        padding: 7px 6px; border-radius: 6px; background: var(--raised); color: #c7ccc7; border-left: 3px solid #475549; }
    .chip.spread { border-left-color: var(--accent); }
    .chip.ou { border-left-color: var(--blue); }
    .bet-main { flex: 1; min-width: 0; }
    .bet-pick { font-size: 0.92rem; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .bet-sub { font-size: 0.67rem; color: var(--text-faint); margin-top: 2px; font-family: 'Inter', sans-serif; }
    .grade { text-align: center; font-size: 1.1rem; font-weight: 800; line-height: 1; min-width: 34px; font-family: 'Sora', sans-serif; }
    .grade small { display: block; font-size: 0.48rem; letter-spacing: 0.08em; color: var(--text-faint); margin-top: 3px; font-weight: 700; }
    .g-a { color: var(--green); } .g-b { color: #e2e8f0; } .g-c { color: var(--text-faint); }

    /* pick cards (top plays) and prop cards */
    .card-grid, .prop-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 12px; margin-bottom: 0.5rem; }
    .pick-card, .prop-card { background: var(--surface); border: 1px solid var(--border-soft); border-left: 3px solid var(--border);
        border-radius: var(--radius); padding: 13px 15px; display: flex; align-items: center; gap: 12px;
        transition: border-color .15s; }
    .pick-card:hover, .prop-card:hover { border-color: var(--border); }
    .pick-card.tier-strong, .prop-card.tier-strong { border-left-color: var(--green); }
    .pc-body { flex: 1; min-width: 0; }
    .pc-title { font-size: 0.71rem; color: var(--text-dim); font-weight: 600; letter-spacing: 0.02em; }
    .pc-pick { font-size: 1.03rem; font-weight: 700; color: var(--text); margin: 3px 0 4px; line-height: 1.25; }
    .pc-meta { font-size: 0.71rem; color: var(--text-faint); }
    .pc-meta b { color: #c7ccc7; }
    .imgs { display: flex; flex: none; gap: 5px; }
    .imgs img { width: 34px; height: 34px; object-fit: contain; background: var(--raised);
        border: 1px solid var(--border-soft); border-radius: 9px; padding: 5px; box-sizing: border-box; }
    .avatar { width: 54px; height: 54px; border-radius: 10px; flex: none; background: var(--surface-2) center top / cover no-repeat;
        border: 1px solid var(--border-soft); }

    .badge { display: inline-block; padding: 2px 9px; border-radius: 5px; font-size: 0.66rem; font-weight: 700;
        letter-spacing: 0.03em; white-space: nowrap; }
    .badge-strong { background: var(--green-soft); color: var(--green); }
    .badge-lean { background: var(--amber-soft); color: var(--amber); }

    /* app header */
    .app-header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
        padding: 18px 24px; margin-bottom: 8px; border-radius: var(--radius-lg);
        background: var(--surface); border: 1px solid var(--border-soft); }
    .app-header-left { display: flex; align-items: center; gap: 14px; }
    .app-mark { width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; background: var(--raised); border: 1px solid var(--border-soft); flex: none; }
    .app-title { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.02em; color: var(--text); line-height: 1.2; }
    .app-subtitle { font-size: 0.78rem; color: var(--text-faint); margin-top: 2px; font-family: 'Inter', sans-serif; }
    .app-header-right { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill { display: inline-flex; align-items: center; gap: 6px; padding: 6px 13px; border-radius: 5px;
        font-size: 0.72rem; font-weight: 700; background: var(--raised); color: var(--text-dim); border: 1px solid var(--border-soft); }
    .pill.live::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--green);
        box-shadow: 0 0 0 3px var(--green-soft); }
    .pill.week { color: var(--accent-ink); background: var(--accent); border-color: transparent; font-weight: 800; }

    /* custom warning banner (replaces st.warning to match theme) */
    .warn-banner { display: flex; gap: 12px; padding: 16px 20px; border-radius: var(--radius); margin: 1rem 0 0.2rem;
        background: var(--amber-soft); border: 1px solid rgba(240,178,62,0.35); color: #f6d9a3; font-size: 0.85rem; line-height: 1.55; }
    .warn-banner .icon { font-size: 1.2rem; flex: none; }
    .warn-banner b { color: #fff; }

    /* custom ledger table (Track Record) */
    .ledger-wrap { border: 1px solid var(--border-soft); border-radius: var(--radius); overflow: hidden; margin-bottom: 1.2rem; }
    table.ledger { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
    table.ledger th { text-align: left; background: var(--raised); color: var(--text-faint); font-weight: 700;
        font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; padding: 11px 14px; }
    table.ledger td { padding: 10px 14px; border-top: 1px solid var(--border-soft); color: var(--text-dim); white-space: nowrap; }
    table.ledger tr:hover td { background: rgba(255,255,255,0.02); }
    table.ledger td.pick-col { color: var(--text); font-weight: 600; white-space: normal; }
    table.ledger tr.row-win td:first-child { box-shadow: inset 3px 0 0 var(--green); }
    table.ledger tr.row-loss td:first-child { box-shadow: inset 3px 0 0 var(--red); }
    table.ledger tr.row-push td:first-child { box-shadow: inset 3px 0 0 #9ca3af; }
    table.ledger tr.row-pending td:first-child { box-shadow: inset 3px 0 0 var(--blue); }
    table.ledger .result-chip { display: inline-block; padding: 3px 11px; border-radius: 5px; font-size: 0.7rem; font-weight: 800; letter-spacing: 0.03em; }
    .result-win { background: var(--green-soft); color: var(--green); }
    .result-loss { background: var(--red-soft); color: #f87171; }
    .result-push { background: rgba(156,163,175,0.14); color: #9ca3af; }
    .result-pending { background: var(--blue-soft); color: var(--blue); }

    /* ---- restyle native Streamlit chrome to match ---- */
    .stTabs [role="tablist"] { gap: 4px; background: var(--surface); padding: 5px; border-radius: var(--radius-sm);
        border: 1px solid var(--border-soft); width: fit-content; }
    .stTabs [data-testid="stTab"] { height: 40px; border-radius: 7px; padding: 0 20px; color: var(--text-faint);
        font-weight: 700; font-size: 0.86rem; background: transparent; font-family: 'Sora', sans-serif;
        display: flex; align-items: center; transition: color .15s; }
    .stTabs [data-testid="stTab"][aria-selected="true"] { background: var(--accent) !important; color: var(--accent-ink) !important; }
    .stTabs [data-testid="stTab"][aria-selected="true"] p { color: var(--accent-ink) !important; }
    .stTabs .react-aria-SelectionIndicator { display: none !important; }

    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] { background: var(--surface); border: 1px solid var(--border-soft);
        border-radius: var(--radius-sm); padding: 4px; gap: 0; }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label { flex: 1; justify-content: center; margin: 0;
        padding: 7px 10px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; transition: background .15s; }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:has(input:checked) { background: var(--accent); color: var(--accent-ink); }
    [data-testid="stSidebar"] .stRadio svg { display: none; }

    .stTextInput input, .stMultiSelect [data-baseweb="select"] > div, .stTextInput > div > div {
        background: var(--surface) !important; border-color: var(--border) !important; border-radius: var(--radius-sm) !important; }
    .stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important; }
    [data-baseweb="tag"] { background: var(--raised) !important; border-radius: 5px !important; }
    label p { color: var(--text-dim) !important; font-size: 0.8rem !important; font-weight: 600 !important; }

    [data-testid="stExpander"] { border: 1px solid var(--border-soft) !important; border-radius: var(--radius) !important;
        background: var(--surface); overflow: hidden; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border-soft); border-radius: var(--radius); overflow: hidden; }

    .stButton > button[kind="primary"] { border-radius: var(--radius-sm); font-weight: 800; border: none;
        background: var(--accent) !important; color: var(--accent-ink) !important; }
    .stButton > button[kind="primary"] p { color: var(--accent-ink) !important; }

    [data-testid="stAlert"] { background: var(--amber-soft) !important; border: 1px solid rgba(240,178,62,0.35) !important;
        border-radius: var(--radius) !important; color: #f6d9a3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def parse_utc(iso_str: str) -> datetime:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))


def to_et(iso_str: str) -> datetime:
    return parse_utc(iso_str).astimezone(ET)


def fmt_time(iso_str: str) -> str:
    """'SAT 3:30 PM ET' — Eastern, DST-aware."""
    try:
        dt = to_et(iso_str)
        return f"{dt.strftime('%a').upper()} {dt.strftime('%I:%M %p').lstrip('0')} ET"
    except Exception:
        return iso_str or ""


def day_label(iso_str: str) -> str:
    return to_et(iso_str).strftime("%A, %b %d").upper()


def fmt_signed(x, decimals: int = 1) -> str:
    return "—" if x is None else f"{x:+.{decimals}f}"


NFL_LOGO_CODE = {"LA": "lar", "WAS": "wsh"}


def team_logo(sport: str, name_or_abbr: str, espn_id) -> str | None:
    if sport == "cfb":
        return f"https://a.espncdn.com/i/teamlogos/ncaa/500/{int(espn_id)}.png" if espn_id else None
    code = NFL_LOGO_CODE.get(name_or_abbr, (name_or_abbr or "").lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png"


@st.cache_data(ttl=3600)
def headshot_lookup() -> dict:
    """(player name, team) -> ESPN headshot URL, from the cached nflverse rosters
    (which carry ESPN player ids). Falls back to the roster's own headshot URL."""
    files = sorted(glob(str(Path(__file__).parent / "data" / "cache" / "rosters_*.parquet")))
    if not files:
        return {}
    df = pd.read_parquet(files[-1])
    df = df[df["season"] == df["season"].max()]
    out = {}
    for name, team, espn, url in zip(df["player_name"], df["team"], df["espn_id"], df["headshot_url"]):
        if pd.notna(espn):
            out[(name, team)] = f"https://a.espncdn.com/i/headshots/nfl/players/full/{int(float(espn))}.png"
        elif isinstance(url, str) and url.startswith("http"):
            out[(name, team)] = url
    return out


GRADE_STEPS = [(1.5, "A"), (1.25, "A-"), (1.0, "B+"), (0.8, "B"), (0.6, "B-"), (0.4, "C+"), (0.0, "C")]


def edge_ratio(edge: float | None, strong_at: float) -> float:
    """Edge size in units of that market's 'strong' threshold — puts spreads,
    totals and props (all different scales) on one comparable scale."""
    if edge is None or not strong_at:
        return 0.0
    return abs(edge) / strong_at


def grade(edge: float | None, strong_at: float) -> tuple[str, str]:
    """Letter grade = how large the model's disagreement with the line is,
    relative to that market's 'strong' threshold. NOT a win probability."""
    if edge is None:
        return "—", "g-c"
    r = edge_ratio(edge, strong_at)
    letter = next(l for cut, l in GRADE_STEPS if r >= cut)
    return letter, {"A": "g-a", "B": "g-b", "C": "g-c"}[letter[0]]


PROP_STRONG_AT = 2.0  # a prop is 'strong' at >= 2.0x its market's minimum meaningful edge (models/props_model.py)


def refresh_prop(p: dict) -> dict:
    """Re-evaluate a stored prop with the CURRENT edge logic (so a stale tier can't
    linger) and add 'strength': its edge measured in units of that market's
    minimum meaningful edge. This keeps a +8-yard tweak on a tiny line from
    outranking a real 80-yard gap on a big one."""
    m = re.search(r"sample (\d+) games", p.get("rationale") or "")
    ev = evaluate_prop_edge(p["projection"], p["market_line"], market=p["market"],
                            sample_games=int(m.group(1)) if m else None)
    p.update(pick=ev["pick"], confidence=ev["confidence"], edge_pct=ev["edge_pct"])
    unit = MARKET_MIN_ABS_EDGE.get(p["market"], 1.0) or 1.0
    p["strength"] = abs(p["projection"] - p["market_line"]) / unit
    return p


def current_week_rows(rows: list[dict]) -> tuple[list[dict], int | None]:
    """Only the current week: the earliest week that still has a game to play
    (or the latest week if everything's started)."""
    now = datetime.now(timezone.utc)
    weeks = [r["week"] for r in rows if r.get("week") is not None]
    if not weeks:
        return rows, None
    future = [r["week"] for r in rows if r.get("week") is not None and parse_utc(r["commence_time"]) > now]
    wk = min(future) if future else max(weeks)
    return [r for r in rows if r.get("week") == wk], wk


def render(html_str: str) -> None:
    st.markdown(html_str, unsafe_allow_html=True)


def empty_note(text: str) -> None:
    render(f'<div class="empty-note">{esc(text)}</div>')


def section_title(text: str, count: int | None = None) -> None:
    suffix = f' <span class="section-count">({count})</span>' if count is not None else ""
    render(f'<div class="section-title">{text}{suffix}</div>')


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown('<div style="font-weight:800;font-size:1.05rem;margin-bottom:2px;">🏈 Edge Board</div>', unsafe_allow_html=True)
st.sidebar.caption("An independent model's picks vs. the market — not a guarantee.")

league = st.sidebar.radio("League", ["NFL", "College"], horizontal=True)
IS_CFB = league == "College"
SPORT = "cfb" if IS_CFB else "nfl"
THRESH = (
    {"spread": CFB_EDGE_THRESHOLDS["spread"]["strong"], "total": CFB_EDGE_THRESHOLDS["total"]["strong"]}
    if IS_CFB else {"spread": EDGE_THRESHOLDS["strong"], "total": EDGE_THRESHOLDS["strong"]}
)


BOOK_LABEL = {"fanduel": "FD", "draftkings": "DK"}


def book_tag(p: dict) -> str:
    b = p.get("line_book")
    return f"{BOOK_LABEL.get(b, (b or '').upper())} · " if b else ""


def team_name(abbr: str) -> str:
    return abbr if IS_CFB else nickname(abbr)


def spread_view(p: dict) -> dict | None:
    m, edge = p.get("market_spread_home"), p.get("spread_edge")
    if m is None or edge is None:
        return None
    side = "home" if edge > 0 else "away"
    line = m if side == "home" else -m
    model_line = -p["proj_margin"] if side == "home" else p["proj_margin"]
    g, gcls = grade(edge, THRESH["spread"])
    name = team_name(p["home_team"] if side == "home" else p["away_team"])
    return {"text": f"{name} {fmt_spread(line)}", "sub": f"{book_tag(p)}Model {fmt_spread(round(model_line, 1))} · edge {edge:+.1f}",
            "grade": g, "gcls": gcls, "conf": p.get("spread_confidence"), "side": side, "ratio": edge_ratio(edge, THRESH["spread"])}


def total_view(p: dict) -> dict | None:
    line, edge = p.get("market_total"), p.get("total_edge")
    if line is None or edge is None:
        return None
    g, gcls = grade(edge, THRESH["total"])
    return {"text": f"{'Over' if edge > 0 else 'Under'} {line:g}", "sub": f"{book_tag(p)}Model {p['proj_total']:.1f} · edge {edge:+.1f}",
            "grade": g, "gcls": gcls, "conf": p.get("total_confidence"), "ratio": edge_ratio(edge, THRESH["total"])}


def logo_img(sport: str, p: dict, side: str) -> str:
    abbr = p["home_team"] if side == "home" else p["away_team"]
    url = team_logo(sport, abbr, p.get("home_espn_id") if side == "home" else p.get("away_espn_id"))
    return f'<img src="{url}" alt="" loading="lazy">' if url else ""


def bet_row(kind: str, view: dict | None) -> str:
    chip = f'<span class="chip {"spread" if kind == "SPREAD" else "ou"}">{kind}</span>'
    if view is None:
        return f'<div class="bet">{chip}<div class="bet-main"><div class="bet-pick">—</div></div><div class="grade g-c">—<small>GRADE</small></div></div>'
    return (f'<div class="bet">{chip}<div class="bet-main"><div class="bet-pick">{esc(view["text"])}</div>'
            f'<div class="bet-sub">{esc(view["sub"])}</div></div>'
            f'<div class="grade {view["gcls"]}">{view["grade"]}<small>GRADE</small></div></div>')


def game_card(p: dict, sport: str) -> str:
    sv, tv = spread_view(p), total_view(p)
    has_strong = bool((sv and sv["conf"] == "strong") or (tv and tv["conf"] == "strong"))
    started = parse_utc(p["commence_time"]) <= datetime.now(timezone.utc)
    left = fmt_time(p["commence_time"]) + (" · STARTED" if started else "")
    right = '<span class="tag">STRONG PLAY</span>' if has_strong else (f"WEEK {int(p['week'])}" if p.get("week") else "")
    ah, hh = round(p["proj_away_score"]), round(p["proj_home_score"])

    def trow(side, name, score, win):
        return (f'<div class="team-row {"win" if win else ""}">{logo_img(sport, p, side)}'
                f'<span class="tname">{esc(name)}</span><span class="tscore">{score}</span></div>')

    return (f'<div class="game-card {"has-strong" if has_strong else ""}">'
            f'<div class="gc-head"><span>{left}</span><span>{right}</span></div>'
            f'{trow("away", team_name(p["away_team"]), ah, ah > hh)}{trow("home", team_name(p["home_team"]), hh, hh > ah)}'
            f'<div class="bets">{bet_row("SPREAD", sv)}{bet_row("O/U", tv)}</div></div>')


def pick_card(title: str, pick_text: str, confidence: str | None, meta: str, imgs_html: str = "") -> str:
    tier = "tier-strong" if confidence == "strong" else "tier-lean" if confidence == "lean" else ""
    badge = f'<span class="badge badge-{confidence}">{confidence.upper()}</span>' if confidence in ("strong", "lean") else ""
    lead = f'<div class="imgs">{imgs_html}</div>' if imgs_html else ""
    return (f'<div class="pick-card {tier}">{lead}'
            f'<div class="pc-body"><div class="pc-title">{esc(title)}</div><div class="pc-pick">{esc(pick_text)}</div>'
            f'<div class="pc-meta">{meta}</div></div>{badge}</div>')


def prop_card(p: dict) -> str:
    url = headshot_lookup().get((p["player_name"], p["team"]))
    avatar = f'<div class="avatar" style="background-image:url(\'{url}\')"></div>' if url else '<div class="avatar"></div>'
    market = p["market"].replace("player_", "").replace("_", " ").title()
    g, gcls = grade(p["strength"], PROP_STRONG_AT)
    tier = "tier-strong" if p["confidence"] == "strong" else "tier-lean" if p["confidence"] == "lean" else ""
    return (f'<div class="prop-card {tier}">{avatar}<div class="pc-body">'
            f'<div class="pc-title">{esc(p["player_name"])} · {esc(p["team"])} vs {esc(p["opponent"])} · {fmt_time(p["commence_time"])}</div>'
            f'<div class="pc-pick">{esc(p["pick"] or "—")} {p["market_line"]:g} {esc(market)}</div>'
            f'<div class="pc-meta">Model <b>{p["projection"]}</b> · edge <b>{fmt_signed((p["edge_pct"] or 0) * 100)}%</b></div></div>'
            f'<div class="grade {gcls}">{g}<small>GRADE</small></div></div>')


def spotlight_card(eyebrow: str, logos_html: str, pick_text: str, meta: str, g: str, gcls: str) -> str:
    return (f'<div class="spotlight-card">'
            f'<div class="spotlight-top"><span class="spotlight-eyebrow">🏆 {esc(eyebrow)}</span></div>'
            f'<div class="spotlight-logos">{logos_html}</div>'
            f'<div class="spotlight-pick">{esc(pick_text)}</div>'
            f'<div class="spotlight-meta">{meta}</div>'
            f'<div class="grade {gcls} spotlight-grade">{g}<small>GRADE</small></div>'
            f'</div>')


def find_play_of_week(rows: list[dict]):
    """The single strong pick (spread or total, whichever) with the biggest
    edge relative to its own threshold. None if nothing is strong this week."""
    candidates = []
    for p in rows:
        if p.get("spread_confidence") == "strong":
            candidates.append((spread_view(p)["ratio"], "spread", p))
        if p.get("total_confidence") == "strong":
            candidates.append((total_view(p)["ratio"], "total", p))
    return max(candidates, key=lambda c: c[0]) if candidates else None


def find_prop_of_week(rows: list[dict]) -> dict | None:
    strong = [p for p in rows if p.get("confidence") == "strong"]
    return max(strong, key=lambda p: p["strength"]) if strong else None


if st.sidebar.button("🔄 Refresh picks now", width="stretch", type="primary"):
    with st.spinner("Pulling latest odds and scores..."):
        result = subprocess.run(
            [sys.executable, "-m", "pipeline.cfb_picks" if IS_CFB else "pipeline.generate_picks"],
            capture_output=True, text=True, cwd=str(Path(__file__).parent),
        )
    with st.sidebar.expander("Refresh log"):
        st.code(result.stdout[-3000:] if result.stdout else result.stderr[-3000:])
    st.rerun()

with st.sidebar.expander("How to read this"):
    if IS_CFB:
        st.markdown(
            "**Grade** — how far the model's number is from the line (A = biggest gap). It is not a "
            "win probability. STRONG = gap ≥ 10 pts (spread) or ≥ 6 pts (total) — that's the only "
            "cutoff; every pick that clears it shows up, there's no separate top-N limit.\n\n"
            "**Score predictions** are the big numbers on each game card.\n\n"
            "**Times** are Eastern. **Backtest** — unvalidated; see the banner and README."
        )
    else:
        st.markdown(
            f"**Grade** — how far the model's number is from the line (A = biggest gap). It is not a "
            f"win probability. STRONG = gap ≥ {EDGE_THRESHOLDS['strong']} pts (≥13% for props) — that's "
            f"the only cutoff; every pick that clears it shows up, there's no separate top-N limit.\n\n"
            f"**Pick stability** — a pick only changes once the edge moves more than "
            f"±{PICK_STABILITY_THRESHOLD} pts, or there's fresh injury news.\n\n"
            f"**Times** are Eastern. **Backtest** — the model hasn't shown a proven edge yet; see the README."
        )

# ---------------------------------------------------------------------------
# Data (current week only)
# ---------------------------------------------------------------------------

with get_conn() as conn:
    all_proj = [dict(r) for r in get_all_active_projections(conn) if (r["sport"] or "nfl") == SPORT]
    all_props = [] if IS_CFB else [dict(r) for r in get_all_active_props(conn) if (r["sport"] or "nfl") == "nfl"]
    tracked_all = [dict(r) for r in get_track_record(conn) if (r["sport"] or "nfl") == SPORT]

projections, week_no = current_week_rows(all_proj)
props = [refresh_prop(r) for r in all_props if week_no is None or r.get("week") == week_no]
projections.sort(key=lambda r: r["commence_time"])

last_updated = max((p["last_checked_at"] for p in projections + props if p.get("last_checked_at")), default=None)
updated_str = fmt_time(last_updated) if last_updated else None

pills = ['<span class="pill live">Live</span>']
if week_no is not None:
    pills.append(f'<span class="pill week">{"College " if IS_CFB else "NFL "}Week {int(week_no)}</span>')
if updated_str:
    pills.append(f'<span class="pill">Updated {esc(updated_str)}</span>')
pills.append('<span class="pill">All times ET</span>')

render(
    '<div class="app-header"><div class="app-header-left">'
    f'<div class="app-mark">🏈</div><div><div class="app-title">Edge Board</div>'
    f'<div class="app-subtitle">{"College Football" if IS_CFB else "NFL"} · '
    f'{"Score-based ridge ratings" if IS_CFB else "EPA-based projections"} vs. FanDuel/DraftKings</div></div></div>'
    f'<div class="app-header-right">{"".join(pills)}</div></div>'
)

if IS_CFB:
    render(
        '<div class="warn-banner"><div class="icon">⚠️</div><div><b>College is unvalidated.</b> Backtest on '
        "2022-2025 closing lines (3,000 games): the model's spread picks hit ~50% at every edge size and its "
        "margin error (13.2 pts) is well above the market's (12.0). Totals looked better on 2024-25 (54-57%) "
        "but not on 2022-23, so that's unproven too. Treat score predictions as a second opinion and let the "
        "Track Record decide whether any pick tier is worth betting.</div></div>"
    )

labels = ["🏆 Top Plays", "🗓️ Slate"] + ([] if IS_CFB else ["🏈 Player Props"]) + ["📒 Track Record"]
tabs = st.tabs(labels)
tab_top, tab_slate = tabs[0], tabs[1]
tab_props = None if IS_CFB else tabs[2]
tab_track = tabs[-1]

# ---------------------------------------------------------------------------
# Top Plays (strong only — a fixed edge threshold, not a top-N cap)
# ---------------------------------------------------------------------------

def strong_sorted(rows, conf_key, edge_key):
    picked = [r for r in rows if r.get(conf_key) == "strong"]
    picked.sort(key=lambda r: -abs(r[edge_key] or 0))
    return picked


with tab_top:
    st.caption("Every pick that clears the model's strong-confidence threshold — not a fixed count, just "
               "whatever the model actually likes this week. The model's own ranking, not a validated hit "
               "rate; the Track Record tab has the real results.")

    play_of_week = find_play_of_week(projections)
    prop_of_week = None if IS_CFB else find_prop_of_week(props)

    if play_of_week or prop_of_week:
        section_title("This Week's Best")
        spotlights = []
        if play_of_week:
            _, kind, p = play_of_week
            matchup = f"{team_name(p['away_team'])} @ {team_name(p['home_team'])} · {fmt_time(p['commence_time'])}"
            if kind == "spread":
                sv = spread_view(p)
                spotlights.append(spotlight_card(
                    "Play of the Week", logo_img(SPORT, p, sv["side"]), sv["text"],
                    f"{matchup}<br>{esc(sv['sub'])}", sv["grade"], sv["gcls"]))
            else:
                tv = total_view(p)
                spotlights.append(spotlight_card(
                    "Play of the Week", logo_img(SPORT, p, "away") + logo_img(SPORT, p, "home"), tv["text"].upper(),
                    f"{matchup}<br>{esc(tv['sub'])}", tv["grade"], tv["gcls"]))
        if prop_of_week:
            pw = prop_of_week
            url = headshot_lookup().get((pw["player_name"], pw["team"]))
            avatar = f'<div class="spotlight-avatar" style="background-image:url(\'{url}\')"></div>' if url else '<div class="spotlight-avatar"></div>'
            market = pw["market"].replace("player_", "").replace("_", " ").title()
            g, gcls = grade(pw["strength"], PROP_STRONG_AT)
            spotlights.append(spotlight_card(
                "Prop of the Week", avatar, f"{pw['player_name']} — {pw['pick']} {pw['market_line']:g} {market}",
                f"{pw['team']} vs {pw['opponent']} · {fmt_time(pw['commence_time'])}<br>"
                f"Model <b>{pw['projection']}</b> · edge <b>{fmt_signed((pw['edge_pct'] or 0) * 100)}%</b>",
                g, gcls))
        render(f'<div class="spotlight-grid">{"".join(spotlights)}</div>')

    top_spreads = strong_sorted(projections, "spread_confidence", "spread_edge")
    top_totals = strong_sorted(projections, "total_confidence", "total_edge")
    top_props = strong_sorted(props, "confidence", "strength")

    render('<div class="stat-row">'
           f'<div class="stat-item"><b>{len(top_spreads)}</b>Spread picks</div>'
           f'<div class="stat-item"><b>{len(top_totals)}</b>Total picks</div>'
           + ('' if IS_CFB else f'<div class="stat-item"><b>{len(top_props)}</b>Prop picks</div>') + '</div>')

    section_title("🎯 All Spread Picks")
    if not top_spreads:
        empty_note("No spread picks clear the edge threshold right now.")
    else:
        cards = []
        for p in top_spreads:
            sv = spread_view(p)
            cards.append(pick_card(
                f"{team_name(p['away_team'])} @ {team_name(p['home_team'])} · {fmt_time(p['commence_time'])}",
                sv["text"], "strong", f"{esc(sv['sub'])} · market {fmt_spread(p['market_spread_home'])} (home)",
                logo_img(SPORT, p, sv["side"])))
        render(f'<div class="card-grid">{"".join(cards)}</div>')

    section_title("📈 All Total (O/U) Picks")
    if not top_totals:
        empty_note("No total picks clear the edge threshold right now.")
    else:
        cards = []
        for p in top_totals:
            tv = total_view(p)
            cards.append(pick_card(
                f"{team_name(p['away_team'])} @ {team_name(p['home_team'])} · {fmt_time(p['commence_time'])}",
                tv["text"].upper(), "strong", esc(tv["sub"]),
                logo_img(SPORT, p, "away") + logo_img(SPORT, p, "home")))
        render(f'<div class="card-grid">{"".join(cards)}</div>')

    if not IS_CFB:
        section_title("🏈 All Player Props")
        if not top_props:
            empty_note("No player props clear the edge threshold right now.")
        else:
            render(f'<div class="prop-grid">{"".join(prop_card(p) for p in top_props)}</div>')

# ---------------------------------------------------------------------------
# Slate — every game this week
# ---------------------------------------------------------------------------

with tab_slate:
    if not projections:
        empty_note("No games loaded yet. Click Refresh picks now in the sidebar.")
    else:
        st.caption("Big numbers are the model's projected final score. Grade = size of the model's gap to the line "
                   "(not a win probability). Green top edge = a strong play.")
        by_day: dict[str, list[dict]] = {}
        for p in projections:
            by_day.setdefault(day_label(p["commence_time"]), []).append(p)
        for day, games in by_day.items():
            render(f'<div class="day-bar"><span>{day}</span><span>{len(games)} GAMES</span></div>')
            render(f'<div class="slate">{"".join(game_card(g, SPORT) for g in games)}</div>')

        with st.expander("Table view"):
            rows = [{
                "Kickoff (ET)": fmt_time(p["commence_time"]),
                "Matchup": f"{p['away_team']} @ {p['home_team']}",
                "Model score": f"{p['proj_away_score']:.0f}-{p['proj_home_score']:.0f}",
                "Margin (home)": fmt_signed(p["proj_margin"]), "Total": f"{p['proj_total']:.1f}",
                "Mkt spread (home)": fmt_signed(p["market_spread_home"]), "Mkt total": p["market_total"],
                "Spread edge": fmt_signed(p["spread_edge"]), "Total edge": fmt_signed(p["total_edge"]),
            } for p in projections]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Player props (NFL)
# ---------------------------------------------------------------------------

if tab_props is not None:
    with tab_props:
        if not props:
            empty_note("No prop projections yet for this week.")
        else:
            c1, c2, c3 = st.columns([1.2, 1.6, 1.2])
            with c1:
                tier_choice = st.radio("Show", ["Strong only", "Strong + lean"], horizontal=True, key="prop_tier")
            with c2:
                markets = sorted({p["market"] for p in props})
                market_filter = st.multiselect("Market", markets, default=markets,
                                               format_func=lambda m: m.replace("player_", "").replace("_", " ").title())
            with c3:
                query = st.text_input("Player search", "", placeholder="e.g. Mahomes").strip().lower()

            allowed = {"strong"} if tier_choice == "Strong only" else {"strong", "lean"}
            shown = [p for p in props if p["pick"] and p["confidence"] in allowed and p["market"] in market_filter
                     and (not query or query in p["player_name"].lower())]
            shown.sort(key=lambda p: (0 if p["confidence"] == "strong" else 1, -p["strength"]))
            cap = 90
            st.caption(f"{len(shown)} props" + (f" (showing the top {cap})" if len(shown) > cap else ""))
            if shown:
                render(f'<div class="prop-grid">{"".join(prop_card(p) for p in shown[:cap])}</div>')
            else:
                empty_note("No props match those filters.")

# ---------------------------------------------------------------------------
# Track Record
# ---------------------------------------------------------------------------

RESULT_LABELS = {"win": "WIN", "loss": "LOSS", "push": "PUSH", "pending": "PENDING"}


def summarize(rows: list[dict]) -> dict:
    graded = [t for t in rows if t["result"] != "pending"]
    wins = sum(t["result"] == "win" for t in graded)
    losses = sum(t["result"] == "loss" for t in graded)
    pushes = sum(t["result"] == "push" for t in graded)
    clv = [t["clv"] for t in rows if t["clv"] is not None]
    return {
        "wins": wins, "losses": losses, "pushes": pushes, "pending": len(rows) - len(graded),
        "win_rate": wins / (wins + losses) if (wins + losses) else None,
        "avg_clv": sum(clv) / len(clv) if clv else None,
        "beat_close": (sum(v > 0 for v in clv) / len(clv)) if clv else None,
    }


def stat_bar(s: dict) -> None:
    win_rate = "—" if s["win_rate"] is None else f"{s['win_rate']:.1%}"
    beat_close = "—" if s["beat_close"] is None else f"{s['beat_close']:.0%}"
    render(
        '<div class="stat-row">'
        f'<div class="stat-item"><b>{s["wins"]}-{s["losses"]}-{s["pushes"]}</b>Record</div>'
        f'<div class="stat-item"><b>{win_rate}</b>Win rate</div>'
        f'<div class="stat-item"><b>{s["pending"]}</b>Pending</div>'
        f'<div class="stat-item"><b>{fmt_stat(s["avg_clv"])}</b>Avg CLV</div>'
        f'<div class="stat-item"><b>{beat_close}</b>Beat closing line</div>'
        '</div>'
    )


def fmt_stat(x) -> str:
    return "—" if x is None else f"{x:.1f}"


def ledger_table(rows: list[dict]) -> None:
    head = "".join(f"<th>{h}</th>" for h in ("Kickoff (ET)", "Pick", "Result", "Actual", "Open", "Close", "CLV"))
    body = []
    for t in sorted(rows, key=lambda t: t["kickoff"], reverse=True):
        body.append(
            f"<tr class=\"row-{esc(t['result'])}\"><td>{esc(fmt_time(t['kickoff']))}</td>"
            f'<td class="pick-col">{esc(t["description"])}</td>'
            f'<td><span class="result-chip result-{t["result"]}">{RESULT_LABELS.get(t["result"], t["result"])}</span></td>'
            f"<td>{esc(fmt_stat(t['actual_value']))}</td>"
            f"<td>{esc(fmt_stat(t['open_line']))}</td>"
            f"<td>{esc(fmt_stat(t['close_line']))}</td>"
            f"<td>{esc(fmt_stat(t['clv']))}</td></tr>"
        )
    render(f'<div class="ledger-wrap"><table class="ledger"><thead><tr>{head}</tr></thead>'
           f'<tbody>{"".join(body)}</tbody></table></div>')


with tab_track:
    st.caption("Every strong-confidence pick, locked at kickoff (so it can't be edited after the fact) and graded "
               "once the game's final. This is the real record — separate from the backtest in the README.")
    tracked = tracked_all
    if not tracked:
        empty_note("No picks locked yet — strong picks lock in automatically once their game kicks off.")
    else:
        section_title("Overall")
        stat_bar(summarize(tracked))
        st.caption("CLV: positive means betting the moment a pick went strong would have beaten the line at kickoff.")

        sections = [("🎯 Spreads", "spread"), ("📈 Totals", "total")]
        if not IS_CFB:
            sections.append(("🏈 Player Props", "prop"))

        for label, ptype in sections:
            rows = [t for t in tracked if t["pick_type"] == ptype]
            section_title(label, len(rows))
            if not rows:
                empty_note(f"No {label.split(' ', 1)[1].lower()} picks locked yet.")
                continue
            stat_bar(summarize(rows))
            ledger_table(rows)

st.caption("Educational research tool, not a guarantee of anything. Sports betting involves real financial risk — "
           "size bets within your means. Model edges are estimates, not certainties.")
