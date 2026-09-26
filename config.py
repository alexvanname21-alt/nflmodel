"""Project-wide constants and paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# On Streamlit Community Cloud there's no .env file — secrets are set in the
# app's dashboard and only exposed via st.secrets, not the environment. Mirror
# them into os.environ so the rest of the code can keep reading env vars either way.
try:
    import streamlit as st
    for _key, _val in st.secrets.items():
        os.environ.setdefault(_key, str(_val))
except Exception:
    pass

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

DB_PATH = ROOT_DIR / "storage" / "nfl_model.db"

# When set (e.g. on Streamlit Community Cloud, pointed at a free Supabase
# Postgres project), storage/db.py uses Postgres instead of the local SQLite
# file, so the Track Record survives cloud redeploys. Unset = local SQLite,
# unchanged from before.
DATABASE_URL = os.environ.get("DATABASE_URL", "")

# How many past seasons feed the ratings model.
HISTORY_SEASONS = 5

# Within a season, how much more a recent game counts vs an early one
# (exponential decay applied when weighting games into ratings).
INTRA_SEASON_DECAY = 0.96  # per week

# Across seasons, how much last season counts vs two seasons ago, etc.
SEASON_WEIGHTS = {0: 1.0, 1: 0.55, 2: 0.30, 3: 0.18, 4: 0.10}  # 0 = most recent completed season

# League-average points/EPA blended in for small samples (Bayesian prior weight,
# in "pseudo-games").
PRIOR_GAMES = 4

# Edge thresholds for labeling picks in the UI.
EDGE_THRESHOLDS = {
    "lean": 1.0,     # points of edge vs market to call it a "lean"
    "strong": 2.5,   # points of edge to call it a "strong play"
}

# How much of the model's raw projection to trust vs. shrink toward the
# market's own line: final = BLEND_WEIGHT * model + (1 - BLEND_WEIGHT) * market.
# 1.0 = pure raw model (no shrinkage). Chosen by sweeping values against
# pipeline/backtest.py — see README's backtest section for the current value's
# justification. Re-run the sweep and update this if the model changes.
BLEND_WEIGHT = 1.0

# Don't re-flip a published pick unless the edge moves by more than this many
# points, or there's a flagged injury change. Prevents noise-chasing near kickoff.
PICK_STABILITY_THRESHOLD = 1.0

ODDS_API_BASE = "https://api.the-odds-api.com"
NFL_SPORT_KEY = "americanfootball_nfl"

CFB_SPORT_KEY = "americanfootball_ncaaf"

# College edge thresholds, in points. Much wider than the NFL's because the
# college model is far noisier (margin MAE ~13 pts) and college spreads/totals
# are bigger numbers. These are NOT validated to predict anything — the college
# backtest (pipeline/cfb_backtest.py) shows spreads at ~50% ATS at every edge
# size. Tiers exist so a live track record can accumulate and judge them.
CFB_EDGE_THRESHOLDS = {
    "spread": {"lean": 6.0, "strong": 10.0},
    "total": {"lean": 4.0, "strong": 6.0},
}
