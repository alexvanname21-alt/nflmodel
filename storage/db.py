from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from config import DATABASE_URL, DB_PATH, PICK_STABILITY_THRESHOLD

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_PATH_PG = Path(__file__).parent / "schema_postgres.sql"

# Postgres (e.g. a free Supabase project) is used instead of local SQLite
# whenever DATABASE_URL is set — see config.py. This keeps the Track Record
# ledger alive across Streamlit Cloud redeploys, which wipe local disk.
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras

_NAMED_PARAM = re.compile(r":(\w+)")


def _to_pg_sql(sql: str) -> str:
    """Translate this file's sqlite-style placeholders (?, :name) to
    psycopg2's (%s, %(name)s), so every query here and in pipeline/track_picks.py
    is written once and runs unmodified on both backends."""
    return _NAMED_PARAM.sub(r"%(\1)s", sql).replace("?", "%s")


class _PGConn:
    """Wraps a psycopg2 connection with the same conn.execute(...) /
    conn.executescript(...) convenience methods sqlite3.Connection provides,
    since every call site here and in track_picks.py relies on those existing
    directly on the connection object, not just on a cursor."""

    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql: str, params=None):
        cur = self._raw.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(_to_pg_sql(sql), params)
        return cur

    def executescript(self, sql: str) -> None:
        self._raw.cursor().execute(sql)

    def commit(self) -> None:
        self._raw.commit()

    def close(self) -> None:
        self._raw.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    if USE_POSTGRES:
        conn = _PGConn(psycopg2.connect(DATABASE_URL))
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        if USE_POSTGRES:
            # A fresh Supabase project always starts from this file with every
            # column already present, so none of the sqlite ALTER-migrations
            # below are needed on that path.
            conn.executescript(SCHEMA_PATH_PG.read_text())
            return

        conn.executescript(SCHEMA_PATH.read_text())
        # CREATE TABLE IF NOT EXISTS won't add columns to tables that already
        # exist on disk, so add the college 'sport' column to older databases.
        for table in ("games", "tracked_picks"):
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
            if "sport" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN sport TEXT DEFAULT 'nfl'")
        proj_cols = [r["name"] for r in conn.execute("PRAGMA table_info(game_projections)")]
        if "line_book" not in proj_cols:
            conn.execute("ALTER TABLE game_projections ADD COLUMN line_book TEXT")
        game_cols = [r["name"] for r in conn.execute("PRAGMA table_info(games)")]
        for col in ("home_espn_id", "away_espn_id"):
            if col not in game_cols:
                conn.execute(f"ALTER TABLE games ADD COLUMN {col} INTEGER")


def upsert_game(conn, game: dict) -> None:
    conn.execute(
        """
        INSERT INTO games (game_id, season, week, home_team, away_team, commence_time,
                            home_score, away_score, completed, sport, home_espn_id, away_espn_id)
        VALUES (:game_id, :season, :week, :home_team, :away_team, :commence_time,
                :home_score, :away_score, :completed, :sport, :home_espn_id, :away_espn_id)
        ON CONFLICT(game_id) DO UPDATE SET
            home_score=excluded.home_score,
            away_score=excluded.away_score,
            completed=excluded.completed,
            commence_time=excluded.commence_time,
            home_espn_id=COALESCE(excluded.home_espn_id, games.home_espn_id),
            away_espn_id=COALESCE(excluded.away_espn_id, games.away_espn_id)
        """,
        {"sport": "nfl", "home_espn_id": None, "away_espn_id": None, **game},
    )


_SAVE_RATINGS_SQL = (
    """
    INSERT INTO team_ratings
        (season, computed_at, team, off_epa_pass, off_epa_rush,
         def_epa_pass, def_epa_rush, off_epa_total, def_epa_total,
         net_epa, games_sampled)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (season, computed_at, team) DO UPDATE SET
        off_epa_pass=excluded.off_epa_pass, off_epa_rush=excluded.off_epa_rush,
        def_epa_pass=excluded.def_epa_pass, def_epa_rush=excluded.def_epa_rush,
        off_epa_total=excluded.off_epa_total, def_epa_total=excluded.def_epa_total,
        net_epa=excluded.net_epa, games_sampled=excluded.games_sampled
    """
    if USE_POSTGRES else
    """
    INSERT OR REPLACE INTO team_ratings
        (season, computed_at, team, off_epa_pass, off_epa_rush,
         def_epa_pass, def_epa_rush, off_epa_total, def_epa_total,
         net_epa, games_sampled)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
)


def save_team_ratings(conn, season: int, ratings_df, computed_at: str | None = None) -> None:
    computed_at = computed_at or now_iso()
    rows = ratings_df.to_dict("records")
    for r in rows:
        conn.execute(
            _SAVE_RATINGS_SQL,
            (
                season, computed_at, r["team"], r["off_epa_pass"], r["off_epa_rush"],
                r["def_epa_pass"], r["def_epa_rush"], r["off_epa_total"], r["def_epa_total"],
                r["net_epa"], r["games_sampled"],
            ),
        )


def get_latest_ratings(conn, season: int):
    row = conn.execute(
        "SELECT MAX(computed_at) as latest FROM team_ratings WHERE season = ?", (season,)
    ).fetchone()
    if row is None or row["latest"] is None:
        return []
    return conn.execute(
        "SELECT * FROM team_ratings WHERE season = ? AND computed_at = ?",
        (season, row["latest"]),
    ).fetchall()


def get_active_projection(conn, game_id: str):
    return conn.execute(
        "SELECT * FROM game_projections WHERE game_id = ? AND is_active = 1", (game_id,)
    ).fetchone()


def upsert_game_projection(conn, proj: dict) -> str:
    """
    Enforces pick stability: only writes a new active row (and deactivates the
    old one) if the edge moved beyond PICK_STABILITY_THRESHOLD or an injury
    note is present. Otherwise just bumps last_checked_at on the existing row.
    Returns "new", "updated", or "unchanged".
    """
    existing = get_active_projection(conn, proj["game_id"])
    ts = now_iso()

    if existing is None:
        proj["generated_at"] = ts
        proj["last_checked_at"] = ts
        _insert_projection(conn, proj)
        return "new"

    spread_moved = abs((proj["spread_edge"] or 0) - (existing["spread_edge"] or 0)) > PICK_STABILITY_THRESHOLD
    total_moved = abs((proj["total_edge"] or 0) - (existing["total_edge"] or 0)) > PICK_STABILITY_THRESHOLD
    injury_change = bool(proj.get("injury_note")) and proj.get("injury_note") != existing["injury_note"]
    # The pick itself flipping (including to/from "no bet") always counts,
    # even if the edge magnitude moved less than the stability threshold —
    # otherwise a stale recommendation can stick around forever once the
    # underlying edge-evaluation logic changes (e.g. a threshold fix) without
    # the raw edge value itself moving enough to trigger a refresh.
    spread_pick_changed = proj["spread_pick"] != existing["spread_pick"]
    total_pick_changed = proj["total_pick"] != existing["total_pick"]

    # A moved market line or a different source book is always recorded as a new
    # version (the pick itself is NOT flipped by this — only the numbers behind it
    # are refreshed). Earlier versions are kept, so the opening line that CLV is
    # measured from is preserved.
    line_changed = (
        proj.get("market_spread_home") != existing["market_spread_home"]
        or proj.get("market_total") != existing["market_total"]
        or proj.get("line_book") != existing["line_book"]
    )

    if spread_moved or total_moved or injury_change or spread_pick_changed or total_pick_changed or line_changed:
        conn.execute(
            "UPDATE game_projections SET is_active = 0 WHERE id = ?", (existing["id"],)
        )
        proj["generated_at"] = ts
        proj["last_checked_at"] = ts
        _insert_projection(conn, proj)
        return "updated"

    conn.execute(
        "UPDATE game_projections SET last_checked_at = ? WHERE id = ?", (ts, existing["id"])
    )
    return "unchanged"


def _insert_projection(conn, p: dict) -> None:
    conn.execute(
        """
        INSERT INTO game_projections
            (game_id, generated_at, last_checked_at, proj_home_score, proj_away_score,
             proj_margin, proj_total, market_spread_home, market_total, market_ml_home,
             market_ml_away, spread_edge, total_edge, spread_pick, total_pick,
             spread_confidence, total_confidence, weather_json, injury_note, line_book, is_active)
        VALUES (:game_id, :generated_at, :last_checked_at, :proj_home_score, :proj_away_score,
                :proj_margin, :proj_total, :market_spread_home, :market_total, :market_ml_home,
                :market_ml_away, :spread_edge, :total_edge, :spread_pick, :total_pick,
                :spread_confidence, :total_confidence, :weather_json, :injury_note, :line_book, 1)
        """,
        p,
    )


def get_active_prop(conn, game_id: str, player_name: str, market: str):
    return conn.execute(
        """SELECT * FROM player_prop_projections
           WHERE game_id = ? AND player_name = ? AND market = ? AND is_active = 1""",
        (game_id, player_name, market),
    ).fetchone()


def upsert_prop_projection(conn, prop: dict) -> str:
    existing = get_active_prop(conn, prop["game_id"], prop["player_name"], prop["market"])
    ts = now_iso()

    if existing is None:
        prop["generated_at"] = ts
        prop["last_checked_at"] = ts
        _insert_prop(conn, prop)
        return "new"

    old_edge_pts = abs((existing["edge_pct"] or 0) * (existing["market_line"] or 1))
    new_edge_pts = abs((prop["edge_pct"] or 0) * (prop["market_line"] or 1))
    # Same reasoning as upsert_game_projection: a pick flipping (including
    # to/from "no bet") always counts, independent of the edge-magnitude
    # stability threshold, so a stale pick can't survive a logic change.
    pick_changed = prop["pick"] != existing["pick"]
    confidence_changed = prop["confidence"] != existing["confidence"]
    if abs(new_edge_pts - old_edge_pts) > PICK_STABILITY_THRESHOLD or pick_changed or confidence_changed:
        conn.execute(
            "UPDATE player_prop_projections SET is_active = 0 WHERE id = ?", (existing["id"],)
        )
        prop["generated_at"] = ts
        prop["last_checked_at"] = ts
        _insert_prop(conn, prop)
        return "updated"

    conn.execute(
        "UPDATE player_prop_projections SET last_checked_at = ? WHERE id = ?", (ts, existing["id"])
    )
    return "unchanged"


def _insert_prop(conn, p: dict) -> None:
    conn.execute(
        """
        INSERT INTO player_prop_projections
            (game_id, generated_at, last_checked_at, player_name, team, opponent,
             market, projection, market_line, edge_pct, pick, confidence, rationale, is_active)
        VALUES (:game_id, :generated_at, :last_checked_at, :player_name, :team, :opponent,
                :market, :projection, :market_line, :edge_pct, :pick, :confidence, :rationale, 1)
        """,
        p,
    )


def get_all_active_projections(conn):
    return conn.execute(
        """SELECT gp.*, g.home_team, g.away_team, g.commence_time, g.week, g.sport,
                  g.home_espn_id, g.away_espn_id
           FROM game_projections gp JOIN games g ON gp.game_id = g.game_id
           WHERE gp.is_active = 1
           ORDER BY g.commence_time"""
    ).fetchall()


def get_all_active_props(conn):
    return conn.execute(
        """SELECT pp.*, g.home_team, g.away_team, g.commence_time, g.week, g.sport
           FROM player_prop_projections pp JOIN games g ON pp.game_id = g.game_id
           WHERE pp.is_active = 1
           ORDER BY g.commence_time, ABS(pp.edge_pct) DESC"""
    ).fetchall()


def deactivate_stale_projections(conn, season: int, week: int, sport: str = 'nfl') -> int:
    """Marks inactive any game_projections row whose game isn't in the
    current target week — otherwise a projection from a week no longer being
    generated (e.g. after the current-week filter shipped, or once that
    week's games are done) lingers as 'active' forever. Returns rows affected."""
    cur = conn.execute(
        """
        UPDATE game_projections SET is_active = 0
        WHERE is_active = 1 AND game_id IN (
            SELECT game_id FROM games WHERE sport = ? AND season = ? AND (week IS NULL OR week != ?)
        )
        """,
        (sport, season, week),
    )
    return cur.rowcount


def deactivate_stale_props(conn, season: int, week: int, sport: str = 'nfl') -> int:
    """Same idea as deactivate_stale_projections, for player props — otherwise
    last week's props never get marked inactive once we stop regenerating them,
    and they keep piling up as 'active' forever."""
    cur = conn.execute(
        """
        UPDATE player_prop_projections SET is_active = 0
        WHERE is_active = 1 AND game_id IN (
            SELECT game_id FROM games WHERE sport = ? AND season = ? AND (week IS NULL OR week != ?)
        )
        """,
        (sport, season, week),
    )
    return cur.rowcount


def deactivate_props_for_players(conn, player_names: set[str]) -> int:
    """Marks inactive any prop row for a player who's now excluded (IR/cut/
    manually excluded) — otherwise a stale row from before they were excluded
    keeps showing up. Returns rows affected."""
    if not player_names:
        return 0
    placeholders = ",".join("?" * len(player_names))
    cur = conn.execute(
        f"""UPDATE player_prop_projections SET is_active = 0
            WHERE is_active = 1 AND player_name IN ({placeholders})""",
        tuple(player_names),
    )
    return cur.rowcount


# ---------------------------------------------------------------------------
# Pick tracking / grading — the live "is the model actually good" ledger.
# ---------------------------------------------------------------------------

_LOCK_PICK_SQL = (
    """
    INSERT INTO tracked_picks
        (sport, pick_type, game_id, season, week, kickoff, description, player_name,
         market, side, open_line, close_line, clv, model_projection, confidence, locked_at)
    VALUES (:sport, :pick_type, :game_id, :season, :week, :kickoff, :description, :player_name,
            :market, :side, :open_line, :close_line, :clv, :model_projection, :confidence, :locked_at)
    ON CONFLICT (pick_type, game_id, player_name, market) DO NOTHING
    """
    if USE_POSTGRES else
    """
    INSERT OR IGNORE INTO tracked_picks
        (sport, pick_type, game_id, season, week, kickoff, description, player_name,
         market, side, open_line, close_line, clv, model_projection, confidence, locked_at)
    VALUES (:sport, :pick_type, :game_id, :season, :week, :kickoff, :description, :player_name,
            :market, :side, :open_line, :close_line, :clv, :model_projection, :confidence, :locked_at)
    """
)


def lock_pick(conn, pick: dict) -> bool:
    """Snapshots a pick into the permanent tracked_picks ledger. No-ops if
    this exact pick (type + game + player + market) is already locked, since
    the UNIQUE constraint + INSERT-if-absent makes this safe to call
    repeatedly. Returns True if a new row was actually inserted."""
    cur = conn.execute(_LOCK_PICK_SQL, {"sport": "nfl", **pick})
    return cur.rowcount > 0


def get_pending_tracked_picks(conn):
    return conn.execute("SELECT * FROM tracked_picks WHERE result = 'pending'").fetchall()


def grade_tracked_pick(conn, pick_id: int, result: str, actual_value: float | None) -> None:
    conn.execute(
        "UPDATE tracked_picks SET result = ?, actual_value = ?, graded_at = ? WHERE id = ?",
        (result, actual_value, now_iso(), pick_id),
    )


def get_track_record(conn):
    """All tracked picks, most recent first, for the dashboard ledger."""
    return conn.execute(
        "SELECT * FROM tracked_picks ORDER BY kickoff DESC, id DESC"
    ).fetchall()
