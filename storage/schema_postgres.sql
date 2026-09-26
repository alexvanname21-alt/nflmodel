CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,   -- odds API event id
    season        INTEGER,
    week          INTEGER,
    home_team     TEXT,
    away_team     TEXT,
    commence_time TEXT,               -- ISO UTC
    home_score    INTEGER,
    away_score    INTEGER,
    completed     INTEGER DEFAULT 0,
    sport         TEXT DEFAULT 'nfl',  -- 'nfl' | 'cfb'
    home_espn_id  INTEGER,             -- college team ids, for logos
    away_espn_id  INTEGER
);

CREATE TABLE IF NOT EXISTS team_ratings (
    season          INTEGER,
    computed_at     TEXT,
    team            TEXT,
    off_epa_pass    REAL,
    off_epa_rush    REAL,
    def_epa_pass    REAL,
    def_epa_rush    REAL,
    off_epa_total   REAL,
    def_epa_total   REAL,
    net_epa         REAL,
    games_sampled   INTEGER,
    PRIMARY KEY (season, computed_at, team)
);

-- One row per game per pick generation. When we regenerate, we only INSERT
-- a new row if the pick actually changed per the stability rule; otherwise
-- we update last_checked_at on the existing row. That gives a full audit
-- trail of when/why a pick flipped.
CREATE TABLE IF NOT EXISTS game_projections (
    id                  SERIAL PRIMARY KEY,
    game_id             TEXT NOT NULL,
    generated_at        TEXT NOT NULL,
    last_checked_at     TEXT NOT NULL,
    proj_home_score     REAL,
    proj_away_score     REAL,
    proj_margin         REAL,          -- home minus away
    proj_total          REAL,
    market_spread_home  REAL,
    market_total        REAL,
    market_ml_home      REAL,
    market_ml_away      REAL,
    spread_edge         REAL,          -- model margin minus market margin, in points
    total_edge          REAL,          -- model total minus market total, in points
    spread_pick         TEXT,          -- e.g. "KC -3.5"
    total_pick          TEXT,          -- e.g. "OVER 47.5"
    spread_confidence    TEXT,          -- lean / strong
    total_confidence     TEXT,
    weather_json        TEXT,
    injury_note          TEXT,
    line_book            TEXT,          -- book the market numbers came from (fanduel / draftkings)
    is_active            INTEGER DEFAULT 1,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS player_prop_projections (
    id                SERIAL PRIMARY KEY,
    game_id           TEXT NOT NULL,
    generated_at      TEXT NOT NULL,
    last_checked_at   TEXT NOT NULL,
    player_name       TEXT,
    team              TEXT,
    opponent          TEXT,
    market            TEXT,       -- e.g. player_rush_yds
    projection        REAL,
    market_line       REAL,
    edge_pct          REAL,       -- (projection - line) / line
    pick              TEXT,       -- OVER / UNDER
    confidence        TEXT,
    rationale         TEXT,
    is_active         INTEGER DEFAULT 1,
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

CREATE TABLE IF NOT EXISTS closing_lines (
    game_id       TEXT,
    market        TEXT,
    closing_value REAL,
    captured_at   TEXT,
    PRIMARY KEY (game_id, market)
);

CREATE INDEX IF NOT EXISTS idx_proj_game ON game_projections(game_id, is_active);
CREATE INDEX IF NOT EXISTS idx_props_game ON player_prop_projections(game_id, is_active);

-- One row per pick we've actually committed to tracking (strong-confidence
-- only, snapshotted at kickoff so it can never change after the fact), plus
-- its graded result once the game's final. This is the live "is the model
-- actually good" ledger, separate from the backtest.
CREATE TABLE IF NOT EXISTS tracked_picks (
    id                SERIAL PRIMARY KEY,
    sport             TEXT DEFAULT 'nfl',          -- 'nfl' | 'cfb'
    pick_type         TEXT NOT NULL,               -- 'spread' | 'total' | 'prop'
    game_id           TEXT NOT NULL,
    season            INTEGER,
    week              INTEGER,
    kickoff           TEXT,
    description       TEXT,                        -- "Cardinals +5" / "OVER 47.5" / "J.Jefferson OVER 74.5 Receiving Yds"
    player_name       TEXT NOT NULL DEFAULT '',     -- '' for game-level picks (spread/total)
    market            TEXT NOT NULL DEFAULT '',     -- '' for game-level picks
    side              TEXT,                         -- 'home'/'away' (spread), 'over'/'under' (total, props)
    open_line         REAL,                         -- the number when this pick FIRST went strong
    close_line        REAL,                         -- the number at kickoff (what we actually graded against)
    clv               REAL,                         -- open_line vs close_line, in our favor's direction; positive = good
    model_projection  REAL,
    confidence        TEXT,
    locked_at         TEXT NOT NULL,
    result            TEXT NOT NULL DEFAULT 'pending',   -- 'win' | 'loss' | 'push' | 'pending'
    actual_value      REAL,
    graded_at         TEXT,
    UNIQUE(pick_type, game_id, player_name, market)
);

CREATE INDEX IF NOT EXISTS idx_tracked_pending ON tracked_picks(result);
