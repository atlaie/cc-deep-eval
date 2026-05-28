-- schema.sql
-- Engagement-level ledger for the PySyft Mode B endpoints (Phase 1).
--
-- Sits in front of phase3_egress_encoder.ExportLedger. Adds engagement and
-- session accounting without modifying the encoder's per-bundle ledger.
--
-- Concurrency: the Python wrapper sets PRAGMA journal_mode = WAL at connect
-- time. All writes that follow a cap-check go through BEGIN IMMEDIATE so
-- the read-then-insert pair is atomic with respect to concurrent workers.

CREATE TABLE IF NOT EXISTS engagement (
    id              TEXT PRIMARY KEY,                       -- uuid4
    auditor_id      TEXT NOT NULL,                          -- context.user_client.metadata.email
    created_at      REAL NOT NULL,
    token_budget    INTEGER NOT NULL,                       -- engagement-wide tokens_out ceiling
    plot_budget     INTEGER NOT NULL,                       -- engagement-wide plot ceiling
    exemplar_budget INTEGER NOT NULL,                       -- engagement-wide exemplar ceiling
    closed_at       REAL,                                   -- non-NULL once engagement ends
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_engagement_auditor ON engagement(auditor_id);

CREATE TABLE IF NOT EXISTS session (
    id              TEXT PRIMARY KEY,                       -- uuid4
    engagement_id   TEXT NOT NULL REFERENCES engagement(id),
    started_at      REAL NOT NULL,
    ended_at        REAL                                    -- non-NULL on explicit close or cap-trigger
);
CREATE INDEX IF NOT EXISTS idx_session_engagement ON session(engagement_id);

CREATE TABLE IF NOT EXISTS bundle (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES session(id),
    ts              REAL NOT NULL,
    endpoint        TEXT NOT NULL,                          -- e.g. prepilot.capture_residual_stream
    tokens_in       INTEGER NOT NULL,
    tokens_out      INTEGER NOT NULL,
    aggregate_bytes INTEGER NOT NULL,
    bundle_bytes    INTEGER NOT NULL,
    bundle_sha256   TEXT NOT NULL,
    n_plots         INTEGER NOT NULL,
    n_exemplars     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bundle_session ON bundle(session_id);
