"""
engagement_ledger.py — engagement/session/bundle accounting (Phase 1).

Wraps a SQLite DB defined in schema.sql alongside this file. Sits between
PySyft endpoints and the existing phase3_egress_encoder.ExportLedger:

    Endpoint  ──►  EngagementLedger (this file)
                         │
                         ▼
                    /v1/egress_eval (loopback)  ──►  ExportLedger (encoder)

EngagementLedger adds engagement-level caps (auditor's total budget across
sessions) and session scoping. ExportLedger inside the encoder stays
unmodified — per-bundle bytes/sha/plots still recorded there per the brief's
§4 measurements.

Concurrency:
    PRAGMA journal_mode = WAL at connect → readers don't block writers.
    BEGIN IMMEDIATE around any cap-check that precedes an insert → atomic
    vs concurrent workers. The Phase 1 cap-enforcement experiment (M3) and
    concurrent-writes experiment (M5) measure this path; the BEGIN IMMEDIATE
    latency IS one of the quantities under test (column
    `pysyft_ledger_insert_seconds` in `phase3_pysyft_driver` rows).
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_TOKEN_BUDGET = 100_000
DEFAULT_PLOT_BUDGET = 200
DEFAULT_EXEMPLAR_BUDGET = 200

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class EngagementCapExceeded(Exception):
    """Raised when the auditor's engagement budget is exhausted.

    The endpoint layer catches this and returns a structured payload
    `{"error": "engagement_cap_exceeded", "kind": ..., "used": ..., "cap": ...}`
    to the auditor rather than re-raising. PySyft surfaces it as a normal
    return value; the typed marker on the dict lets the driver categorise
    failures cleanly.
    """
    def __init__(self, kind: str, used: int, cap: int, engagement_id: str):
        super().__init__(
            f"engagement cap exceeded: {kind} used={used} cap={cap} "
            f"engagement_id={engagement_id}"
        )
        self.kind = kind
        self.used = used
        self.cap = cap
        self.engagement_id = engagement_id

    def to_payload(self) -> dict:
        return {
            "error": "engagement_cap_exceeded",
            "kind": self.kind,
            "used": self.used,
            "cap": self.cap,
            "engagement_id": self.engagement_id,
        }


@dataclass
class EngagementTotals:
    tokens_in: int
    tokens_out: int
    aggregate_bytes: int
    bundle_bytes: int
    n_plots: int
    n_exemplars: int
    n_bundles: int
    n_sessions: int


class EngagementLedger:
    """SQLite-backed engagement/session/bundle store.

    Thread-safe via check_same_thread=False; each call manages its own
    transaction. Connection is shared across PySyft worker threads
    (orchestra.launch n_consumers > 1).
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.db_path),
            isolation_level=None,           # autocommit; we manage txns explicitly
            check_same_thread=False,
        )
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.executescript(SCHEMA_PATH.read_text())

    def close(self) -> None:
        self.conn.close()

    # ===== engagement ====================================================

    def create_engagement(
        self,
        auditor_id: str,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        plot_budget: int = DEFAULT_PLOT_BUDGET,
        exemplar_budget: int = DEFAULT_EXEMPLAR_BUDGET,
        notes: str = "",
    ) -> str:
        engagement_id = str(uuid.uuid4())
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO engagement (id, auditor_id, created_at, "
                "token_budget, plot_budget, exemplar_budget, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    engagement_id, auditor_id, time.time(),
                    token_budget, plot_budget, exemplar_budget, notes,
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return engagement_id

    def get_or_create_engagement(self, auditor_id: str, **defaults) -> str:
        """Resolve the auditor's most recent open engagement; create with
        defaults if none exists. Used by endpoints when the auditor has not
        been pre-registered (test / dev path; production setup pre-registers
        engagements via a separate provisioning step)."""
        row = self.conn.execute(
            "SELECT id FROM engagement WHERE auditor_id = ? AND closed_at IS NULL "
            "ORDER BY created_at DESC LIMIT 1",
            (auditor_id,),
        ).fetchone()
        if row:
            return row[0]
        return self.create_engagement(auditor_id, **defaults)

    def close_engagement(self, engagement_id: str) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE engagement SET closed_at = ? WHERE id = ?",
                (time.time(), engagement_id),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def engagement_totals(self, engagement_id: str) -> EngagementTotals:
        row = self.conn.execute(
            """
            SELECT COALESCE(SUM(b.tokens_in),       0),
                   COALESCE(SUM(b.tokens_out),      0),
                   COALESCE(SUM(b.aggregate_bytes), 0),
                   COALESCE(SUM(b.bundle_bytes),    0),
                   COALESCE(SUM(b.n_plots),         0),
                   COALESCE(SUM(b.n_exemplars),     0),
                   COUNT(b.id),
                   COUNT(DISTINCT s.id)
            FROM session s
            LEFT JOIN bundle b ON b.session_id = s.id
            WHERE s.engagement_id = ?
            """,
            (engagement_id,),
        ).fetchone()
        return EngagementTotals(
            tokens_in=int(row[0]), tokens_out=int(row[1]),
            aggregate_bytes=int(row[2]), bundle_bytes=int(row[3]),
            n_plots=int(row[4]), n_exemplars=int(row[5]),
            n_bundles=int(row[6]), n_sessions=int(row[7]),
        )

    # ===== session =======================================================

    def end_session(self, session_id: str) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "UPDATE session SET ended_at = ? WHERE id = ?",
                (time.time(), session_id),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def start_session_or_raise(
        self,
        auditor_id: str,
        *,
        engagement_defaults: Optional[dict] = None,
    ) -> tuple[str, str]:
        """Atomic engagement-cap check + session insert.

        Returns (engagement_id, session_id). Raises EngagementCapExceeded
        if any cap is met or exceeded. The whole block is one BEGIN
        IMMEDIATE txn so concurrent workers can't race past the cap.
        """
        defaults = engagement_defaults or {}
        engagement_id = self.get_or_create_engagement(auditor_id, **defaults)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            row = self.conn.execute(
                "SELECT token_budget, plot_budget, exemplar_budget "
                "FROM engagement WHERE id = ?",
                (engagement_id,),
            ).fetchone()
            token_cap, plot_cap, exemplar_cap = (
                int(row[0]), int(row[1]), int(row[2]),
            )

            tot = self.conn.execute(
                """
                SELECT COALESCE(SUM(b.tokens_out),   0),
                       COALESCE(SUM(b.n_plots),      0),
                       COALESCE(SUM(b.n_exemplars),  0)
                FROM session s
                LEFT JOIN bundle b ON b.session_id = s.id
                WHERE s.engagement_id = ?
                """,
                (engagement_id,),
            ).fetchone()
            tokens_used, plots_used, exemplars_used = (
                int(tot[0]), int(tot[1]), int(tot[2]),
            )

            if tokens_used >= token_cap:
                self.conn.execute("ROLLBACK")
                raise EngagementCapExceeded(
                    "tokens", tokens_used, token_cap, engagement_id,
                )
            if plots_used >= plot_cap:
                self.conn.execute("ROLLBACK")
                raise EngagementCapExceeded(
                    "plots", plots_used, plot_cap, engagement_id,
                )
            if exemplars_used >= exemplar_cap:
                self.conn.execute("ROLLBACK")
                raise EngagementCapExceeded(
                    "exemplars", exemplars_used, exemplar_cap, engagement_id,
                )

            session_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO session (id, engagement_id, started_at) "
                "VALUES (?, ?, ?)",
                (session_id, engagement_id, time.time()),
            )
            self.conn.execute("COMMIT")
        except Exception:
            # Best-effort rollback — may already be rolled back by the
            # cap-exceed branches above.
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        return engagement_id, session_id

    # ===== bundle ========================================================

    def record_bundle(
        self,
        session_id: str,
        endpoint: str,
        *,
        tokens_in: int,
        tokens_out: int,
        aggregate_bytes: int,
        bundle_bytes: int,
        bundle_sha256: str,
        n_plots: int,
        n_exemplars: int = 0,
    ) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO bundle "
                "(session_id, ts, endpoint, tokens_in, tokens_out, "
                "aggregate_bytes, bundle_bytes, bundle_sha256, n_plots, "
                "n_exemplars) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id, time.time(), endpoint,
                    tokens_in, tokens_out,
                    aggregate_bytes, bundle_bytes, bundle_sha256,
                    n_plots, n_exemplars,
                ),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
