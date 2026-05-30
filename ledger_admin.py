#!/usr/bin/env python3
"""
ledger_admin.py — runtime engagement-budget control for the live Datasite.

Registers `diagnostic.ledger_admin`, a worker-side admin endpoint that adjusts
engagement budgets directly on the container's SQLite ledger. This is the
runtime equivalent of redeploying with different budget env vars — used to set
tight caps for M3 (cap-enforcement test) and to restore generous caps for the
measurement matrix, all on one deploy. It mirrors the real governance flow:
the data owner sets/raises an auditor's budget.

Two modes:
  - WRITE (default): set budgets, optionally reset usage.
  - READ-ONLY (--read-only, or endpoint kwarg read_only=True): return the
    current budgets + usage WITHOUT mutating anything. M5's --check-budget
    uses this so reading the ledger never clobbers the caps under test.

The read payload also includes a PER-ENGAGEMENT usage breakdown
(`per_engagement`), so callers can verify budget ISOLATION (auditor A's
accrual stays under A's own cap; A cannot spend B's budget) and not just the
global invariant.

Usage:
    # M3: reset usage, tight token cap (trips after ~3 calls)
    DATASITE_URL=https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \
        python ledger_admin.py --token-budget 80 --plot-budget 1000 \
            --exemplar-budget 1000 --reset-usage

    # Restore generous caps before the matrix:
    DATASITE_URL=... python ledger_admin.py --token-budget 1000000 \
        --plot-budget 100000 --exemplar-budget 100000 --reset-usage

    # Read-only snapshot (does NOT change budgets or usage):
    DATASITE_URL=... python ledger_admin.py --read-only

The endpoint persists after first run; subsequent invocations just call it
(re-register is idempotent).
"""
import argparse
import os
import sys

import syft as sy


# Worker-side admin body. No annotated params (avoids the 0.9.5 signature
# validator); defined in a file so inspect.getsource works.
#
# read_only=True short-circuits all writes and returns the current snapshot,
# including a per-engagement usage breakdown. When read_only=False the
# token/plot/exemplar budgets are applied as before.
@sy.api_endpoint_method()
def _ledger_admin(context, token_budget=100000, plot_budget=1000,
                  exemplar_budget=1000, reset_usage=False, read_only=False):
    import sqlite3
    DB = "/workspace/engagement_ledger.sqlite"
    conn = sqlite3.connect(DB)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        before = conn.execute(
            "SELECT id, token_budget, plot_budget, exemplar_budget "
            "FROM engagement").fetchall() if "engagement" in tables else []

        if not read_only:
            if reset_usage:
                # Zero cumulative usage by clearing bundle + session rows.
                for t in ("bundle", "session"):
                    if t in tables:
                        conn.execute("DELETE FROM %s" % t)
            conn.execute(
                "UPDATE engagement SET token_budget=?, plot_budget=?, "
                "exemplar_budget=?",
                (int(token_budget), int(plot_budget), int(exemplar_budget)),
            )
            conn.commit()

        after = conn.execute(
            "SELECT id, token_budget, plot_budget, exemplar_budget "
            "FROM engagement").fetchall() if "engagement" in tables else []

        # Global usage (sum across all bundles / all engagements).
        usage = conn.execute(
            "SELECT COALESCE(SUM(b.tokens_out),0), COALESCE(SUM(b.n_plots),0) "
            "FROM bundle b").fetchone() if "bundle" in tables else (0, 0)

        # Per-engagement usage breakdown: join engagement -> session -> bundle
        # so we can verify per-engagement budget ISOLATION, not just the global
        # invariant. Each row: [engagement_id, auditor_id, token_budget,
        # tokens_out_used, plots_used, exemplars_used, n_sessions, n_bundles].
        per_engagement = []
        if {"engagement", "session", "bundle"} <= set(tables):
            per_engagement = [list(r) for r in conn.execute(
                """
                SELECT e.id, e.auditor_id, e.token_budget,
                       COALESCE(SUM(b.tokens_out), 0),
                       COALESCE(SUM(b.n_plots), 0),
                       COALESCE(SUM(b.n_exemplars), 0),
                       COUNT(DISTINCT s.id),
                       COUNT(b.id)
                FROM engagement e
                LEFT JOIN session s ON s.engagement_id = e.id
                LEFT JOIN bundle  b ON b.session_id = s.id
                GROUP BY e.id, e.auditor_id, e.token_budget
                """
            ).fetchall()]

        return {
            "ok": True,
            "read_only": bool(read_only),
            "tables": tables,
            "engagements_before": [list(r) for r in before],
            "engagements_after": [list(r) for r in after],
            "reset_usage": bool(reset_usage) and not read_only,
            "tokens_out_used_now": int(usage[0]),
            "plots_used_now": int(usage[1]),
            "per_engagement": per_engagement,
        }
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasite-url", default=os.environ.get("DATASITE_URL"))
    ap.add_argument("--token-budget", type=int, default=80)
    ap.add_argument("--plot-budget", type=int, default=1000)
    ap.add_argument("--exemplar-budget", type=int, default=1000)
    ap.add_argument("--reset-usage", action="store_true")
    ap.add_argument("--read-only", action="store_true",
                    help="return the current budgets + usage snapshot without "
                         "mutating anything")
    args = ap.parse_args()
    if not args.datasite_url:
        print("set --datasite-url or DATASITE_URL", file=sys.stderr)
        return 2

    client = sy.login(url=args.datasite_url, port=443,
                      email="info@openmined.org", password="changethis")

    # PySyft pins an endpoint's signature from the body it was FIRST registered
    # with. If the live endpoint predates a body change (e.g. the read_only /
    # per_engagement patch), calling it with the new kwarg fails arg validation
    # before the body runs. So we delete-then-register rather than skip-if-
    # present, guaranteeing the deployed body matches this file. (Re-running is
    # safe: delete is best-effort, register is unconditional.)
    PATH = "diagnostic.ledger_admin"
    existing = {getattr(v, "path", None) for v in client.custom_api.api_endpoints()}
    if PATH in existing:
        for attempt in (
            lambda: client.api.services.api.delete(endpoint_path=PATH),
            lambda: client.custom_api.delete(endpoint_path=PATH),
            lambda: client.api.services.api.delete(PATH),
        ):
            try:
                attempt()
                print(f"delete: removed stale {PATH}")
                break
            except Exception:  # noqa: BLE001
                continue
        client.refresh()

    ep = sy.TwinAPIEndpoint(
        path=PATH,
        description="runtime engagement-budget control (data-owner admin)",
        mock_function=_ledger_admin,
        private_function=_ledger_admin,
    )
    rtype = type(client.custom_api.add(endpoint=ep)).__name__
    print("register:", rtype)
    if "Error" in rtype:
        print(f"[fatal] could not register {PATH}: {rtype}", file=sys.stderr)
        return 2
    client.refresh()

    res = client.api.services.diagnostic.ledger_admin(
        token_budget=args.token_budget,
        plot_budget=args.plot_budget,
        exemplar_budget=args.exemplar_budget,
        reset_usage=args.reset_usage,
        read_only=args.read_only,
    )
    res = res.get() if hasattr(res, "get") else res
    import json
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
