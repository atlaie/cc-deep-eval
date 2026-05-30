#!/usr/bin/env python3
"""
register_auditors.py — register N distinct PySyft users on the live Datasite so
the M5 concurrent-evaluator run has two (or more) real, distinct auditor
identities sharing one engagement gateway.

WHY THIS EXISTS
    `_common.py` resolves the engagement key as
    `auditor_id = context.user_client.metadata.email`. The Tinfoil debug shim
    is identity-blind (authenticated: false), but PySyft carries its OWN user
    identity independent of the shim. So two evaluator processes logging in as
    two different PySyft users yield two distinct auditor_ids and two distinct
    engagements under the same Datasite — with NO image rebuild. This is the
    same per-evaluator identity mechanism the white-paper pilot uses; only the
    transport (verified proxy + attestation) changes for the CC-on version.

DEFAULT REGISTRATION IS MANUAL (admin-driven) in PySyft 0.9.5, so we register
each auditor from the admin client. PySyft creates DATA_SCIENTIST users.

THE PERMISSION QUESTION (resolved at runtime, not assumed)
    A TwinAPIEndpoint exposes a mock_function and a private_function. Depending
    on the Datasite's permission config, a DATA_SCIENTIST may hit only the mock
    path (all-zeros payload, no ledger write) instead of the private path. M5
    needs the PRIVATE path (real encoder call + real SQLite write) to measure
    contention. We do not assume which applies: --verify-private smoke-tests one
    registered auditor against `prepilot.capture_residual_stream` and reports
    whether it reached the private path (pysyft_total_seconds > 0 AND
    bundle_bytes > 0). If it only reached the mock, pass --promote-admin to
    elevate the auditors to admin role (distinct emails preserved, so distinct
    auditor_ids preserved) — still a valid M5 because the quantity under test is
    SQLite write contention across distinct engagements, not the role label.

Run from the cc-deep-eval repo root:

    DATASITE_URL=https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \\
        python register_auditors.py --n 2 --verify-private

    # if the smoke test shows the mock path was used:
    DATASITE_URL=... python register_auditors.py --n 2 --promote-admin --verify-private

Exit codes:
  0  all auditors registered (and, if --verify-private, private path confirmed)
  2  user error
  5  --verify-private ran and the private path was NOT reached (mock only)
"""
from __future__ import annotations

import argparse
import os
import sys

import syft as sy


DEFAULT_DOMAIN = "example.com"  # RFC 2606 doc domain; passes pydantic's
                                # email validator. NOT a reserved special-use
                                # TLD like .test/.local/.invalid, which pydantic
                                # rejects ("special-use or reserved name").
DEFAULT_PASSWORD = "auditor_password"   # debug-only; CC-on uses real creds.


def auditor_credentials(n: int, domain: str) -> list[dict]:
    """Stable, deterministic credentials for N auditors: a, b, c, ..."""
    out = []
    for i in range(n):
        letter = chr(ord("a") + i)
        out.append({
            "name": f"Auditor {letter.upper()}",
            "email": f"auditor_{letter}@{domain}",
            "password": DEFAULT_PASSWORD,
        })
    return out


def _existing_emails(admin_client) -> set[str]:
    out: set[str] = set()
    try:
        for u in admin_client.users:
            e = getattr(u, "email", None)
            if e:
                out.add(e)
    except Exception as exc:  # noqa: BLE001
        print(f"[register][warn] could not enumerate users: {exc}")
    return out


def register_one(admin_client, cred: dict) -> bool:
    """Register a single DATA_SCIENTIST. Returns True on success/already-exists."""
    try:
        res = admin_client.register(
            name=cred["name"],
            email=cred["email"],
            password=cred["password"],
            password_verify=cred["password"],
        )
        rtype = type(res).__name__
        if "Error" in rtype:
            # Already-exists surfaces here on some builds; treat as benign.
            print(f"[register] {cred['email']} -> {rtype}: {res}")
            return "exist" in str(res).lower()
        print(f"[register][ok] {cred['email']} -> {rtype}")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "exist" in msg or "already" in msg:
            print(f"[register] {cred['email']} already exists")
            return True
        print(f"[register][FAIL] {cred['email']}: {type(exc).__name__}: {exc}")
        return False


def promote_to_admin(admin_client, email: str) -> None:
    """Best-effort role elevation. API shape varies across 0.9.x point builds;
    try the documented paths in order."""
    from syft.service.user.user_roles import ServiceRole  # type: ignore
    target = None
    for cand in ("ADMIN", "DATA_OWNER"):
        target = getattr(ServiceRole, cand, None)
        if target is not None:
            break
    if target is None:
        print(f"[promote][warn] no ADMIN/DATA_OWNER role enum found; skipping {email}")
        return
    for attempt in (
        lambda: _update_role_via_users_collection(admin_client, email, target),
        lambda: admin_client.api.services.user.update(
            uid=_uid_for_email(admin_client, email), role=target),
    ):
        try:
            res = attempt()
            print(f"[promote][ok] {email} -> {target} ({type(res).__name__})")
            return
        except Exception:  # noqa: BLE001
            continue
    print(f"[promote][warn] could not promote {email}; leaving as DATA_SCIENTIST")


def _uid_for_email(admin_client, email: str):
    for u in admin_client.users:
        if getattr(u, "email", None) == email:
            return u.id
    raise KeyError(email)


def _update_role_via_users_collection(admin_client, email: str, role):
    for u in admin_client.users:
        if getattr(u, "email", None) == email:
            return u.update(role=role)
    raise KeyError(email)


def verify_private_path(datasite_url: str, port: int, cred: dict) -> bool:
    """Log in AS the auditor and call capture_residual_stream once. Return True
    iff the PRIVATE path executed (non-zero pysyft_total and a real bundle).
    The mock path returns all-zeros, which we detect and treat as 'not private'.
    """
    try:
        client = sy.login(url=datasite_url, port=port,
                          email=cred["email"], password=cred["password"])
    except Exception as exc:  # noqa: BLE001
        print(f"[verify][FAIL] login as {cred['email']} failed: "
              f"{type(exc).__name__}: {exc}\n"
              f"          (was the auditor actually registered above?)")
        return False
    client.refresh()
    try:
        raw = client.api.services.prepilot.capture_residual_stream(
            prompt="m5 private-path smoke test", max_new_tokens=8,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verify][FAIL] {cred['email']} endpoint call raised: "
              f"{type(exc).__name__}: {exc}")
        return False
    res = raw.get() if hasattr(raw, "get") else raw
    if not isinstance(res, dict):
        res = getattr(raw, "syft_action_data", {}) or {}
    pt = (res.get("pysyft_timings") or {})
    eg = (res.get("egress") or {})
    total = float(pt.get("total_seconds") or 0.0)
    bundle_bytes = int(eg.get("bundle_bytes") or 0)
    auditor = res.get("auditor_id")
    is_private = total > 0.0 and bundle_bytes > 0
    print(f"[verify] {cred['email']}: auditor_id={auditor!r} "
          f"pysyft_total={total:.3f}s bundle_bytes={bundle_bytes} "
          f"-> {'PRIVATE' if is_private else 'MOCK (or empty)'}")
    return is_private


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasite-url", default=os.environ.get("DATASITE_URL"))
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--admin-email", default="info@openmined.org")
    ap.add_argument("--admin-password", default="changethis")
    ap.add_argument("--n", type=int, default=2, help="number of auditors")
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--promote-admin", action="store_true",
                    help="elevate registered auditors to admin role (use if the "
                         "private endpoint path is gated to non-DATA_SCIENTIST)")
    ap.add_argument("--enable-signup", action="store_true",
                    help="flip the Datasite to self-service registration first")
    ap.add_argument("--verify-private", action="store_true",
                    help="after registering, smoke-test one auditor against the "
                         "private endpoint path and report which path ran")
    args = ap.parse_args()

    if not args.datasite_url:
        print("[register][fatal] set --datasite-url or DATASITE_URL", file=sys.stderr)
        return 2
    if args.n < 1:
        print("[register][fatal] --n must be >= 1", file=sys.stderr)
        return 2

    admin = sy.login(url=args.datasite_url, port=args.port,
                     email=args.admin_email, password=args.admin_password)

    if args.enable_signup:
        try:
            admin.settings.allow_guest_signup(enable=True)
            print("[register] self-service signup enabled")
        except Exception as exc:  # noqa: BLE001
            print(f"[register][warn] allow_guest_signup failed: {exc}")

    creds = auditor_credentials(args.n, args.domain)
    existing = _existing_emails(admin)
    print(f"[register] existing users: {sorted(existing) or '(none enumerable)'}")

    n_ok = 0
    for c in creds:
        if c["email"] in existing:
            print(f"[register] {c['email']} already present")
            n_ok += 1
        elif register_one(admin, c):
            n_ok += 1
        if args.promote_admin:
            promote_to_admin(admin, c["email"])

    print(f"\n[register] {n_ok}/{len(creds)} auditors ready: "
          f"{[c['email'] for c in creds]}")

    if args.verify_private:
        ok = verify_private_path(args.datasite_url, args.port, creds[0])
        if not ok:
            print("\n[verify] PRIVATE path NOT reached. The endpoint served the "
                  "mock to a DATA_SCIENTIST. Re-run with --promote-admin to "
                  "elevate the auditors, then re-verify.", file=sys.stderr)
            return 5
        print("\n[verify] private path confirmed; M5 will measure real ledger writes.")

    return 0 if n_ok == len(creds) else 2


if __name__ == "__main__":
    raise SystemExit(main())
