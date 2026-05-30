#!/usr/bin/env python3
"""
register_endpoints.py — register the 4 prepilot Mode B endpoints against a
running PySyft Datasite, from the laptop (admin side).

This replaces baking registration into pysyft_datasite_server.py during
development. The endpoint *definitions* are identical to what the container
would register at startup; only the trigger differs (laptop one-time call
vs. container boot). For the attestable production pilot, the same
build_endpoint() calls move back into the server entrypoint so the endpoint
code is covered by the TDX measurement. For pre-pilot measurement runs,
registration is one-time and does not affect per-request overhead, so the
numbers are identical either way.

Run from the cc-deep-eval repo root:

    DATASITE_URL=https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \
        python register_endpoints.py

Re-running is safe: existing endpoints at the same paths are deleted first.
Pass --clean-diagnostics to also remove any leftover diagnostic.* endpoints
from manual testing.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

# Make pysyft_endpoints importable regardless of CWD.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import syft as sy  # noqa: E402

from pysyft_endpoints.endpoints import (  # noqa: E402
    apply_steering,
    capture_attention_stats,
    capture_residual_stream,
    capture_routing,
)

ENDPOINT_MODULES = [
    ("capture_residual_stream", capture_residual_stream),
    ("capture_routing", capture_routing),
    ("capture_attention_stats", capture_attention_stats),
    ("apply_steering", apply_steering),
]


def _existing_paths(client) -> set[str]:
    """Return the set of currently-registered endpoint paths."""
    out: set[str] = set()
    try:
        for view in client.custom_api.api_endpoints():
            # TwinAPIEndpointView exposes .path; fall back to str() if not.
            p = getattr(view, "path", None)
            if p:
                out.add(p)
    except Exception as exc:  # noqa: BLE001
        print(f"[register][warn] could not enumerate endpoints: {exc}")
    return out


def _delete(client, path: str) -> None:
    """Best-effort delete of an endpoint by path (API varies subtly by build)."""
    for attempt in (
        lambda: client.api.services.api.delete(endpoint_path=path),
        lambda: client.custom_api.delete(endpoint_path=path),
        lambda: client.api.services.api.delete(path),
    ):
        try:
            res = attempt()
            print(f"[register] deleted existing '{path}' -> {type(res).__name__}")
            return
        except Exception:  # noqa: BLE001
            continue
    print(f"[register][warn] could not delete '{path}' (may not exist) — continuing")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasite-url", default=os.environ.get("DATASITE_URL"))
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--email", default="info@openmined.org")
    ap.add_argument("--password", default="changethis")
    ap.add_argument("--clean-diagnostics", action="store_true",
                    help="also delete any diagnostic.* endpoints")
    args = ap.parse_args()

    if not args.datasite_url:
        print("[register][fatal] set --datasite-url or DATASITE_URL", file=sys.stderr)
        return 2

    client = sy.login(
        url=args.datasite_url,
        port=args.port,
        email=args.email,
        password=args.password,
    )

    existing = _existing_paths(client)
    print(f"[register] existing endpoints: {sorted(existing) or '(none)'}")

    if args.clean_diagnostics:
        for p in sorted(existing):
            if p.startswith("diagnostic."):
                _delete(client, p)

    # Build + (re)register each endpoint.
    n_ok = 0
    for label, mod in ENDPOINT_MODULES:
        try:
            ep = mod.build_endpoint()
        except Exception as exc:  # noqa: BLE001
            print(f"[register][FAIL] build_endpoint() for {label}: "
                  f"{type(exc).__name__}: {exc}")
            continue

        if ep.path in existing:
            _delete(client, ep.path)

        res = client.custom_api.add(endpoint=ep)
        rtype = type(res).__name__
        if "Error" in rtype:
            print(f"[register][FAIL] add({ep.path}) -> {rtype}: {res}")
            continue
        print(f"[register][ok]   add({ep.path}) -> {rtype}")
        n_ok += 1

    client.refresh()
    final = _existing_paths(client)
    print(f"\n[register] {n_ok}/{len(ENDPOINT_MODULES)} registered")
    print(f"[register] datasite now exposes: {sorted(final)}")
    # Visible on the services tree?
    services = [x for x in dir(client.api.services) if not x.startswith("_")]
    print(f"[register] 'prepilot' on services tree: {'prepilot' in services}")

    return 0 if n_ok == len(ENDPOINT_MODULES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
