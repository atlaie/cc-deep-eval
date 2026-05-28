"""
pysyft_datasite_server.py — production Phase 1 entrypoint.

Replaces pysyft_datasite_spike_server.py. Launches the PySyft 0.9.5
Datasite via sy.orchestra, then registers the four Mode B TwinAPIEndpoints
defined under pysyft_endpoints.endpoints.

Runs as one of three processes inside the Phase 1 image (the other two
are vLLM on 127.0.0.1:8001 and egress_service on 127.0.0.1:8002). The
supervisord wrapper (entrypoint_pysyft.sh) launches this last, after
egress_service has reported healthy on its loopback /health endpoint,
because endpoint registration touches the loopback to validate it.

Operational notes (carried over from spike findings):

  - `sy.orchestra.launch()` does NOT accept a `host=` kwarg in 0.9.5;
    binds 0.0.0.0 by default. Don't pass it.
  - `reset=True` for cold-container safety. The engagement ledger is a
    SEPARATE SQLite DB outside PySyft's internal state, so reset=True here
    doesn't drop our ledger data.
  - Default admin password `changethis` must be replaced via env var
    (`SYFT_ADMIN_PASSWORD`) before the cap-enforcement experiment (M3).
  - The Datasite's own internal DB lives under /tmp/syft/<uuid>/; our
    engagement ledger lives at /workspace/engagement_ledger.sqlite,
    persistent across container restarts via a Tinfoil-mounted volume if
    configured (otherwise ephemeral, which is fine for the experimental runs).
"""
from __future__ import annotations

import os
import signal
import sys
import time

import syft as sy

from pysyft_endpoints.endpoints import (
    apply_steering,
    capture_attention_stats,
    capture_residual_stream,
    capture_routing,
)


def main() -> int:
    sy.requires(">=0.9.5,<0.9.6")

    port = int(os.environ.get("SYFT_PORT", "8000"))
    name = os.environ.get("SYFT_DATASITE_NAME", "prepilot-phase1")
    admin_email = os.environ.get("SYFT_ADMIN_EMAIL", "info@openmined.org")
    admin_password = os.environ.get("SYFT_ADMIN_PASSWORD", "changethis")

    if admin_password == "changethis":
        # Loud warning so this doesn't sneak into the cap-enforcement run.
        print(
            "[server][WARN] SYFT_ADMIN_PASSWORD is the default 'changethis'. "
            "Acceptable for non-CC debug deploys only. Set the Tinfoil org "
            "secret before any production run.",
            flush=True,
        )

    print(f"[server] launching Datasite name={name!r} port={port}", flush=True)
    server = sy.orchestra.launch(
        name=name,
        port=port,
        create_producer=True,
        n_consumers=int(os.environ.get("SYFT_N_CONSUMERS", "4")),
        dev_mode=False,
        reset=True,
    )
    print("[server] Datasite up; logging in as admin to register endpoints",
          flush=True)

    client = sy.login(port=port, email=admin_email, password=admin_password)

    # Register all four TwinAPIEndpoints. Each module exposes build_endpoint().
    endpoint_modules = [
        ("capture_residual_stream",  capture_residual_stream),
        ("capture_routing",          capture_routing),
        ("capture_attention_stats",  capture_attention_stats),
        ("apply_steering",           apply_steering),
    ]
    for label, mod in endpoint_modules:
        ep = mod.build_endpoint()
        res = client.custom_api.add(endpoint=ep)
        if "Error" in type(res).__name__:
            print(f"[server][FATAL] custom_api.add({label}) failed: {res}",
                  file=sys.stderr)
            return 1
        print(f"[server] registered endpoint: {ep.path}", flush=True)

    print(f"[server] all {len(endpoint_modules)} endpoints registered; "
          f"blocking on SIGTERM / SIGINT",
          flush=True)

    stop = False
    def _shutdown(signum, _frame):
        nonlocal stop
        print(f"[server] caught signal {signum}; shutting down", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    while not stop:
        time.sleep(5)

    try:
        server.land()
    except Exception as e:
        print(f"[server] server.land() failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
