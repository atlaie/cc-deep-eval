#!/usr/bin/env python3
"""
pysyft_datasite_spike_server.py — in-container entrypoint for the spike.

Launches sy.orchestra Datasite, registers a single Mode B endpoint
(`prepilot.hello`), and blocks. The endpoint takes no auditor input and
returns a fixed string; the spike is testing the *transport*, not the
endpoint logic.

Designed to run as the Tinfoil container's PID 1 — no supervisord, no
sidecars. Phase 1's `Dockerfile.pysyft` will need supervisord because it
runs vLLM + egress + Datasite as three processes; this image runs one.

Healthcheck (set in tinfoil-config-pysyft-spike.yml):
    curl -sf http://localhost:8000/api/v2/metadata

PySyft 0.9.5's metadata route is always present and unauthenticated; safe
as a liveness probe. If that route changes in a future PySyft minor, update
the tinfoil-config healthcheck accordingly.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import syft as sy


def main() -> int:
    sy.requires(">=0.9.5,<0.9.6")

    port = int(os.environ.get("SYFT_PORT", "8000"))
    name = os.environ.get("SYFT_DATASITE_NAME", "prepilot-spike")

    print(f"[spike] launching Datasite name={name!r} port={port}", flush=True)
    server = sy.orchestra.launch(
        name=name,
        port=port,
        # 0.0.0.0 so the Tinfoil shim can reach us. orchestra.launch defaults
        # to 127.0.0.1 in older versions; passing host explicitly is safer.
        host="0.0.0.0",
        create_producer=True,
        n_consumers=1,
        dev_mode=False,    # SQLite backing
        reset=False,       # keep state across container restarts
    )
    print("[spike] Datasite up; registering endpoint ...", flush=True)

    # Log in as admin to register the endpoint. The default password is in
    # PySyft's README; for the spike that's fine — non-CC host, --debug mode,
    # ephemeral container. Phase 1 will provision a real credential.
    client = sy.login(port=port, email="info@openmined.org", password="changethis")

    @sy.api_endpoint(
        path="prepilot.hello",
        description="PySyft spike: returns a fixed string. Confirms Mode B "
                    "transport works through the Tinfoil shim.",
        settings={"deploy_marker": "tinfoil-spike-v0.0.1"},
    )
    def hello(context) -> str:
        return f"hello from datasite (deploy_marker={context.settings.get('deploy_marker')})"

    res = client.custom_api.add(endpoint=hello)
    if "Error" in type(res).__name__:
        print(f"[spike] FATAL custom_api.add: {res}", file=sys.stderr)
        return 1
    print(f"[spike] endpoint registered: prepilot.hello", flush=True)
    print(f"[spike] datasite ready; blocking on SIGTERM / SIGINT", flush=True)

    # Block until the container is torn down. orchestra.launch() spawns
    # uvicorn in a child thread; the main thread idles.
    stop = False
    def _shutdown(signum, _frame):
        nonlocal stop
        print(f"[spike] caught signal {signum}; shutting down", flush=True)
        stop = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    while not stop:
        time.sleep(5)

    try:
        server.land()
    except Exception as e:
        print(f"[spike] server.land() failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
