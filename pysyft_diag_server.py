#!/usr/bin/env python3
"""
pysyft_diag_server.py — DIAGNOSTIC entrypoint for the spike.

Goal: get the actual PySyft crash output out of a Tinfoil container we
can't SSH into. Strategy:

  1. Start a 30-line HTTP server on port 8000 IMMEDIATELY. Healthcheck
     curls /api/v2/metadata; we return 200 with the diagnostic log as
     the body. Container goes Running, Tinfoil keeps it up, SSH starts
     accepting connections.
  2. Try `import syft` + `sy.orchestra.launch(...)` in a background
     thread. Capture stdout / stderr / exceptions into the same log buffer.
  3. Laptop runs:
         curl -H "Authorization: Bearer $TINFOIL_API_KEY" \\
              https://pysyft-spike.debug.pour-demain.containers.tinfoil.dev/api/v2/metadata
     and sees the full traceback.

This is NOT the production spike entrypoint. It's a one-shot diagnostic
swap. Once we know the crash, we go back to `pysyft_datasite_spike_server.py`.
"""
from __future__ import annotations

import http.server
import io
import os
import socketserver
import sys
import threading
import time
import traceback

LOG_LOCK = threading.Lock()
LOG_BUF = io.StringIO()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with LOG_LOCK:
        LOG_BUF.write(line + "\n")
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def dump() -> bytes:
    with LOG_LOCK:
        return LOG_BUF.getvalue().encode()


# ===== background: try to launch PySyft, capture everything =================

def try_pysyft() -> None:
    try:
        log("=== diagnostic boot ===")
        log(f"python: {sys.version.split(chr(10))[0]}")
        log(f"executable: {sys.executable}")
        log(f"cwd: {os.getcwd()}")
        log(f"env keys (sanitised): {sorted(k for k in os.environ if 'KEY' not in k.upper())}")

        log("--- pip list (head) ---")
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pip", "list"],
                           capture_output=True, text=True, timeout=30)
        for line in r.stdout.splitlines()[:40]:
            log(line)

        log("--- importing syft ---")
        import syft as sy
        log(f"syft.__version__ = {sy.__version__}")

        log("--- sy.requires(>=0.9.5,<0.9.6) ---")
        sy.requires(">=0.9.5,<0.9.6")
        log("requires: OK")

        log("--- sy.orchestra.launch(...) ---")
        # orchestra binds 0.0.0.0:port automatically. BUT — port 8000 is
        # already taken by the diagnostic HTTP server. Use 8001 internally;
        # PySyft will fail to bind 8000, so this surfaces a port conflict
        # which is what we want to see (or not) in the log.
        port = int(os.environ.get("SYFT_PORT", "8001"))
        name = os.environ.get("SYFT_DATASITE_NAME", "prepilot-spike-diag")
        server = sy.orchestra.launch(
            name=name,
            port=port,
            create_producer=True,
            n_consumers=1,
            dev_mode=False,
            reset=True,
        )
        log(f"orchestra.launch: returned {type(server).__name__}")
        log("=== PySyft started successfully ===")
        # Keep the thread alive so the launched server doesn't get GC'd.
        while True:
            time.sleep(60)

    except Exception as e:
        log(f"!!! exception: {type(e).__name__}: {e}")
        log("--- traceback ---")
        for line in traceback.format_exc().splitlines():
            log(line)
        log("=== PySyft FAILED to start; container staying up for inspection ===")


# ===== HTTP server: always 200, body = diagnostic log =======================

class DiagHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):    # noqa: N802
        body = dump()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # quiet the default access-log spam from healthcheck curls
    def log_message(self, format, *args):  # noqa: A002
        return


def main() -> int:
    log("diag entrypoint starting")

    # Background thread: PySyft startup attempt.
    t = threading.Thread(target=try_pysyft, daemon=True)
    t.start()

    # Foreground: HTTP server on 8000 (the port the healthcheck and the
    # Tinfoil shim both expect). This must stay alive — it's PID 1's job.
    log("starting diagnostic HTTP server on 0.0.0.0:8000")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", 8000), DiagHandler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
