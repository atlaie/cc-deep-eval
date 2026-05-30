#!/usr/bin/env python3
"""
ctx_probe.py — inspect what auditor identity the PySyft worker `context`
actually exposes on this deploy, to decide whether distinct-auditor M5 is
reachable without a rebuild.

Background: on the live debug deploy, capture_residual_stream resolved
auditor_id='unknown', meaning `_common.py`'s `context.user_client.metadata.email`
chain raised AttributeError. If `context` carries identity by another route
(credentials / verify key), we can map it to the auditor email with only a
laptop-side register_endpoints.py re-register (no image rebuild). If context is
genuinely identity-blind here, M5 runs as a single shared engagement.

Run (logs in as the AUDITOR, not admin, so the probe sees a non-admin context):

    DATASITE_URL=https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \\
        python ctx_probe.py --as-auditor auditor_a@example.com
"""
import argparse
import json
import os
import sys

import syft as sy


@sy.api_endpoint_method()
def _ctxprobe(context):
    out = {}
    for attr in ("credentials", "role", "user_client", "server", "node",
                 "syft_client", "id"):
        try:
            v = getattr(context, attr)
            out[attr] = f"{type(v).__name__} = {str(v)[:160]}"
        except Exception as e:  # noqa: BLE001
            out[attr] = f"<{type(e).__name__}: {e}>"

    # The exact chain _common.py uses today:
    try:
        out["user_client.metadata.email"] = context.user_client.metadata.email
    except Exception as e:  # noqa: BLE001
        out["user_client.metadata.email"] = f"{type(e).__name__}: {e}"

    # Candidate alternative routes to a stable per-user identity:
    # 1. credentials is usually a SyftVerifyKey (per-user keypair) even under
    #    the debug shim, because PySyft auth is independent of the Tinfoil shim.
    try:
        out["credentials_repr"] = repr(context.credentials)[:200]
    except Exception as e:  # noqa: BLE001
        out["credentials_repr"] = f"{type(e).__name__}: {e}"

    # 2. Look up the user by verify key via the server's user service, which is
    #    what would let us resolve credentials -> email inside the endpoint.
    try:
        # context.server gives the worker-side server handle on 0.9.x
        server = getattr(context, "server", None) or getattr(context, "node", None)
        out["server_present"] = server is not None
        if server is not None:
            # Best-effort: enumerate the user service stash for an email match.
            # We don't assume the exact API; just record what's reachable.
            user_service = None
            try:
                user_service = server.get_service("UserService")
            except Exception:
                try:
                    user_service = server.services.user
                except Exception:
                    user_service = None
            out["user_service_reachable"] = user_service is not None
    except Exception as e:  # noqa: BLE001
        out["server_lookup"] = f"{type(e).__name__}: {e}"

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasite-url", default=os.environ.get("DATASITE_URL"))
    ap.add_argument("--port", type=int, default=443)
    ap.add_argument("--as-auditor", default=None,
                    help="log in as this auditor email (default: admin). Use a "
                         "registered auditor so the probe sees a NON-admin context.")
    ap.add_argument("--password", default=None,
                    help="password for --as-auditor (default: auditor_password)")
    args = ap.parse_args()
    if not args.datasite_url:
        print("set --datasite-url or DATASITE_URL", file=sys.stderr)
        return 2

    # Register the probe as ADMIN (only admin can add endpoints), then invoke
    # it as the auditor so context reflects the auditor's identity.
    admin = sy.login(url=args.datasite_url, port=args.port,
                     email="info@openmined.org", password="changethis")
    PATH = "diagnostic.ctxprobe"
    existing = {getattr(v, "path", None) for v in admin.custom_api.api_endpoints()}
    if PATH in existing:
        for attempt in (
            lambda: admin.api.services.api.delete(endpoint_path=PATH),
            lambda: admin.custom_api.delete(endpoint_path=PATH),
        ):
            try:
                attempt(); break
            except Exception:
                continue
        admin.refresh()
    ep = sy.TwinAPIEndpoint(
        path=PATH, description="probe worker context identity",
        mock_function=_ctxprobe, private_function=_ctxprobe,
    )
    print("register:", type(admin.custom_api.add(endpoint=ep)).__name__)
    admin.refresh()

    # Invoke as the chosen identity.
    if args.as_auditor:
        pw = args.password or "auditor_password"
        caller = sy.login(url=args.datasite_url, port=args.port,
                          email=args.as_auditor, password=pw)
        print(f"invoking probe as auditor: {args.as_auditor}")
    else:
        caller = admin
        print("invoking probe as admin")
    caller.refresh()

    res = caller.api.services.diagnostic.ctxprobe()
    res = res.get() if hasattr(res, "get") else res
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
