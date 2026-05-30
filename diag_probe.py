#!/usr/bin/env python3
"""
diag_probe.py — register a worker-side probe endpoint and call it.

The probe runs *inside* the container (PySyft worker), so it can reach the
loopback services (vLLM 8001, egress 8002) that the laptop cannot. It fetches
each service's /openapi.json and returns the title + route list, so we can see
exactly what is listening on 8002 and whether /v1/egress_eval exists.

Run:
    DATASITE_URL=https://pysyft-m1.debug.pour-demain.containers.tinfoil.dev \
        python diag_probe.py
"""
import os
import sys

import syft as sy


# NOTE: only `context` as a param (no annotations) to avoid the signature
# validator; defined in a file so inspect.getsource works.
@sy.api_endpoint_method()
def _probe(context):
    import httpx
    out = {}
    for port in (8001, 8002):
        base = "http://127.0.0.1:%d" % port
        entry = {}
        # openapi route list
        try:
            r = httpx.get(base + "/openapi.json", timeout=5)
            entry["openapi_status"] = r.status_code
            if r.status_code == 200:
                spec = r.json()
                entry["title"] = spec.get("info", {}).get("title")
                entry["paths"] = sorted(spec.get("paths", {}).keys())
        except Exception as exc:
            entry["openapi_error"] = "%s: %s" % (type(exc).__name__, exc)
        # health probe
        try:
            rh = httpx.get(base + "/health", timeout=5)
            entry["health_status"] = rh.status_code
        except Exception as exc:
            entry["health_error"] = "%s: %s" % (type(exc).__name__, exc)
        out[str(port)] = entry
    return out


def main():
    url = os.environ.get("DATASITE_URL")
    if not url:
        print("set DATASITE_URL", file=sys.stderr)
        return 2
    client = sy.login(url=url, port=443,
                      email="info@openmined.org", password="changethis")

    ep = sy.TwinAPIEndpoint(
        path="diagnostic.probe",
        description="probe loopback service routes from the worker",
        mock_function=_probe,
        private_function=_probe,
    )
    # idempotent: delete if present
    for attempt in (
        lambda: client.api.services.api.delete(endpoint_path="diagnostic.probe"),
        lambda: client.custom_api.delete(endpoint_path="diagnostic.probe"),
    ):
        try:
            attempt()
            break
        except Exception:
            continue

    print("add:", type(client.custom_api.add(endpoint=ep)).__name__)
    client.refresh()

    import json
    res = client.api.services.diagnostic.probe()
    # res may be a Pointer; coerce to plain value
    try:
        res = res.get() if hasattr(res, "get") else res
    except Exception:
        pass
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
