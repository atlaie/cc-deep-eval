"""
_common.py — shared endpoint body for the four Mode B TwinAPIEndpoints.

Each of the four endpoints (capture_residual_stream, capture_routing,
capture_attention_stats, apply_steering) differs only in the vllm_xargs
it constructs. The rest of the pipeline — auditor lookup, engagement
cap-check, loopback POST, ledger insert, response shape — is identical.

This module factors that pipeline. The four endpoint files supply:
  - endpoint path
  - description
  - an `xargs_builder` callable that maps endpoint kwargs to vllm_xargs

Per-stage perf_counter() timings are returned to the caller so
`phase3_pysyft_driver.py` can decompose total wall into:
    workflow_seconds      — PySyft worker dispatch + auditor resolve
    approval_seconds      — engagement cap check + session start
    encoder_seconds       — loopback HTTP to /v1/egress_eval
    ledger_seconds        — engagement-ledger bundle insert
    bundle_return_seconds — PySyft serialisation back to client

These map to the brief's Table 9 stage decomposition; the driver row schema
mirrors `phase3_egress_driver_v2.EgressRowV2` plus the five `pysyft_*`
columns above.
"""
from __future__ import annotations


# Module-level constants ---------------------------------------------------
# Duplicated from captures.py because the endpoint function body runs in
# the PySyft worker, where importing from cc-benchbox is not guaranteed.
# Source of truth remains captures.py; if it diverges we resync here.

DEFAULT_PROBE_LAYERS = [12, 23, 39, 51, 62, 70]
DEFAULT_ROUTING_LAYERS = list(range(3, 78))    # GLM-5.1: 75 MoE layers, 0..2 dense.
GLM51_HIDDEN_SIZE = 6144

ENCODER_URL = "http://127.0.0.1:8002/v1/egress_eval"
ENCODER_TIMEOUT_SECONDS = 120.0
DEFAULT_MODEL = "glm-5-1-fp8"
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_EGRESS_STAGES = ["aggregate", "plot", "bundle", "ledger"]

LEDGER_DB_PATH = "/workspace/engagement_ledger.sqlite"


def call_endpoint(
    context,
    *,
    endpoint_id: str,
    prompt: str,
    xargs: dict,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    egress_stages: list = None,
    include_bundle: bool = True,
) -> dict:
    """Shared endpoint body.

    Returns the encoder's response augmented with `pysyft_timings`. On
    engagement cap-exceed, returns a typed error payload instead of raising
    — PySyft surfaces the dict to the auditor and `phase3_pysyft_driver`
    categorises by the `error` field.

    Imports are inline so the function works when serialised by PySyft.
    """
    import time
    import httpx

    # The PySyft worker's Python path includes /workspace; engagement_ledger
    # is therefore importable. If pysyft_endpoints package layout changes,
    # update this import path.
    from pysyft_endpoints.ledger.engagement_ledger import (
        EngagementLedger,
        EngagementCapExceeded,
    )

    t_total_start = time.perf_counter()
    egress_stages = egress_stages or DEFAULT_EGRESS_STAGES

    # ---- 1. workflow dispatch + auditor resolve ----
    t0 = time.perf_counter()
    try:
        auditor_id = context.user_client.metadata.email
    except AttributeError:
        # Fallback path: older PySyft client builds expose metadata via a
        # different attribute. Catch broadly; auditor_id="unknown" is the
        # debug-mode default and the engagement ledger will create an
        # auto-engagement for it.
        auditor_id = "unknown"
    workflow_seconds = time.perf_counter() - t0

    # ---- 2. approval gate (engagement cap + session start) ----
    t0 = time.perf_counter()
    ledger = EngagementLedger(LEDGER_DB_PATH)
    try:
        engagement_id, session_id = ledger.start_session_or_raise(auditor_id)
    except EngagementCapExceeded as e:
        ledger.close()
        return e.to_payload() | {
            "endpoint": endpoint_id,
            "auditor_id": auditor_id,
            "pysyft_timings": {
                "workflow_seconds": workflow_seconds,
                "approval_seconds": time.perf_counter() - t0,
                "encoder_seconds": 0.0,
                "ledger_seconds": 0.0,
                "bundle_return_seconds": 0.0,
                "total_seconds": time.perf_counter() - t_total_start,
            },
        }
    approval_seconds = time.perf_counter() - t0

    # ---- 3. encoder call (loopback to egress_service) ----
    t0 = time.perf_counter()
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_new_tokens,
        "vllm_xargs": xargs,
        "chat_template_kwargs": {"enable_thinking": False},
        "egress": {
            "stages": egress_stages,
            "session_id": session_id,
            "include_bundle": include_bundle,
            "request_id": 0,            # PySyft doesn't carry one; placeholder.
            "pair_id": 0,
            "prompt_class": "auditor",
        },
    }
    try:
        r = httpx.post(ENCODER_URL, json=body, timeout=ENCODER_TIMEOUT_SECONDS)
        r.raise_for_status()
        enc = r.json()
    except Exception as exc:
        ledger.close()
        return {
            "error": "encoder_call_failed",
            "endpoint": endpoint_id,
            "auditor_id": auditor_id,
            "engagement_id": engagement_id,
            "session_id": session_id,
            "detail": f"{type(exc).__name__}: {exc}",
            "pysyft_timings": {
                "workflow_seconds": workflow_seconds,
                "approval_seconds": approval_seconds,
                "encoder_seconds": time.perf_counter() - t0,
                "ledger_seconds": 0.0,
                "bundle_return_seconds": 0.0,
                "total_seconds": time.perf_counter() - t_total_start,
            },
        }
    encoder_seconds = time.perf_counter() - t0

    # ---- 4. ledger insert ----
    t0 = time.perf_counter()
    eg = enc.get("egress") or {}
    completion = enc.get("completion") or {}
    usage = completion.get("usage") or {}
    try:
        ledger.record_bundle(
            session_id=session_id,
            endpoint=endpoint_id,
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            aggregate_bytes=int(eg.get("aggregate_bytes") or 0),
            bundle_bytes=int(eg.get("bundle_bytes") or 0),
            bundle_sha256=str(eg.get("bundle_sha256") or ""),
            n_plots=int(eg.get("n_plots") or 0),
            n_exemplars=0,
        )
    finally:
        ledger.close()
    ledger_seconds = time.perf_counter() - t0

    # ---- 5. bundle return ----
    t0 = time.perf_counter()
    # PySyft handles the actual serialisation back to the auditor; we just
    # record the time spent assembling the response object here.
    result = {
        "endpoint": endpoint_id,
        "auditor_id": auditor_id,
        "engagement_id": engagement_id,
        "session_id": session_id,
        "completion": completion,
        "egress": eg,
        "pysyft_timings": {
            "workflow_seconds": workflow_seconds,
            "approval_seconds": approval_seconds,
            "encoder_seconds": encoder_seconds,
            "ledger_seconds": ledger_seconds,
            "bundle_return_seconds": time.perf_counter() - t0,
            "total_seconds": time.perf_counter() - t_total_start,
        },
    }
    return result


# ===== mock function body ===================================================

def zero_filled_mock(
    context,
    *,
    endpoint_id: str,
    prompt: str = "",
    **_kwargs,
) -> dict:
    """Zero-filled mock body. Audit-isomorphic envelope, no real content.

    Returns a payload matching the private path's shape exactly — same keys,
    same nesting — but with all numeric fields zero and all string fields
    empty. Lets auditor code that handles the private-path response shape
    run unchanged against the mock; verifies envelope compatibility without
    exercising the encoder or ledger.

    Phase 1 ships with this; can be replaced by a richer fixture later
    (e.g., GPT-2 residual stream snapshot) without changing the endpoint
    contract.
    """
    return {
        "endpoint": endpoint_id,
        "auditor_id": "mock",
        "engagement_id": "00000000-0000-0000-0000-000000000000",
        "session_id":    "00000000-0000-0000-0000-000000000000",
        "completion": {
            "choices": [{"message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
        "egress": {
            "stages_run": list(DEFAULT_EGRESS_STAGES),
            "aggregate_bytes": 0,
            "bundle_bytes": 0,
            "raw_payload_bytes": 0,
            "n_plots": 0,
            "bundle_sha256": "0" * 64,
            "bundle_signature_hex": "",
            "deserialize_seconds": 0.0,
            "encoder_total_seconds": 0.0,
            "aggregate_seconds": 0.0,
            "plot_seconds": 0.0,
            "bundle_seconds": 0.0,
            "ledger_seconds": 0.0,
        },
        "pysyft_timings": {
            "workflow_seconds":      0.0,
            "approval_seconds":      0.0,
            "encoder_seconds":       0.0,
            "ledger_seconds":        0.0,
            "bundle_return_seconds": 0.0,
            "total_seconds":         0.0,
        },
    }
