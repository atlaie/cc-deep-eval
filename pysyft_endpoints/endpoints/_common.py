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
    workflow_seconds          — PySyft worker dispatch + auditor resolve
    approval_seconds          — engagement cap check + session start
    encoder_seconds           — loopback HTTP to /v1/egress_eval
    ledger_seconds            — engagement-ledger bundle insert
    response_assembly_seconds — building the return dict (NOT serialisation)

ATTRIBUTION NOTE (was a mislabel; corrected here):
    The previous field name `bundle_return_seconds` implied this stage
    captured the cost of serialising the response back to the auditor.
    It does NOT. PySyft performs that serialisation AFTER this function
    returns, so it is unobservable from inside the endpoint body. What
    this stage actually measures is the wall to assemble the Python dict
    — invariably sub-millisecond. The real serialise+transport cost is
    captured laptop-side by the driver as `transport_serialize_seconds`
    (= wall − pysyft_total). Renaming the server stage to
    `response_assembly_seconds` keeps Table 9 honest: the four governance
    stages (workflow/approval/ledger/assembly) are server-measured and
    genuinely tiny; the serialise/transport slice is a separate,
    correctly-labelled line, not silently folded into a PySyft stage.

These map to the brief's Table 9 stage decomposition; the driver row schema
mirrors `phase3_egress_driver_v2.EgressRowV2` plus the five `pysyft_*`
columns above.

DIAGNOSTIC INSTRUMENTATION (2026-05-29, wedge investigation):
    Jobs were observed hanging in PROCESSING with n_iters=0, GPU idle, no
    log — a parked worker, before any inference. The prime suspect is this
    module's stage-3 encoder POST blocking on a non-responsive egress_service
    (loopback :8002) or its vLLM call. Changes to isolate that:
      - ENCODER_TIMEOUT_SECONDS 120 -> 45, so the worker FAILS FAST below the
        ~60s laptop->TEE client cutoff instead of hanging past it. A real
        encoder call is ~hundreds of ms (Table 9), so 45s is enormous
        headroom; a 45s timeout firing means the encoder is genuinely hung.
      - explicit stderr logging immediately BEFORE and AFTER the POST, so
        `docker logs` shows whether the worker reached the encoder, how long
        it waited, and whether it returned or timed out. This is the signal
        that distinguishes "hang IS the encoder POST" from "hang is upstream
        of it" (auditor resolve / ledger BEGIN IMMEDIATE / worker dispatch).
      - the timeout/exception payload now records encoder_seconds and a clear
        error kind, so the failure is fully diagnosable client-side via
        job.result (which returns through the allowlisted path).
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
# Real-run value: encoder work for heavy cells (residual stream + many tokens)
# can exceed 45s. The diagnostic 45s confirmed the encoder was genuinely slow,
# not hung (read timeout fired on real work). Set high enough not to truncate
# legitimate heavy cells; the laptop->TEE path's own ~60s cutoff is the real
# ceiling on blocking calls (separate concern, not this timeout).
ENCODER_TIMEOUT_SECONDS = 300.0
# Separate, short connect timeout: if egress_service isn't even accepting
# connections, fail in seconds, not after the full read timeout. httpx takes
# a (connect, read, write, pool) tuple via httpx.Timeout.
ENCODER_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_MODEL = "glm-5-1"   # must match vLLM --served-model-name in the deploy
                            # (tinfoil-config-pysyft.yml). NOTE: brief egress
                            # deploys use "glm-5-1-fp8"; standardise at convergence.
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_EGRESS_STAGES = ["aggregate", "plot", "bundle", "ledger"]

LEDGER_DB_PATH = "/workspace/engagement_ledger.sqlite"


def _log(msg: str) -> None:
    """Best-effort stderr log from inside the PySyft worker, flushed so it
    lands in `docker logs` immediately. Prefixed for grep. Never raises."""
    try:
        import sys as _sys
        import time as _time
        print(f"[endpoint {_time.strftime('%H:%M:%S')}] {msg}",
              file=_sys.stderr, flush=True)
    except Exception:
        pass


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
    _log(f"ENTER {endpoint_id} max_new_tokens={max_new_tokens} "
         f"stages={egress_stages}")

    # ---- 1. workflow dispatch + auditor resolve ----
    t0 = time.perf_counter()
    try:
        auditor_id = context.user.email
    except AttributeError:
        # Fallback path: older PySyft client builds expose metadata via a
        # different attribute. Catch broadly; auditor_id="unknown" is the
        # debug-mode default and the engagement ledger will create an
        # auto-engagement for it.
        auditor_id = "unknown"
    workflow_seconds = time.perf_counter() - t0
    _log(f"stage1 auditor_resolve={workflow_seconds*1000:.1f}ms "
         f"auditor_id={auditor_id}")

    # ---- 2. approval gate (engagement cap + session start) ----
    t0 = time.perf_counter()
    _log("stage2 opening ledger / start_session_or_raise "
         "(BEGIN IMMEDIATE)...")
    ledger = EngagementLedger(LEDGER_DB_PATH)
    try:
        engagement_id, session_id = ledger.start_session_or_raise(auditor_id)
    except EngagementCapExceeded as e:
        ledger.close()
        _log(f"stage2 CAP EXCEEDED for auditor_id={auditor_id}")
        return e.to_payload() | {
            "endpoint": endpoint_id,
            "auditor_id": auditor_id,
            "pysyft_timings": {
                "workflow_seconds": workflow_seconds,
                "approval_seconds": time.perf_counter() - t0,
                "encoder_seconds": 0.0,
                "ledger_seconds": 0.0,
                "response_assembly_seconds": 0.0,
                "total_seconds": time.perf_counter() - t_total_start,
            },
        }
    approval_seconds = time.perf_counter() - t0
    _log(f"stage2 approval={approval_seconds*1000:.1f}ms "
         f"engagement={engagement_id} session={session_id}")

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
    # Diagnostic: log immediately before the POST. If the worker hangs at the
    # encoder, this is the LAST line that appears for the job in docker logs.
    _log(f"stage3 POST {ENCODER_URL} "
         f"(connect_timeout={ENCODER_CONNECT_TIMEOUT_SECONDS}s "
         f"read_timeout={ENCODER_TIMEOUT_SECONDS}s)...")
    timeout = httpx.Timeout(
        ENCODER_TIMEOUT_SECONDS,
        connect=ENCODER_CONNECT_TIMEOUT_SECONDS,
    )
    try:
        r = httpx.post(ENCODER_URL, json=body, timeout=timeout)
        r.raise_for_status()
        enc = r.json()
    except httpx.ConnectTimeout as exc:
        ledger.close()
        dt = time.perf_counter() - t0
        _log(f"stage3 CONNECT TIMEOUT after {dt:.1f}s — egress_service not "
             f"accepting connections on :8002")
        return _encoder_error(
            "encoder_connect_timeout", endpoint_id, auditor_id,
            engagement_id, session_id,
            f"connect timeout after {dt:.1f}s to {ENCODER_URL}; egress_service "
            f"not accepting connections",
            workflow_seconds, approval_seconds, dt, t_total_start)
    except httpx.ReadTimeout as exc:
        ledger.close()
        dt = time.perf_counter() - t0
        _log(f"stage3 READ TIMEOUT after {dt:.1f}s — egress_service accepted "
             f"the connection but did not respond (it or its vLLM call is hung)")
        return _encoder_error(
            "encoder_read_timeout", endpoint_id, auditor_id,
            engagement_id, session_id,
            f"read timeout after {dt:.1f}s from {ENCODER_URL}; egress_service "
            f"accepted but did not respond within {ENCODER_TIMEOUT_SECONDS}s "
            f"(egress_service or its vLLM loopback is hung)",
            workflow_seconds, approval_seconds, dt, t_total_start)
    except Exception as exc:
        ledger.close()
        dt = time.perf_counter() - t0
        _log(f"stage3 ENCODER ERROR after {dt:.1f}s: {type(exc).__name__}: {exc}")
        return _encoder_error(
            "encoder_call_failed", endpoint_id, auditor_id,
            engagement_id, session_id,
            f"{type(exc).__name__}: {exc}",
            workflow_seconds, approval_seconds, dt, t_total_start)
    encoder_seconds = time.perf_counter() - t0
    _log(f"stage3 encoder OK in {encoder_seconds:.2f}s")

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
    _log(f"stage4 ledger_insert={ledger_seconds*1000:.1f}ms")

    # ---- 5. response assembly ----
    # NOTE: this is the cost of building the return dict, NOT of serialising
    # it back to the auditor. PySyft serialises after we return; that cost is
    # captured laptop-side by the driver as `transport_serialize_seconds`.
    # This stage is invariably sub-millisecond and is labelled accordingly in
    # Table 9 to avoid implying it measures the wire cost.
    t0 = time.perf_counter()
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
            "response_assembly_seconds": time.perf_counter() - t0,
            "total_seconds": time.perf_counter() - t_total_start,
        },
    }
    _log(f"EXIT {endpoint_id} total={result['pysyft_timings']['total_seconds']:.2f}s")
    return result


def _encoder_error(kind, endpoint_id, auditor_id, engagement_id, session_id,
                   detail, workflow_seconds, approval_seconds, encoder_seconds,
                   t_total_start):
    """Build a typed encoder-failure payload with timings. Centralised so the
    three encoder except-branches stay consistent and fully diagnosable
    client-side via job.result."""
    import time
    return {
        "error": kind,
        "endpoint": endpoint_id,
        "auditor_id": auditor_id,
        "engagement_id": engagement_id,
        "session_id": session_id,
        "detail": detail,
        "pysyft_timings": {
            "workflow_seconds": workflow_seconds,
            "approval_seconds": approval_seconds,
            "encoder_seconds": encoder_seconds,
            "ledger_seconds": 0.0,
            "response_assembly_seconds": 0.0,
            "total_seconds": time.perf_counter() - t_total_start,
        },
    }


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
            "workflow_seconds":          0.0,
            "approval_seconds":          0.0,
            "encoder_seconds":           0.0,
            "ledger_seconds":            0.0,
            "response_assembly_seconds": 0.0,
            "total_seconds":             0.0,
        },
    }