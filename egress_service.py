#!/usr/bin/env python3
"""
egress_service.py — in-TEE egress pipeline service.

Runs inside the Tinfoil CC container alongside vLLM.  Exposes the white-paper
Tier-1 egress endpoint where the encoder pipeline executes inside the TEE,
keeping raw captures (4-25 MB residual stream + attention stats per request)
from ever crossing the trust boundary.  Only the bounded aggregate + signed
evidence bundle leaves.

Topology (single container, two processes):

    container port 8000  ← public, TLS-terminated by Tinfoil ←─── auditor laptop
                          │
                          └── egress_service (this file)
                              ├── POST /v1/egress_eval        (encoder pipeline)
                              ├── POST /v1/chat/completions   (pass-through)
                              ├── GET  /health                (combined)
                              └── GET  /metrics               (pass-through)
                          │
    container port 8001  ← loopback only
                          └── vllm.entrypoints.openai.api_server
                              ├── POST /v1/chat/completions
                              ├── GET  /health
                              └── GET  /metrics

The pass-through routes mean existing C1/C2/C3/C4/C5 cells (phase3_vllm_driver,
phase3_grad_driver) work against this image without code changes — they hit
the same vLLM API, the proxy is transparent.  The E-cells use /v1/egress_eval.

Why a separate process instead of a vLLM plugin: keeps the encoder
maintenance loop independent of vLLM upgrades, and gives us a clean
"vLLM-only" loopback baseline to compare against if we ever need to
isolate proxy overhead.

Startup is driven by /workspace/entrypoint.sh in the image:

    python3 -m vllm.entrypoints.openai.api_server \\
        --host 127.0.0.1 --port 8001 \\
        --model zai-org/GLM-5.1-FP8 ... &
    python3 /workspace/egress_service.py \\
        --port 8000 \\
        --vllm-url http://127.0.0.1:8001 \\
        --signing-key /workspace/signing_key.pem \\
        --ledger-db /workspace/ledger.sqlite \\
        --plot-dir /workspace/plots \\
        --bundle-dir /workspace/bundles &
    wait

Environment-variable overrides are accepted for all CLI flags so the same
image can run with different config.
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

# In-container deps — these are imported from /workspace/, where the image build
# stages captures.py and phase3_egress_encoder.py.
sys.path.insert(0, "/workspace")
from captures import (  # noqa: E402
    CaptureRequest,
    CaptureResult,
    _estimate_payload_bytes,
    extract_activations,
    extract_attention_stats,
    extract_routing,
)
from phase3_egress_encoder import (  # noqa: E402
    ALL_STAGES,
    EgressPipeline,
    ExportLedger,
    load_or_create_signing_key,
)


SCHEMA_VERSION = "egress-service-v1"
log = logging.getLogger("egress")


# ====================================================================
# Request / response schemas
# ====================================================================

class EgressMeta(BaseModel):
    """Auditor-supplied metadata for ledger + bundle attribution."""

    stages: list[str] = Field(default_factory=list)
    request_id: int = 0
    pair_id: int = 0
    prompt_class: str = ""
    session_id: str = "default"
    include_bundle: bool = True
    """If True, the signed bundle bytes are returned base64-encoded in the
    response.  If False, the bundle is persisted server-side and the response
    contains only the SHA + signature.  For the cost measurement the default
    True is the white-paper-faithful choice — the bundle is the actual
    deliverable that crosses the egress boundary."""


class EgressEvalRequest(BaseModel):
    """Mirror of the OpenAI chat-completion body with an extra 'egress' field."""

    model: str
    messages: list[dict[str, Any]]
    temperature: float = 0.0
    max_tokens: int = 32
    vllm_xargs: dict[str, Any] = Field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    egress: EgressMeta = Field(default_factory=EgressMeta)


class EgressTimingsResponse(BaseModel):
    """Server-side per-stage timing breakdown.  These are the authoritative
    timings — the laptop driver records its own end-to-end wall but trusts
    these for stage decomposition."""

    deserialize_seconds: float
    encoder_total_seconds: float
    aggregate_seconds: float
    plot_seconds: float
    bundle_seconds: float
    ledger_seconds: float
    aggregate_bytes: int
    bundle_bytes: int
    n_plots: int
    stages_run: list[str]
    bundle_sha256: Optional[str] = None
    bundle_signature_hex: Optional[str] = None
    bundle_b64: Optional[str] = None
    raw_payload_bytes: int = 0
    """Size of the raw vLLM response on the loopback link, for reference.
    Auditor never sees this content — it stays inside the TEE."""
    session_totals: Optional[dict[str, int]] = None


class EgressEvalResponse(BaseModel):
    completion: dict[str, Any]
    egress: EgressTimingsResponse


# ====================================================================
# Service state (singletons; constructed once at startup)
# ====================================================================

class ServiceState:
    """Per-process singleton holding the vLLM client, signing key, ledger,
    and current EgressPipeline factory."""

    def __init__(
        self,
        vllm_url: str,
        signing_key_path: Path,
        ledger_db_path: Path,
        bundle_dir: Path,
        plot_dir: Path,
        http_timeout: float,
    ):
        self.vllm_url = vllm_url.rstrip("/")
        self.bundle_dir = bundle_dir
        self.plot_dir = plot_dir
        self.signing_key = load_or_create_signing_key(signing_key_path)
        self.ledger_db_path = ledger_db_path

        # Persistent HTTP client for vLLM loopback.  Loopback is plaintext
        # http://, which is fine because we never leave the container.
        self.http = httpx.AsyncClient(
            timeout=http_timeout, limits=httpx.Limits(max_connections=64),
        )

        # One ExportLedger per session_id (auditor-controlled).  Connections
        # are SQLite-cheap so we open lazily.
        self._ledgers: dict[str, ExportLedger] = {}

    def get_ledger(self, session_id: str) -> ExportLedger:
        if session_id not in self._ledgers:
            self._ledgers[session_id] = ExportLedger(
                self.ledger_db_path, session_id,
            )
        return self._ledgers[session_id]

    def build_pipeline(self, meta: EgressMeta) -> EgressPipeline:
        """Construct an EgressPipeline scoped to a single request.  We
        rebuild per request because the `stages` set is auditor-controlled
        (it's part of the request body) and we want the EgressPipeline
        constructor's dependency check to run on every request as a
        cheap sanity gate."""
        stages = set(meta.stages)
        ledger = self.get_ledger(meta.session_id) if "ledger" in stages else None
        return EgressPipeline(
            stages=stages,
            signing_key=self.signing_key,
            out_dir=self.bundle_dir,
            ledger=ledger,
        )

    async def close(self):
        await self.http.aclose()
        for ledger in self._ledgers.values():
            ledger.close()


STATE: Optional[ServiceState] = None


# ====================================================================
# FastAPI app
# ====================================================================

app = FastAPI(title="egress-service", version=SCHEMA_VERSION)


@app.get("/health")
async def health() -> JSONResponse:
    """Combined health check: service is alive and vLLM loopback is up.
    Auditor drivers poll this same endpoint they always have."""
    if STATE is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    try:
        r = await STATE.http.get(f"{STATE.vllm_url}/health", timeout=5.0)
        vllm_ok = r.status_code == 200
    except Exception as e:
        return JSONResponse(
            {"status": "vllm_unreachable", "error": str(e)},
            status_code=503,
        )
    if not vllm_ok:
        return JSONResponse(
            {"status": "vllm_not_ready", "vllm_status": r.status_code},
            status_code=503,
        )
    return JSONResponse({
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "vllm_url": STATE.vllm_url,
    })


@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_chat_completions(request: Request) -> Response:
    """Pass-through to vLLM so existing C-cells still work."""
    return await _proxy(request, "/v1/chat/completions")


@app.api_route("/v1/saliency", methods=["POST"])
async def proxy_saliency(request: Request) -> Response:
    """Pass-through to vLLM saliency endpoint (gradient sidecar)."""
    return await _proxy(request, "/v1/saliency")


@app.get("/metrics")
async def proxy_metrics() -> Response:
    """Pass-through to vLLM /metrics."""
    if STATE is None:
        raise HTTPException(503, "service starting")
    r = await STATE.http.get(f"{STATE.vllm_url}/metrics", timeout=10.0)
    return Response(content=r.content, media_type="text/plain", status_code=r.status_code)


async def _proxy(request: Request, path: str) -> Response:
    if STATE is None:
        raise HTTPException(503, "service starting")
    body = await request.body()
    # Forward all headers except Host (httpx sets it from the URL).
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
    r = await STATE.http.post(
        f"{STATE.vllm_url}{path}",
        content=body, headers=headers,
    )
    # Strip transfer-encoding so the response framing is clean.
    resp_headers = {
        k: v for k, v in r.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=resp_headers,
        media_type=r.headers.get("content-type"),
    )


# ====================================================================
# /v1/egress_eval — the white-paper path
# ====================================================================

@app.post("/v1/egress_eval", response_model=EgressEvalResponse)
async def egress_eval(req: EgressEvalRequest) -> EgressEvalResponse:
    """Wraps a chat completion with the egress pipeline.

    Wire-level flow:

      1. POST to loopback vLLM /v1/chat/completions with the same body.
      2. Deserialize captures with captures.extract_* (TEE-local).
      3. Run EgressPipeline (aggregate/plot/bundle/ledger as configured).
      4. Return completion (text + usage only) + per-stage timings + signed
         bundle.  The raw `activations` / `attention_stats` / `routing`
         fields are stripped from the completion before it leaves the TEE.

    The auditor's laptop driver records its own end-to-end wall_seconds
    independently; the timings returned here are the authoritative
    in-TEE stage decomposition.
    """
    if STATE is None:
        raise HTTPException(503, "service starting")

    # ---- 1. loopback to vLLM ----
    vllm_body = {
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "extra_body": {
            "vllm_xargs": req.vllm_xargs,
            "chat_template_kwargs": req.chat_template_kwargs,
        },
    }
    # vLLM's OpenAI endpoint accepts top-level vllm_xargs in 0.20.0 — the
    # extra_body wrapping is the OpenAI-client convention.  Try the flat
    # form first; if vLLM rejects, fall back.
    flat = {
        "model": req.model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "vllm_xargs": req.vllm_xargs,
        "chat_template_kwargs": req.chat_template_kwargs,
    }
    vllm_r = await STATE.http.post(
        f"{STATE.vllm_url}/v1/chat/completions",
        json=flat,
        timeout=300.0,
    )
    if vllm_r.status_code != 200:
        log.warning(
            "vLLM returned %d on egress_eval; body=%s",
            vllm_r.status_code, vllm_r.text[:500],
        )
        raise HTTPException(vllm_r.status_code, vllm_r.text)
    raw = vllm_r.json()
    raw_payload_bytes = _estimate_payload_bytes(raw)

    usage = raw.get("usage") or {}
    choices = raw.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    completion_text = message.get("content") or ""
    tokens_in = int(usage.get("prompt_tokens") or 0)
    tokens_out = int(usage.get("completion_tokens") or 0)

    # ---- 2. deserialize captures (in-TEE) ----
    capture_request = CaptureRequest(
        residual_layers=req.vllm_xargs.get("output_residual_stream"),
        routing_layers=req.vllm_xargs.get("output_routing"),
        attention_layers=req.vllm_xargs.get("output_attention_stats"),
    )
    result = CaptureResult(
        prompt="<elided>",  # never used downstream; aggregates are content-agnostic
        text=completion_text,
        raw=raw,
        wall_seconds=0.0,
        payload_bytes=raw_payload_bytes,
        prompt_tokens=tokens_in,
    )

    t_deser_start = time.perf_counter()
    if capture_request.residual_layers is not None:
        result.activations = extract_activations(raw, capture_request.residual_layers)
    if capture_request.attention_layers is not None:
        result.attention_stats = extract_attention_stats(raw)
    if capture_request.routing_layers is not None:
        result.routing = extract_routing(raw)
    deserialize_seconds = time.perf_counter() - t_deser_start

    # ---- 3. encoder pipeline ----
    pipeline = STATE.build_pipeline(req.egress)
    metadata = {
        "request_id": req.egress.request_id,
        "pair_id": req.egress.pair_id,
        "prompt_class": req.egress.prompt_class,
        "session_id": req.egress.session_id,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "raw_payload_bytes": raw_payload_bytes,
    }
    timings = pipeline.run(
        result,
        request_id=req.egress.request_id,
        pair_id=req.egress.pair_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        metadata=metadata,
    )

    # ---- 4. bundle bytes (optional) + session totals ----
    bundle_b64 = None
    bundle_sha = None
    bundle_sig = None
    if "bundle" in req.egress.stages:
        bundle_path = STATE.bundle_dir / "bundles" / f"r{req.egress.request_id:04d}.tar"
        if bundle_path.exists():
            bundle_bytes_disk = bundle_path.read_bytes()
            import hashlib
            bundle_sha = hashlib.sha256(bundle_bytes_disk).hexdigest()
            bundle_sig = STATE.signing_key.sign(bundle_bytes_disk).hex()
            if req.egress.include_bundle:
                bundle_b64 = base64.b64encode(bundle_bytes_disk).decode("ascii")

    session_totals = None
    if "ledger" in req.egress.stages:
        ledger = STATE.get_ledger(req.egress.session_id)
        session_totals = ledger.session_totals()

    # ---- strip raw captures from the completion before returning ----
    # The auditor only sees the completion text + usage.  raw['activations'],
    # raw['attention_stats'], raw['routing'] never leave the TEE; they only
    # contributed to the bounded aggregate.
    completion_clean = {
        "id": raw.get("id"),
        "object": raw.get("object"),
        "created": raw.get("created"),
        "model": raw.get("model"),
        "choices": raw.get("choices"),  # contains text but no tensor blobs
        "usage": raw.get("usage"),
    }
    # Defense-in-depth: walk choices to ensure no instrumentation key
    # snuck into the message envelope (shouldn't happen with vllm-lens
    # v1.1.0, but cheap to enforce).
    for ch in completion_clean.get("choices") or []:
        for k in ("activations", "attention_stats", "routing"):
            ch.pop(k, None)

    return EgressEvalResponse(
        completion=completion_clean,
        egress=EgressTimingsResponse(
            deserialize_seconds=deserialize_seconds,
            encoder_total_seconds=timings.total_seconds,
            aggregate_seconds=timings.aggregate_seconds,
            plot_seconds=timings.plot_seconds,
            bundle_seconds=timings.bundle_seconds,
            ledger_seconds=timings.ledger_seconds,
            aggregate_bytes=timings.aggregate_bytes,
            bundle_bytes=timings.bundle_bytes,
            n_plots=timings.n_plots,
            stages_run=list(timings.stages_run),
            bundle_sha256=bundle_sha,
            bundle_signature_hex=bundle_sig,
            bundle_b64=bundle_b64,
            raw_payload_bytes=raw_payload_bytes,
            session_totals=session_totals,
        ),
    )


# ====================================================================
# CLI / entrypoint
# ====================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("EGRESS_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("EGRESS_PORT", "8000")))
    p.add_argument("--vllm-url",
                   default=os.environ.get("VLLM_LOOPBACK_URL", "http://127.0.0.1:8001"))
    p.add_argument("--signing-key", type=Path,
                   default=Path(os.environ.get("EGRESS_SIGNING_KEY",
                                               "/workspace/signing_key.pem")))
    p.add_argument("--ledger-db", type=Path,
                   default=Path(os.environ.get("EGRESS_LEDGER_DB",
                                               "/workspace/ledger.sqlite")))
    p.add_argument("--bundle-dir", type=Path,
                   default=Path(os.environ.get("EGRESS_BUNDLE_DIR",
                                               "/workspace/bundles")))
    p.add_argument("--plot-dir", type=Path,
                   default=Path(os.environ.get("EGRESS_PLOT_DIR",
                                               "/workspace/plots")))
    p.add_argument("--http-timeout", type=float, default=300.0)
    p.add_argument("--log-level", default="info")
    return p.parse_args()


def main() -> int:
    global STATE
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Pre-create directories so the first request doesn't race them.
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.ledger_db.parent.mkdir(parents=True, exist_ok=True)
    args.signing_key.parent.mkdir(parents=True, exist_ok=True)

    STATE = ServiceState(
        vllm_url=args.vllm_url,
        signing_key_path=args.signing_key,
        ledger_db_path=args.ledger_db,
        bundle_dir=args.bundle_dir,
        plot_dir=args.plot_dir,
        http_timeout=args.http_timeout,
    )
    log.info("egress_service starting on %s:%d, vllm=%s",
             args.host, args.port, args.vllm_url)
    log.info("signing_key=%s ledger=%s bundles=%s plots=%s",
             args.signing_key, args.ledger_db, args.bundle_dir, args.plot_dir)

    uvicorn.run(
        app, host=args.host, port=args.port,
        log_level=args.log_level, access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
