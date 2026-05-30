"""Gradient mode entry point — standalone uvicorn app.

Boots when the container's CMD is `vllm-lens-gradient` (set via the
entrypoint.sh switch on VLLM_LENS_BACKEND=gradient). Loads GradientBackend
and mounts /v1/saliency, /health, /v1/models.

No vLLM in this process. The inference_mode constraint that drives this
architecture is documented in _gradient_backend.py's module docstring.

The HTTP schema matches the original gradient_server.py exactly, so Phase 2
clients (scripts/captures_gradients.py → call_with_gradient_capture) continue
to work without changes.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vllm_lens._gradient_backend import (
    GradientBackend,
    GradientBackendError,
    GradientLoadError,
)

# ---- config -----------------------------------------------------------------
# Defaults match the Phase 2 Tinfoil sidecar mount. Override via env.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/tinfoil/mpk/mpk-fd70ffcf2c2ca3954546b3105150414bcba35838a6ee1a06c283887ad35287ab",
)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "glm-5-1")
PORT = int(os.environ.get("PORT", 8001))
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", 2048))
GPU_MAX_MEMORY_GIB = int(os.environ.get("GPU_MAX_MEMORY_GIB", 130))
EXPECTED_GPUS = int(os.environ.get("EXPECTED_GPUS", 8))

logger = logging.getLogger("vllm_lens.gradient")


# ---- HTTP schema ------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class SaliencyRequest(BaseModel):
    """Body for /v1/saliency. Matches Phase 2 sidecar schema 1:1."""
    messages: List[ChatMessage]
    target_token_id: Optional[int] = None


# ---- lifecycle --------------------------------------------------------------

_BACKEND: Optional[GradientBackend] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _BACKEND
    try:
        _BACKEND = GradientBackend(
            MODEL_PATH,
            max_prompt_tokens=MAX_PROMPT_TOKENS,
            gpu_max_memory_gib=GPU_MAX_MEMORY_GIB,
            expected_gpus=EXPECTED_GPUS,
        )
    except GradientLoadError as e:
        # Re-raise: uvicorn exits with non-zero; Tinfoil surfaces the failure.
        # Better than silently serving 503 forever on a misconfigured deploy.
        logger.error("Gradient backend failed to load: %s", e)
        raise
    yield


app = FastAPI(lifespan=lifespan)


# ---- routes -----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    ready = _BACKEND is not None
    return {"status": "ok" if ready else "loading", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    # Minimal OpenAI compatibility — Tinfoil shim's path list expects it.
    return {
        "data": [
            {"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "vllm-lens"}
        ]
    }


@app.post("/v1/saliency")
async def saliency(req: SaliencyRequest) -> dict[str, Any]:
    if _BACKEND is None:
        raise HTTPException(503, "model still loading")
    try:
        return _BACKEND.compute_input_gradients(
            [m.model_dump() for m in req.messages],
            target_token_id=req.target_token_id,
        )
    except GradientBackendError as e:
        raise HTTPException(500, str(e))


# ---- main -------------------------------------------------------------------


def main() -> int:
    """Console script registered as `vllm-lens-gradient` in pyproject.toml."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
