"""E3 / EII-2 gradient sidecar.

Loads GLM-5.1-FP8 via transformers + compressed-tensors with
device_map="auto" (pipeline-parallel sharding across 8xH200).
Exposes /v1/saliency, which runs one backward pass per request and
returns input-embedding gradients in the Phase 2 transport format.

This is NOT vLLM. vLLM V1 wraps the forward in torch.inference_mode(),
which strictly precludes autograd (PHASE2_PLAN.md sec 4.3), so a
separate process is structurally required.

Auth is handled by the Tinfoil shim per tinfoil-config.gradient.yml's
authenticated-endpoints list. This process receives only post-auth
requests, so no token validation here.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
import zstandard as zstd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mounted by Tinfoil; mpk-prefix derived from tinfoil-config.gradient.yml.
# Override via env for local smoke tests.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/tinfoil/mpk/mpk-fd70ffcf2c2ca3954546b3105150414bcba35838a6ee1a06c283887ad35287ab",
)
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "glm-5-1")
PORT = int(os.environ.get("PORT", 8001))

# 50-pair ToxicChat prompts are short (median ~30 tokens, max well under 1k).
# Bound to prevent OOM from accidentally massive inputs.
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", 2048))

logger = logging.getLogger("gradient_server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)

_MODEL: Optional[Any] = None
_TOKENIZER: Optional[Any] = None
_ZSTD = zstd.ZstdCompressor(level=1)

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------


def _summarize_device_map(model) -> str:
    """One-line summary of how layers are distributed across GPUs."""
    if not hasattr(model, "hf_device_map"):
        return "<no hf_device_map>"
    counts: dict[str, int] = {}
    for _, dev in model.hf_device_map.items():
        counts[str(dev)] = counts.get(str(dev), 0) + 1
    return ", ".join(f"{d}: {n} modules" for d, n in sorted(counts.items()))


def _load_model_eager() -> None:
    """Blocking load at startup. Healthcheck returns 'loading' until done."""
    global _MODEL, _TOKENIZER

    logger.info("Loading tokenizer from %s", MODEL_PATH)
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    logger.info("Loading model from %s (device_map=auto, dtype=auto)", MODEL_PATH)
    t0 = time.time()
    _MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info(
        "Model loaded in %.1fs. Devices: %s",
        time.time() - t0,
        _summarize_device_map(_MODEL),
    )

    # Freeze every parameter. We only need inputs_embeds.grad, not param grads.
    # Setting requires_grad=False on every weight prevents allocation of
    # per-parameter gradient buffers during backward.
    for p in _MODEL.parameters():
        p.requires_grad_(False)
    _MODEL.eval()  # Disable dropout. autograd remains on for inputs_embeds.

    embed = _MODEL.get_input_embeddings()
    embed_param = next(embed.parameters())
    logger.info(
        "Embedding: %s, dtype=%s, device=%s, weight_shape=%s",
        type(embed).__name__,
        embed_param.dtype,
        embed_param.device,
        list(embed_param.shape),
    )


# ---------------------------------------------------------------------------
# Serialization (matches Phase 2 EXTENSION_PHASE2.md sec 5)
# ---------------------------------------------------------------------------


def _serialize_bf16_tensor(t: torch.Tensor) -> dict:
    """bf16 -> int16 bit-reinterpret -> tobytes -> zstd -> base64.

    Matches Phase 2 transport format exactly. The client-side
    deserialize_tensor in captures.py reads this unchanged.
    """
    if t.dtype != torch.bfloat16:
        t = t.to(torch.bfloat16)
    t = t.detach().contiguous().cpu()
    raw = t.view(torch.int16).numpy().tobytes()
    return {
        "data": base64.b64encode(_ZSTD.compress(raw)).decode("ascii"),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(t.shape),
        "compression": "zstd",
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class SaliencyRequest(BaseModel):
    messages: List[ChatMessage]
    # Override the default target. Default = argmax of last-position logits
    # (saliency for "what the model decided to say"). Pass a token id here
    # to compute saliency for a specific target instead.
    target_token_id: Optional[int] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model_eager()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    ready = _MODEL is not None and _TOKENIZER is not None
    return {"status": "ok" if ready else "loading", "model": SERVED_MODEL_NAME}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    """Minimal compatibility stub for the shim's path list."""
    return {
        "data": [
            {"id": SERVED_MODEL_NAME, "object": "model", "owned_by": "pour-demain"}
        ]
    }


@app.post("/v1/saliency")
async def saliency(req: SaliencyRequest) -> dict[str, Any]:
    """One backward pass. Returns input-embedding gradients.

    Loss: NLL on next-token logits with target = argmax(last-position logits)
    unless `target_token_id` is provided.
    Output: dense (seq_len, hidden_size) bf16 gradient tensor under
    `gradients.input_embeddings`, plus diagnostics.
    """
    if _MODEL is None or _TOKENIZER is None:
        raise HTTPException(503, "model still loading")

    t_total = time.time()

    # --- Tokenize ----------------------------------------------------------
    try:
        input_ids = _TOKENIZER.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as e:
        raise HTTPException(400, f"tokenization failed: {e}")

    n_prompt = int(input_ids.shape[-1])
    if n_prompt > MAX_PROMPT_TOKENS:
        raise HTTPException(
            400,
            f"prompt {n_prompt} tokens > MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}",
        )

    embed = _MODEL.get_input_embeddings()
    embed_device = next(embed.parameters()).device
    input_ids = input_ids.to(embed_device)

    # Build inputs_embeds as an autograd leaf. embed() runs without
    # parameter grads (frozen); detach() makes it a leaf; requires_grad_(True)
    # registers it with autograd so loss.backward() populates .grad on it.
    inputs_embeds = embed(input_ids).detach().requires_grad_(True)

    # --- Forward -----------------------------------------------------------
    t_fwd = time.time()
    try:
        outputs = _MODEL(inputs_embeds=inputs_embeds, use_cache=False)
    except Exception as e:
        logger.exception("forward pass failed")
        raise HTTPException(
            500,
            f"forward pass failed: {e!r}\n{traceback.format_exc()}",
        )
    fwd_secs = time.time() - t_fwd

    logits = outputs.logits  # (1, seq_len, vocab_size)
    last_logits = logits[0, -1]  # (vocab_size,)

    target = (
        int(req.target_token_id)
        if req.target_token_id is not None
        else int(last_logits.argmax().item())
    )

    # -log p(target | prompt) under model's own next-token distribution.
    loss = -F.log_softmax(last_logits, dim=-1)[target]

    # --- Backward ----------------------------------------------------------
    t_bwd = time.time()
    try:
        loss.backward()
    except Exception as e:
        logger.exception("backward pass failed")
        raise HTTPException(
            500,
            f"backward pass failed: {e!r}\n{traceback.format_exc()}",
        )
    bwd_secs = time.time() - t_bwd

    if inputs_embeds.grad is None:
        raise HTTPException(
            500,
            "inputs_embeds.grad is None after backward "
            "(autograd graph not connected through CompressedLinear?)",
        )
    if not torch.isfinite(inputs_embeds.grad).all():
        raise HTTPException(500, "inputs_embeds.grad contains non-finite values")

    grad = inputs_embeds.grad[0].detach()  # (seq_len, hidden_size)
    loss_val = float(loss.detach().cpu())
    target_str = _TOKENIZER.decode([target], skip_special_tokens=False)

    grad_payload = _serialize_bf16_tensor(grad.to("cpu"))

    total_secs = time.time() - t_total
    logger.info(
        "saliency: n_prompt=%d fwd=%.2fs bwd=%.2fs total=%.2fs "
        "loss=%.4f target=%r",
        n_prompt,
        fwd_secs,
        bwd_secs,
        total_secs,
        loss_val,
        target_str,
    )

    return {
        "gradients": {
            "input_embeddings": grad_payload,  # (seq_len, hidden_size) bf16
        },
        "diagnostics": {
            "loss": loss_val,
            "target_token_id": target,
            "target_token": target_str,
            "prompt_tokens": n_prompt,
            "fwd_seconds": fwd_secs,
            "bwd_seconds": bwd_secs,
            "total_seconds": total_secs,
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=None,  # Use the root logging config above.
        access_log=False,
    )
