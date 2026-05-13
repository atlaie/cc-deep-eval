"""
captures_gradients.py — Client-side additions for E3 / EII-2 gradient capture.

Designed to be merged into captures.py or imported alongside it. All new
functions and dataclasses mirror the existing routing/attention_stats patterns.

Server side: gradient_server.py exposes /v1/saliency on the prepilot-vllm-lens-grad
deployment. Response schema:

    {
      "gradients":   {"input_embeddings": <bf16 (seq_len, hidden_size) blob>},
      "diagnostics": {"loss": float, "target_token_id": int, "target_token": str,
                      "prompt_tokens": int, "fwd_seconds": float,
                      "bwd_seconds": float, "total_seconds": float}
    }

The blob is the same serialization format as Phase 2 (bf16-as-int16, zstd, base64),
so deserialize_tensor() in captures.py decodes it unchanged.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Import existing captures.py primitives. Adjust if pasting inline.
from captures import (
    CaptureResult,
    GLM51_HIDDEN_SIZE,
    ValueRangeError,
    _estimate_payload_bytes,
    _make_ragged,
    deserialize_tensor,
)


# ===== minimal extensions to existing types ==================================
#
# These two changes go *into* captures.py directly (one line each).
#
# 1) Add to CaptureResult dataclass:
#       gradients: dict[str, np.ndarray] | None = None
#
# 2) Add to CaptureResult.to_meta_json():
#       "has_gradients": self.gradients is not None,
#
# Nothing else in captures.py needs to change — _estimate_payload_bytes already
# walks the "gradients" key (line 219).


# ===== extractor =============================================================

def extract_gradients(raw: dict) -> dict[str, np.ndarray] | None:
    """Input-token gradient payload.

    Returns:
      input_embeddings: fp32, (seq_len, hidden_size)   [reinterpreted from bf16]
    """
    blob = raw.get("gradients")
    if not isinstance(blob, dict):
        return None
    out: dict[str, np.ndarray] = {}
    try:
        for key in ("input_embeddings",):
            if key in blob and isinstance(blob[key], dict):
                out[key] = np.asarray(deserialize_tensor(blob[key]))
    except Exception as e:
        print(f"[warn] gradients deserialize failed: {e}")
        return None
    return out or None


# ===== value-range assertion =================================================

def assert_gradients_valid(
    grads: dict[str, np.ndarray],
    hidden_size: int = GLM51_HIDDEN_SIZE,
) -> None:
    """Per-prompt sanity checks. Cross-prompt degeneracy is checked separately
    via compute_gradient_diagnostics + check_gradients_non_degenerate.

    Catches the "autograd silently disconnected" failure mode that the
    EXTENSION_PHASE2 §11 pattern is meant to prevent.
    """
    g = grads.get("input_embeddings")
    if g is None:
        raise ValueRangeError("gradients payload missing input_embeddings")
    if g.ndim != 2:
        raise ValueRangeError(
            f"gradients shape {g.shape} not 2-D (expected (seq_len, hidden))"
        )
    if g.shape[-1] != hidden_size:
        raise ValueRangeError(
            f"gradients last dim {g.shape[-1]} != hidden_size {hidden_size}"
        )
    if not np.isfinite(g).all():
        raise ValueRangeError("gradients contain NaN/inf")
    # All-zero across every position would mean the backward chain didn't
    # connect to inputs_embeds (e.g. detached at some intermediate step,
    # or CompressedLinear backward is a no-op).
    if float(np.abs(g).max()) == 0.0:
        raise ValueRangeError(
            "gradients are exactly zero across all positions "
            "(autograd path likely not connected through CompressedLinear)"
        )


# ===== cross-prompt diagnostics ==============================================

@dataclass
class GradientDiagnostics:
    """Per-prompt gradient-magnitude statistics across the corpus.

    Three per-prompt scalars:
      total_norm:      ||grad||_F  (Frobenius over (seq_len, hidden_size))
      max_token_norm:  max_t ||grad[t, :]||_2
      mean_token_norm: mean_t ||grad[t, :]||_2

    Cross-prompt mean/std on each. The std is the operative
    non-degeneracy quantity: if the corpus is contrastive (toxic vs benign)
    and the backward is functional, total_norm.std > 0 by a comfortable margin.
    """
    n_prompts: int
    total_norm_mean: float
    total_norm_std: float
    max_token_norm_mean: float
    max_token_norm_std: float
    mean_token_norm_mean: float
    mean_token_norm_std: float
    # Per-prompt arrays for the report, all shape (n_prompts,):
    per_prompt_total_norms: np.ndarray
    per_prompt_max_token_norms: np.ndarray
    per_prompt_mean_token_norms: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "n_prompts": self.n_prompts,
            "total_norm_mean": self.total_norm_mean,
            "total_norm_std": self.total_norm_std,
            "max_token_norm_mean": self.max_token_norm_mean,
            "max_token_norm_std": self.max_token_norm_std,
            "mean_token_norm_mean": self.mean_token_norm_mean,
            "mean_token_norm_std": self.mean_token_norm_std,
            "per_prompt_total_norms": self.per_prompt_total_norms.tolist(),
            "per_prompt_max_token_norms": self.per_prompt_max_token_norms.tolist(),
            "per_prompt_mean_token_norms": self.per_prompt_mean_token_norms.tolist(),
        }


def compute_gradient_diagnostics(
    results: list[CaptureResult],
) -> GradientDiagnostics | None:
    has = [
        r for r in results
        if r.gradients is not None and "input_embeddings" in r.gradients
    ]
    if not has:
        return None

    total_norms: list[float] = []
    max_tok_norms: list[float] = []
    mean_tok_norms: list[float] = []
    for r in has:
        g = r.gradients["input_embeddings"].astype(np.float32)  # (seq_len, hidden)
        # Per-token L2 norm: (seq_len,)
        tok_norms = np.sqrt((g * g).sum(axis=-1))
        total_norms.append(float(np.linalg.norm(g)))
        max_tok_norms.append(float(tok_norms.max()))
        mean_tok_norms.append(float(tok_norms.mean()))

    tot = np.asarray(total_norms)
    mx = np.asarray(max_tok_norms)
    mn = np.asarray(mean_tok_norms)

    return GradientDiagnostics(
        n_prompts=len(has),
        total_norm_mean=float(tot.mean()),
        total_norm_std=float(tot.std()),
        max_token_norm_mean=float(mx.mean()),
        max_token_norm_std=float(mx.std()),
        mean_token_norm_mean=float(mn.mean()),
        mean_token_norm_std=float(mn.std()),
        per_prompt_total_norms=tot,
        per_prompt_max_token_norms=mx,
        per_prompt_mean_token_norms=mn,
    )


def check_gradients_non_degenerate(
    diag: GradientDiagnostics,
    total_norm_std_min: float | None = None,
    max_token_norm_std_min: float | None = None,
) -> dict[str, Any]:
    """Cross-prompt non-degeneracy. Mirrors check_entropy_non_degenerate.

    A degenerate gradient signal looks like one of:
      - identical magnitude across all prompts (std ~ 0)
      - exploding/vanishing (caught by assert_gradients_valid on a per-prompt basis)
      - constant w.r.t. the input (which would also produce std ~ 0)

    The recommended pattern: first run report-only on a smoke pass, eyeball
    the std distribution, then set thresholds for the full validation run.
    Same convention as E2's entropy thresholds.
    """
    report: dict[str, Any] = {
        "thresholds": {
            "total_norm_std_min": total_norm_std_min,
            "max_token_norm_std_min": max_token_norm_std_min,
        },
        "n_prompts": diag.n_prompts,
        "summary": {
            "total_norm_mean": diag.total_norm_mean,
            "total_norm_std": diag.total_norm_std,
            "max_token_norm_mean": diag.max_token_norm_mean,
            "max_token_norm_std": diag.max_token_norm_std,
            "mean_token_norm_mean": diag.mean_token_norm_mean,
            "mean_token_norm_std": diag.mean_token_norm_std,
        },
    }
    if total_norm_std_min is None and max_token_norm_std_min is None:
        report["status"] = "report-only (no thresholds set)"
        return report

    failures: list[dict[str, Any]] = []
    if total_norm_std_min is not None and diag.total_norm_std <= total_norm_std_min:
        failures.append({
            "criterion": f"total_norm.std > {total_norm_std_min}",
            "observed": diag.total_norm_std,
        })
    if (
        max_token_norm_std_min is not None
        and diag.max_token_norm_std <= max_token_norm_std_min
    ):
        failures.append({
            "criterion": f"max_token_norm.std > {max_token_norm_std_min}",
            "observed": diag.max_token_norm_std,
        })
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    return report


# ===== call function =========================================================
#
# Separate from call_with_capture because:
#   1) different endpoint (/v1/saliency, not /v1/chat/completions),
#   2) different request schema (no vllm_xargs, no openai chat shape),
#   3) different base URL (gradient sidecar, different Tinfoil deployment).
# Uses httpx directly (transitive dep of openai, so already installed).


def call_with_gradient_capture(
    base_url: str,
    api_key: str,
    user_prompt: str,
    timeout: float = 600.0,
    dump_path: Path | None = None,
) -> CaptureResult:
    """One /v1/saliency call. Returns a CaptureResult with `gradients` populated.

    base_url: e.g. "https://glm-5-1-prepilot-grad-<id>.tinfoil.sh"
    api_key:  Tinfoil bearer token; the shim validates it before forwarding.
    """
    t0 = time.perf_counter()
    payload = {"messages": [{"role": "user", "content": user_prompt}]}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/saliency",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        raw = response.json()
    except Exception as e:
        return CaptureResult(
            prompt=user_prompt,
            text="",
            raw={},
            wall_seconds=time.perf_counter() - t0,
            error=f"{type(e).__name__}: {e}",
        )
    wall = time.perf_counter() - t0

    if dump_path is not None:
        dump_path.write_text(json.dumps(raw, indent=2, default=str))

    diagnostics = raw.get("diagnostics") or {}
    prompt_tokens = int(diagnostics.get("prompt_tokens") or 0)
    target_token = str(diagnostics.get("target_token") or "")  # the "text" surrogate

    res = CaptureResult(
        prompt=user_prompt,
        text=target_token,            # surrogate: there's no completion, only a target.
        raw=raw,
        wall_seconds=wall,
        payload_bytes=_estimate_payload_bytes(raw),
        prompt_tokens=prompt_tokens,
    )
    res.gradients = extract_gradients(raw)
    return res


# ===== save ==================================================================

def save_gradients_npz(path: Path, results: list[CaptureResult]) -> None:
    """Per-prompt gradient arrays → .npz. Variable seq_len → object dtype."""
    has = [r for r in results if r.gradients is not None]
    if not has:
        return
    bundles: dict[str, np.ndarray] = {}
    arrs = [
        r.gradients["input_embeddings"]
        for r in has
        if "input_embeddings" in r.gradients
    ]
    if arrs:
        try:
            bundles["input_embeddings"] = np.stack(arrs)  # uniform seq_len
        except ValueError:
            bundles["input_embeddings"] = _make_ragged(arrs)  # variable seq_len
    if bundles:
        np.savez(path, **bundles, allow_pickle=True)
