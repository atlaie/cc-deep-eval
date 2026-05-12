"""
Pure-math attention-stats compute for DeepseekV2-style MLA attention.

Extracted from the hook plumbing in `_worker_ext.py` so it can be unit-tested
locally without a vLLM install or a GPU. The hook body in `_worker_ext.py`
becomes a thin wrapper that calls `compute_attention_stats(self_attn, hs)`
and stages the (entropy, rowmax, top10_mass) tuple into `_attn_stats_buffers`.

Approximation: computes Q_nope @ K_nope^T (the no-RoPE channel) and ignores
the k_pe / q_pe RoPE channel. Adequate for entropy/concentration shape
(Phase 3 overhead-measurement use case) but NOT a faithful reproduction of
the model's actual attention weights — see EXTENSION_PHASE2.md §10.

Prefill-only: applies a lower-triangular causal mask. Decode-step usage
would need a different mask construction (current-token vs. KV cache).

Q-projection layout:
  Standard path (q_lora_rank is None):
      hidden → q_proj                          (ColumnParallelLinear)
  LoRA path, separate projections:
      hidden → q_a_proj → q_a_layernorm → q_b_proj
  LoRA path, fused (GLM 5.1 / DeepSeek V2 Tinfoil fork):
      hidden → fused_qkv_a_proj                (DeepSeekV2FusedQkvAProjLinear)
             output[:q_lora_rank]      → q_a_layernorm → q_b_proj
             output[q_lora_rank:]      → kv_a_layernorm split (replaces kv_a_proj_with_mqa)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

LOG2 = math.log(2.0)


def _call(linear: Any, x: torch.Tensor) -> torch.Tensor:
    """vLLM's ReplicatedLinear / ColumnParallelLinear return `(out, bias)`.
    Plain `nn.Linear` returns a tensor. Handle both."""
    out = linear(x)
    return out[0] if isinstance(out, tuple) else out


def compute_attention_stats(
    self_attn: Any,
    hidden_states: torch.Tensor,
    *,
    causal: bool = True,
) -> dict[str, torch.Tensor]:
    """Re-run Q_nope and K_nope projections, build (H, T, T) softmax, return per-head stats.

    Args:
        self_attn: DeepseekV2MLAAttention-like module. Required attributes vary
            by layout (see module docstring). Common to all layouts:
            num_local_heads, qk_head_dim, qk_nope_head_dim, kv_lora_rank,
            v_head_dim, kv_a_layernorm, kv_b_proj.
            Fused layout (GLM 5.1): additionally fused_qkv_a_proj, q_a_layernorm,
            q_b_proj, q_lora_rank.
            Standard LoRA layout: q_a_proj, q_a_layernorm, q_b_proj,
            kv_a_proj_with_mqa.
        hidden_states: (T, hidden_size). The input to `self_attn.forward` — i.e.
            post-input-layernorm activations, before the attention projection.
        causal: apply lower-triangular mask before softmax.

    Returns:
        Dict with three (H, T) float32 tensors on the same device as `hidden_states`:
            entropy:    Shannon entropy of each row's attention distribution, in bits.
                        Range [0, log2(T)]. NaN-safe (0 * log(0) := 0).
            rowmax:     Max attention weight per row. Range [0, 1].
            top10_mass: Sum of top-ceil(T/10) attention weights per row. Range [0, 1].
                        Always >= rowmax. For T <= 10, equals rowmax exactly.
    """
    if hidden_states.dim() != 2:
        raise ValueError(
            f"hidden_states must be (T, hidden_size); got {tuple(hidden_states.shape)}"
        )

    T = hidden_states.shape[0]
    H = self_attn.num_local_heads
    d_nope = self_attn.qk_nope_head_dim
    L = self_attn.kv_lora_rank
    V = self_attn.v_head_dim
    qk_d = self_attn.qk_head_dim  # = qk_nope_head_dim + qk_rope_head_dim

    # ── Q path and KV compressed path ───────────────────────────────────────
    #
    # Three layouts depending on what projections the module exposes:
    #
    # Layout A — fused (GLM 5.1 / Tinfoil DeepSeek V2 fork):
    #   fused_qkv_a_proj output: (T, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
    #   split at q_lora_rank → q_c and kv_a_out
    #
    # Layout B — standard separate LoRA projections:
    #   q_a_proj + kv_a_proj_with_mqa
    #
    # Layout C — non-LoRA (q_lora_rank is None):
    #   q_proj + kv_a_proj_with_mqa
    #
    if hasattr(self_attn, "fused_qkv_a_proj"):
        # Layout A: fused projection used by GLM 5.1 / Tinfoil fork.
        q_lora_rank = self_attn.q_lora_rank
        fused_out = _call(self_attn.fused_qkv_a_proj, hidden_states)
        # (T, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
        q_c = fused_out[..., :q_lora_rank]       # (T, q_lora_rank)
        kv_a_out = fused_out[..., q_lora_rank:]  # (T, kv_lora_rank + qk_rope_head_dim)
        q_c = self_attn.q_a_layernorm(q_c)
        q = _call(self_attn.q_b_proj, q_c)       # (T, H * qk_head_dim)

    elif hasattr(self_attn, "q_a_proj"):
        # Layout B: standard separate LoRA projections.
        q_c = _call(self_attn.q_a_proj, hidden_states)
        q_c = self_attn.q_a_layernorm(q_c)
        q = _call(self_attn.q_b_proj, q_c)
        kv_a_out = _call(self_attn.kv_a_proj_with_mqa, hidden_states)

    elif hasattr(self_attn, "q_proj"):
        # Layout C: non-LoRA direct projection.
        q = _call(self_attn.q_proj, hidden_states)  # (T, H * qk_head_dim)
        kv_a_out = _call(self_attn.kv_a_proj_with_mqa, hidden_states)

    else:
        raise AttributeError(
            "compute_attention_stats: cannot find Q projection on self_attn. "
            f"Available attrs: {[a for a in dir(self_attn) if 'proj' in a]}"
        )

    q = q.view(T, H, qk_d)
    q_nope = q[..., :d_nope]                              # (T, H, d_nope)

    # ── KV compressed path (shared across all layouts) ─────────────────────
    kv_c = kv_a_out[..., :L]
    # k_pe = kv_a_out[..., L:]  # RoPE channel, intentionally ignored
    kv_c = self_attn.kv_a_layernorm(kv_c)
    kv = _call(self_attn.kv_b_proj, kv_c)                 # (T, H * (d_nope + V))
    kv = kv.view(T, H, d_nope + V)
    k_nope = kv[..., :d_nope]                             # (T, H, d_nope)
    # v = kv[..., d_nope:]  # not needed for attention-weight stats

    # ── Scores. Promote to fp32 for numerical stability of softmax/entropy. ──
    q_h = q_nope.permute(1, 0, 2).float()                 # (H, T, d_nope)
    k_h = k_nope.permute(1, 0, 2).float()                 # (H, T, d_nope)
    scale = 1.0 / math.sqrt(d_nope)
    scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scale  # (H, T, T)

    if causal and T > 1:
        mask = torch.triu(
            torch.ones(T, T, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(mask, float("-inf"))

    log_probs = F.log_softmax(scores, dim=-1)             # (H, T, T)
    probs = log_probs.exp()

    # Shannon entropy in bits.
    # Causal masking gives log_probs = -inf at masked positions where probs = 0.
    # 0 * (-inf) = NaN in IEEE 754; the analytic limit (0 log 0 := 0) is what we
    # want. Replace -inf entries in log_probs with 0 *before* multiplying — then
    # 0 * 0 = 0 and masked terms vanish cleanly without post-hoc nan_to_num.
    safe_logp = torch.where(
        torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs)
    )
    entropy = -(probs * safe_logp).sum(dim=-1) / LOG2     # (H, T) in bits

    rowmax = probs.max(dim=-1).values                     # (H, T)

    k = max(1, math.ceil(T / 10))
    top_vals, _ = probs.topk(k, dim=-1)
    top10_mass = top_vals.sum(dim=-1)                     # (H, T)

    return {
        "entropy": entropy.float(),
        "rowmax": rowmax.float(),
        "top10_mass": top10_mass.float(),
    }