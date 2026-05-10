"""
MoE routing capture extension for vllm-lens (Phase 2, Extension 1).

Hook strategy:
  register_forward_pre_hook (with_kwargs=True) on layer.mlp.experts (FusedMoE).
  Glm4MoE calls: self.experts(hidden_states=hidden_states, router_logits=router_logits)
  FusedMoE.forward signature: (self, hidden_states, router_logits, input_ids=None)
  Confirmed from phase2_recon.py output.

We compute sigmoid(router_logits) + torch.topk in the hook.
GLM 5.1 uses use_grouped_topk=True; simple topk is a faithful approximation
for the benchmark overhead profile. The limitation is documented.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from vllm.forward_context import get_forward_context, is_forward_context_available

if TYPE_CHECKING:
    from vllm_lens._worker_ext import HiddenStatesExtension

logger = logging.getLogger(__name__)


def _routing_hook_inner(
    extension: HiddenStatesExtension,
    layer_idx: int,
    module: torch.nn.Module,
    inputs: tuple,
    output: torch.Tensor,
    args: tuple,
) -> None:
    if not is_forward_context_available():
        return

    router_logits = output  # (num_tokens, n_routed_experts)

    # Sanity gate: GLM-5.1 has 256 experts, allow up to 1024 to be safe
    if router_logits.dim() < 2 or router_logits.shape[-1] > 1024:
        return
    # Glm4MoE calls experts(hidden_states=..., router_logits=...)
    router_logits: torch.Tensor | None = kwargs.get("router_logits")
    if router_logits is None and len(args) > 1:
        router_logits = args[1]
    if router_logits is None:
        return

    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return

    if not getattr(extension, "_should_capture", True):
        return

    req_ids = runner.input_batch.req_ids

    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return

    query_start_loc = None
    if hasattr(attn_metadata, "query_start_loc"):
        query_start_loc = attn_metadata.query_start_loc
    elif isinstance(attn_metadata, dict):
        for _meta in attn_metadata.values():
            if hasattr(_meta, "query_start_loc"):
                query_start_loc = _meta.query_start_loc
                break
    if query_start_loc is None:
        return

    top_k: int = getattr(module, "top_k", 8)

    with torch.no_grad():
        probs = torch.sigmoid(router_logits.float())
        topk_weights, topk_ids = torch.topk(probs, k=top_k, dim=-1)
        eps = 1e-7
        p = probs.clamp(eps, 1.0 - eps)
        entropy = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2()).mean(dim=-1)

    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        if req_state is None or req_state.sampling_params is None:
            continue
        extra = req_state.sampling_params.extra_args
        if not extra:
            continue

        output_routing = extra.get("output_routing")
        if output_routing is None:
            continue
        if isinstance(output_routing, list) and layer_idx not in output_routing:
            continue

        start = int(query_start_loc[i].item())
        end = int(query_start_loc[i + 1].item())

        req_topk_ids = topk_ids[start:end].cpu()
        req_topk_weights = topk_weights[start:end].cpu()
        req_entropy = entropy[start:end].cpu()

        if req_id not in extension._routing_buffers:
            extension._routing_buffers[req_id] = {}
        layer_buf = extension._routing_buffers[req_id]
        if layer_idx not in layer_buf:
            layer_buf[layer_idx] = []
        layer_buf[layer_idx].append((req_topk_ids, req_topk_weights, req_entropy))


def _make_routing_hook(extension, layer_idx):
    def hook(module, inputs, output):
        try:
            _routing_hook_inner(extension, layer_idx, module, inputs, output)
        except Exception:
            logger.warning("vllm-lens routing hook error on layer %d, skipping",
                           layer_idx, exc_info=True)
    return hook