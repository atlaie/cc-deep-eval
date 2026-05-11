"""
Worker extension — Phase 2.
Adds MoE routing capture alongside existing residual-stream capture.
"""

from __future__ import annotations

import logging
import math
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer
from vllm_lens.attention_stats import compute_attention_stats
from vllm_lens._helpers.types import SteeringVector

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)
_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)

# GLM 5.1 routing constants. Read from config in a future cleanup.
N_ROUTED_EXPERTS = 256
TOP_K = 8
LOG2 = math.log(2.0)


def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(f"Cannot find decoder layers on {type(model).__name__}.")


def _find_steering_configs(extension, internal_req_id, extra_args):
    results = []
    for external_id, configs in extension._steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in extension._steering_data:
            results.extend(extension._steering_data[steering_id])
    return results


def norm_match(residual, steering, eps=1e-6):
    r_norm = residual.float().norm(dim=-1, keepdim=True)
    v_norm = steering.float().norm(dim=-1, keepdim=True)
    return (steering * (r_norm / (v_norm + eps))).to(residual.dtype)


def _apply_steering(configs, layer_idx, target, start, end, abs_start):
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)
        if vec.dim() == 1:
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            pos_indices = cfg.position_indices if cfg.position_indices is not None else list(range(vec.shape[0]))
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale


def _hook_inner(extension, layer_idx, output):
    if not is_forward_context_available():
        return None
    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    if num_reqs == 0:
        return None
    req_ids = runner.input_batch.req_ids
    ctx = get_forward_context()
    attn_metadata = ctx.attn_metadata
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
        if attn_metadata is None:
            return None
    query_start_loc = None
    for _meta in attn_metadata.values():
        if hasattr(_meta, "query_start_loc"):
            query_start_loc = getattr(_meta, "query_start_loc")
            break
    if query_start_loc is None:
        logger.warning("No attention metadata with query_start_loc found. Skipping.")
        return None

    per_req_steering = []
    needs_steering = False
    for i in range(num_reqs):
        req_id = req_ids[i]
        req_state = runner.requests.get(req_id)
        extra = req_state.sampling_params.extra_args if req_state and req_state.sampling_params else None
        configs = _find_steering_configs(extension, req_id, extra)
        per_req_steering.append(configs)
        if configs:
            needs_steering = True

    modified_output = None
    if needs_steering:
        if isinstance(output, tuple):
            modified_output = (output[0].clone(), output[1])
            target = modified_output[0]
        else:
            modified_output = output.clone()
            target = modified_output
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        for i in range(num_reqs):
            if not per_req_steering[i]:
                continue
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            n_query = end - start
            if seq_lens is not None:
                sl = seq_lens[i]
                sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
                abs_start = int(sl_val - n_query)
            else:
                abs_start = 0
            _apply_steering(per_req_steering[i], layer_idx, target, start, end, abs_start)

    if getattr(extension, "_should_capture", True):
        capture_src = modified_output if modified_output is not None else output
        if isinstance(capture_src, tuple):
            hidden_states = capture_src[0] + capture_src[1] if capture_src[1] is not None else capture_src[0]
        else:
            hidden_states = capture_src
        for i in range(num_reqs):
            req_id = req_ids[i]
            req_state = runner.requests.get(req_id)
            if req_state is None or req_state.sampling_params is None:
                continue
            extra = req_state.sampling_params.extra_args
            if not extra:
                continue
            output_residual_stream = extra.get("output_residual_stream")
            if output_residual_stream is None:
                continue
            if isinstance(output_residual_stream, list) and layer_idx not in output_residual_stream:
                continue
            start = query_start_loc[i].item()
            end = query_start_loc[i + 1].item()
            activation = hidden_states[start:end].cpu()
            if req_id not in extension._captured_states:
                extension._captured_states[req_id] = {}
            layer_states = extension._captured_states[req_id]
            if layer_idx not in layer_states:
                layer_states[layer_idx] = []
            layer_states[layer_idx].append(activation)

    return modified_output


def _make_hook(extension, layer_idx):
    def hook(_module, _input, output):
        try:
            return _hook_inner(extension, layer_idx, output)
        except Exception:
            logger.warning("vllm-lens hook error on layer %d, skipping", layer_idx, exc_info=True)
            return None
    return hook


def _make_gate_routing_hook(
    extension,
    layer_idx: int,
    n_experts: int = N_ROUTED_EXPERTS,
    top_k: int = TOP_K,
) -> Callable:
    """Post-hook for `layer.mlp.gate` (ReplicatedLinear).

    `out` is `(router_logits, bias_or_None)`; `out[0]` is `(num_tokens, n_experts)`.
    Computes top-k expert ids/weights via sigmoid (GLM 5.1's scoring), plus
    softmax-distribution entropy in bits. This is an *approximation* of the model's
    true routing decision (does not apply `e_score_correction_bias`, `num_expert_group`,
    `topk_group`); see EXTENSION_PHASE2.md §7. Adequate for overhead profiling and
    coarse interpretability; not a substitute for ground-truth routing.

    Closure-binds `layer_idx` via default arg to avoid the late-binding-in-loop trap.
    Wrapped in try/except so a bug in instrumentation can never kill the worker.
    `_verified` is per-hook state — emits a one-shot shape/range log on first call.
    """
    state = {"verified": False}

    def hook(_module, _args, out, _li=layer_idx, _ne=n_experts, _tk=top_k):
        try:
            if not is_forward_context_available():
                return None
            if not getattr(extension, "_should_capture", True):
                return None  # gate is ReplicatedLinear → only rank 0 captures
            runner = extension.model_runner
            num_reqs = runner.input_batch.num_reqs
            if num_reqs == 0:
                return None
            req_ids = runner.input_batch.req_ids
            ctx = get_forward_context()
            attn_metadata = ctx.attn_metadata
            if attn_metadata is None:
                return None
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[0]
                if attn_metadata is None:
                    return None
            query_start_loc = None
            for _meta in attn_metadata.values():
                if hasattr(_meta, "query_start_loc"):
                    query_start_loc = getattr(_meta, "query_start_loc")
                    break
            if query_start_loc is None:
                return None

            logits = out[0] if isinstance(out, tuple) else out

            # One-shot startup verification: gate output must be (N, n_experts).
            if not state["verified"]:
                shape = tuple(logits.shape)
                ok = (logits.dim() == 2 and shape[-1] == _ne)
                logger.info(
                    "[gate-verify] L%d shape=%s dtype=%s last_dim_ok=%s (expect %d)",
                    _li, shape, logits.dtype, ok, _ne,
                )
                state["verified"] = True
                if not ok:
                    logger.warning(
                        "[gate-verify] L%d wrong target — quarantining routing capture",
                        _li,
                    )
            # Hard guard on every call (cheap; protects against shape drift).
            if logits.dim() != 2 or logits.shape[-1] != _ne:
                return None

            # Skip work if no request on this batch wants this layer's routing.
            wanted = []
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
                if isinstance(output_routing, list) and _li not in output_routing:
                    continue
                wanted.append((i, req_id))
            if not wanted:
                return None

            # Compute on the full batch once.
            logits_f32 = logits.float()
            scores = torch.sigmoid(logits_f32)
            topk_weights, topk_ids = scores.topk(_tk, dim=-1)
            log_probs = F.log_softmax(logits_f32, dim=-1)
            probs = log_probs.exp()
            entropy_bits = -(probs * log_probs).sum(dim=-1) / LOG2

            for i, req_id in wanted:
                start = int(query_start_loc[i].item())
                end = int(query_start_loc[i + 1].item())
                ids_slice = topk_ids[start:end].to(torch.int16).cpu()
                weights_slice = topk_weights[start:end].to(torch.bfloat16).cpu()
                entropy_slice = entropy_bits[start:end].to(torch.float32).cpu()

                if req_id not in extension._routing_buffers:
                    extension._routing_buffers[req_id] = {}
                layer_dict = extension._routing_buffers[req_id]
                if _li not in layer_dict:
                    layer_dict[_li] = []
                layer_dict[_li].append((ids_slice, weights_slice, entropy_slice))

        except Exception:
            logger.warning(
                "vllm-lens routing hook error on layer %d", _li, exc_info=True
            )
        return None  # post-hook never modifies output

    return hook


# ── Phase 2 E2: Attention stats capture ───────────────────────────────────────
# Approximation note: computes Q_nope @ K_nope^T attention proxy (ignores RoPE
# k_pe and shared-MQA k_pe broadcast). Work shape is representative for Phase 3
# overhead measurement. NOT valid for interpretability claims. Prefill-only.

LOG2_E = math.log2(math.e)  # conversion factor: nats → bits

def _make_attn_pre_hook(extension, layer_idx: int) -> Callable:
    """Pre-hook on self_attn. Uses with_kwargs=True (PyTorch ≥ 2.0) because
    vLLM calls self_attn with keyword args — args is empty in a standard hook.
    Verified on hardware: args_len=0, kwargs_keys=['hidden_states', ...].

    Computes Q_nope @ K_nope^T attention-weight proxy per requesting request,
    stores (entropy, rowmax, top10_mass) in _attn_stats_buffers.
    Gated by output_attention_stats xarg. Prefill-only (causal=True).
    """
    state = {"verified": False}

    def hook(module, args, kwargs, _li=layer_idx):
        try:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[1] if len(args) > 1 else args[0]
            if hidden_states is None:
                logger.warning("attn pre-hook L%d: hidden_states not found in args or kwargs", _li)
                return None

            # One-shot consolidated verification: layer 0 only, one file.
            if not state["verified"]:
                state["verified"] = True
                if _li == 0:
                    try:
                        with open("/dev/shm/attn_verify.txt", "w") as f:
                            f.write(
                                f"type={type(module).__name__}\n"
                                f"hs.shape={tuple(hidden_states.shape)} dtype={hidden_states.dtype}\n"
                                f"num_local_heads={getattr(module, 'num_local_heads', '?')}\n"
                                f"qk_head_dim={getattr(module, 'qk_head_dim', '?')}\n"
                                f"qk_nope_head_dim={getattr(module, 'qk_nope_head_dim', '?')}\n"
                                f"kv_lora_rank={getattr(module, 'kv_lora_rank', '?')}\n"
                                f"v_head_dim={getattr(module, 'v_head_dim', '?')}\n"
                                f"has q_a_proj={hasattr(module, 'q_a_proj')}\n"
                                f"has q_a_layernorm={hasattr(module, 'q_a_layernorm')}\n"
                                f"has q_b_proj={hasattr(module, 'q_b_proj')}\n"
                                f"has kv_a_proj_with_mqa={hasattr(module, 'kv_a_proj_with_mqa')}\n"
                                f"has kv_a_layernorm={hasattr(module, 'kv_a_layernorm')}\n"
                                f"has kv_b_proj={hasattr(module, 'kv_b_proj')}\n"
                            )
                    except Exception:
                        pass

            if not getattr(extension, "_should_capture", True):
                return None
            if not is_forward_context_available():
                return None

            runner = extension.model_runner
            num_reqs = runner.input_batch.num_reqs
            if num_reqs == 0:
                return None
            req_ids = runner.input_batch.req_ids
            ctx = get_forward_context()
            attn_metadata = ctx.attn_metadata
            if attn_metadata is None:
                return None
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[0]
                if attn_metadata is None:
                    return None
            query_start_loc = None
            for _meta in attn_metadata.values():
                if hasattr(_meta, "query_start_loc"):
                    query_start_loc = _meta.query_start_loc
                    break
            if query_start_loc is None:
                return None

            wanted = []
            for i in range(num_reqs):
                req_id = req_ids[i]
                req_state = runner.requests.get(req_id)
                if req_state is None or req_state.sampling_params is None:
                    continue
                extra = req_state.sampling_params.extra_args
                if not extra:
                    continue
                output_attention_stats = extra.get("output_attention_stats")
                if output_attention_stats is None:
                    continue
                if isinstance(output_attention_stats, list) and _li not in output_attention_stats:
                    continue
                wanted.append((i, req_id))
            if not wanted:
                return None

            qsl = query_start_loc.tolist()
            for i, req_id in wanted:
                start = int(qsl[i])
                end = int(qsl[i + 1])
                hs_req = hidden_states[start:end]       # (T_req, hidden_size)
                stats = compute_attention_stats(module, hs_req)  # (H, T_req) each

                if req_id not in extension._attn_stats_buffers:
                    extension._attn_stats_buffers[req_id] = {}
                layer_dict = extension._attn_stats_buffers[req_id]
                if _li not in layer_dict:
                    layer_dict[_li] = []
                layer_dict[_li].append((
                    stats["entropy"].cpu(),
                    stats["rowmax"].cpu(),
                    stats["top10_mass"].cpu(),
                ))

        except Exception:
            logger.warning("attn pre-hook L%d failed", _li, exc_info=True)
        return None  # pre-hook: None = don't modify args/kwargs

    return hook

class HiddenStatesExtension:
    """Mixin injected into vLLM's GPU Worker at runtime."""

    if TYPE_CHECKING:
        model_runner: Any
        rank: int
        parallel_config: ParallelConfig

    _captured_states: dict = {}
    _hooks_installed: bool = False
    _steering_data: dict = {}
    _should_capture: bool = True
    _routing_buffers: dict = {}   # req_id → layer_idx → [(ids, weights, entropy)]
    _attn_q_staging: dict = {}       # transient: layer_idx → Q tensor, cleared by kv_hook
    _attn_stats_buffers: dict = {}   # req_id → layer_idx → [(entropy, max_attn, top10_mass)]
    _attn_num_local_heads: int = 0
    _attn_qk_head_dim: int = 0
    _attn_qk_nope_head_dim: int = 0
    _attn_v_head_dim: int = 0

    def get_attn_stats_data(self, external_req_id: str) -> bytes | None:
        prefix = f"{external_req_id}-"
        for req_id in list(self._attn_stats_buffers):
            if req_id.startswith(prefix):
                layer_dict = self._attn_stats_buffers.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                entropy_l, max_l, top10_l = [], [], []
                for idx in sorted_indices:
                    steps = layer_dict[idx]
                    entropy_l.append(torch.cat([s[0] for s in steps], dim=-1))  # (H, T_total)
                    max_l.append(torch.cat([s[1] for s in steps], dim=-1))
                    top10_l.append(torch.cat([s[2] for s in steps], dim=-1))
                return _ZSTD_COMPRESSOR.compress(pickle.dumps({
                    "attention_stats": {
                        "layer_indices":    sorted_indices,
                        "per_head_entropy": torch.stack(entropy_l, dim=0),    # (L, H, T)
                        "per_head_max":     torch.stack(max_l, dim=0),        # (L, H, T)
                        "top10pct_mass":    torch.stack(top10_l, dim=0),      # (L, H, T)
                    }
                }))
        return None

    def clear_attn_stats_data(self, external_req_id: str) -> None:
        prefix = f"{external_req_id}-"
        for req_id in list(self._attn_stats_buffers):
            if req_id.startswith(prefix):
                del self._attn_stats_buffers[req_id]

    def install_hooks(self) -> None:
        """Install all forward hooks. Idempotent.

        Hook strategy (Phase 2):
        E1 — MoE routing: post-hook on layer.mlp.gate (ReplicatedLinear).
            Output is (router_logits, bias); unpack [0] for logits.
        E2 — Attention stats: post-hook on layer.self_attn
            (DeepseekV2MLAAttention). FlashMLA fuses q_b_proj/kv_b_proj
            into a single kernel — their nn.Module.__call__ is never
            invoked, so hooks on them are silent. The outer self_attn
            module always fires; the hook re-runs projections manually
            to recover Q_nope and K_nope.
        Residual stream: post-hook on the decoder layer (Phase 1, unchanged).

        All diagnostic writes are wrapped in try/except that never propagates.
        Hook bodies are wrapped in try/except per EXTENSION_PHASE2.md §11.
        layer_idx bound via default arg to avoid late-binding-in-loop.
        """
        if self._hooks_installed:
            return
        self._hooks_installed = True

        self._captured_states = {}
        self._steering_data = {}
        self._routing_buffers = {}
        self._attn_q_staging = {}    # unused in new design; kept for compat
        self._attn_stats_buffers = {}

        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        layers = _get_layers(self.model_runner.model)

        # ── Canary 1: confirm rank-0 reached this point ───────────────────────
        if self._should_capture:
            try:
                with open("/dev/shm/install_hooks_ok.txt", "w") as f:
                    f.write(
                        f"rank={self.rank} tp={tp_size} "
                        f"n_layers={len(layers)}\n"
                    )
            except Exception:
                pass  # never propagate

        n_residual = 0
        n_routing = 0
        n_attn = 0

        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue

            # ── Residual stream (Phase 1, unchanged) ─────────────────────────
            layer.register_forward_hook(_make_hook(self, layer_idx))
            n_residual += 1

            # ── MoE routing on gate (Phase 2 E1) ─────────────────────────────
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "gate"):
                mlp.gate.register_forward_hook(
                    _make_gate_routing_hook(self, layer_idx)
                )
                n_routing += 1

            # ── Attention stats on self_attn (Phase 2 E2) ────────────────────
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                attn.register_forward_pre_hook(
                    _make_attn_pre_hook(self, layer_idx), with_kwargs=True
                )
                n_attn += 1

        # ── Canary 2: confirm loop completed and hook counts ──────────────────
        if self._should_capture:
            try:
                with open("/dev/shm/install_hooks_ok.txt", "a") as f:
                    f.write(
                        f"n_residual={n_residual} "
                        f"n_routing={n_routing} "
                        f"n_attn={n_attn}\n"
                    )
            except Exception:
                pass
    
    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        sv_list: list[SteeringVector] = pickle.loads(pickled_data)
        device = next(self.model_runner.model.parameters()).device
        dtype = next(self.model_runner.model.parameters()).dtype
        num_layers = len(_get_layers(self.model_runner.model))
        vectors = []
        for sv in sv_list:
            for idx in sv.layer_indices:
                if idx < 0 or idx >= num_layers:
                    raise ValueError(f"layer_index {idx} out of range [0, {num_layers})")
            vectors.append(sv.model_copy(update={"activations": sv.activations.to(device=device, dtype=dtype)}))
        self._steering_data[key] = vectors

    def clear_steering_data(self, key: str) -> None:
        self._steering_data.pop(key, None)

    def clear_captured_states(self, external_req_id: str) -> None:
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                del self._captured_states[req_id]

    def get_captured_states(self, external_req_id: str) -> bytes | None:
        prefix = f"{external_req_id}-"
        for req_id in list(self._captured_states):
            if req_id.startswith(prefix):
                layer_dict = self._captured_states.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                per_layer = [torch.cat(layer_dict[idx], dim=0) for idx in sorted_indices]
                stacked = torch.stack(per_layer, dim=0)
                return _ZSTD_COMPRESSOR.compress(
                    pickle.dumps({"activations": {"residual_stream": stacked}})
                )
        return None

    def _debug_captured_states_count(self) -> int:
        return len(self._captured_states)

    # ── Phase 2: routing data ─────────────────────────────────────────────────

    def get_routing_data(self, external_req_id: str) -> bytes | None:
        """Retrieve MoE routing data for a completed request and remove it."""
        prefix = f"{external_req_id}-"
        for req_id in list(self._routing_buffers):
            if req_id.startswith(prefix):
                layer_dict = self._routing_buffers.pop(req_id)
                sorted_indices = sorted(layer_dict.keys())
                ids_layers, weights_layers, entropy_layers = [], [], []
                for idx in sorted_indices:
                    steps = layer_dict[idx]
                    ids_layers.append(torch.cat([s[0] for s in steps], dim=0))
                    weights_layers.append(torch.cat([s[1] for s in steps], dim=0))
                    entropy_layers.append(torch.cat([s[2] for s in steps], dim=0))
                return _ZSTD_COMPRESSOR.compress(pickle.dumps({
                    "routing": {
                        "layer_indices": sorted_indices,
                        "topk_ids":        torch.stack(ids_layers, dim=0).to(torch.int16),
                        "topk_weights":    torch.stack(weights_layers, dim=0).to(torch.bfloat16),
                        "routing_entropy": torch.stack(entropy_layers, dim=0),
                    }
                }))
        return None

    def clear_routing_data(self, external_req_id: str) -> None:
        """Remove routing data without returning it (cleanup on abort)."""
        prefix = f"{external_req_id}-"
        for req_id in list(self._routing_buffers):
            if req_id.startswith(prefix):
                del self._routing_buffers[req_id]