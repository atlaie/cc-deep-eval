"""
Worker extension — Phase 2.
Adds MoE routing capture alongside existing residual-stream capture.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_lens._helpers.types import SteeringVector
from vllm_lens._routing_ext import _make_routing_hook

if TYPE_CHECKING:
    from jaxtyping import Float, Int
    from vllm.config import ParallelConfig

logger = logging.getLogger(__name__)
_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)


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
    _routing_buffers: dict = {}   # Phase 2: req_id → layer_idx → [(ids, weights, entropy)]

    def install_hooks(self) -> None:
        """Install residual-stream post-hooks and MoE routing pre-hooks. Idempotent."""
        if self._hooks_installed:
            return
        self._hooks_installed = True
        self._captured_states = {}
        self._steering_data = {}
        self._routing_buffers = {}

        tp_size = self.parallel_config.tensor_parallel_size
        self._should_capture = tp_size <= 1 or self.rank % tp_size == 0

        layers = _get_layers(self.model_runner.model)
        n_routing = 0
        for layer_idx, layer in enumerate(layers):
            if isinstance(layer, PPMissingLayer):
                continue
            layer.register_forward_hook(_make_hook(self, layer_idx))
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "experts"):
                mlp.experts.register_forward_pre_hook(
                    _make_routing_hook(self, layer_idx), with_kwargs=True
                )
                n_routing += 1

        if n_routing:
            logger.info("vllm-lens: installed routing pre-hooks on %d MoE layers", n_routing)

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
