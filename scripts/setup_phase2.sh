#!/usr/bin/env bash
# setup_phase2.sh — run from the root of cc-deep-eval
# Creates vllm_lens_ext/ (vendored Phase 2 vllm-lens fork),
# updates Dockerfile, and adds phase2_smoke.py to scripts/.
set -euo pipefail

ROOT="$(pwd)"
EXT="$ROOT/vllm_lens_ext"
PKG="$EXT/vllm_lens"
HELPERS="$PKG/_helpers"

echo "[setup] creating directory structure..."
mkdir -p "$HELPERS"

# ── pyproject.toml ────────────────────────────────────────────────────────────
cat > "$EXT/pyproject.toml" << 'EOF'
[project]
name = "vllm-lens"
version = "1.1.0.phase2"
description = "vLLM plugin for interacting with activations during inference (Phase 2 fork)"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "vllm>=0.16.0",
    "zstandard>=0.23.0",
]

[build-system]
build-backend = "hatchling.build"
requires = ["hatchling"]

[tool.hatch.build.targets.wheel]
packages = ["vllm_lens"]

[project.entry-points."vllm.general_plugins"]
activations = "vllm_lens._activations_plugin:register"

[project.entry-points.inspect_ai]
vllm_lens = "vllm_lens._inspect_entry"
EOF

# ── vllm_lens/__init__.py (unchanged from v1.1.0) ────────────────────────────
cat > "$PKG/__init__.py" << 'EOF'
from importlib.metadata import PackageNotFoundError, version

from vllm_lens._helpers._serialize import (
    decode_activations,
    deserialize_tensor,
    serialize_activations,
    serialize_tensor,
)
from vllm_lens._helpers.types import SteeringVector

try:
    __version__ = version("vllm-lens")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "decode_activations",
    "deserialize_tensor",
    "serialize_activations",
    "serialize_tensor",
    "SteeringVector",
    "__version__",
]
EOF

# ── vllm_lens/_inspect_entry.py (unchanged from v1.1.0) ──────────────────────
cat > "$PKG/_inspect_entry.py" << 'EOF'
from inspect_ai.model import modelapi


@modelapi(name="vllm-lens")
def vllm_lens():
    from .inspect_provider import VLLMLensAPI

    return VLLMLensAPI
EOF

# ── vllm_lens/_helpers/__init__.py ────────────────────────────────────────────
touch "$HELPERS/__init__.py"

# ── vllm_lens/_helpers/_serialize.py (unchanged from v1.1.0) ─────────────────
cat > "$HELPERS/_serialize.py" << 'EOF'
"""Tensor serialization helpers for vllm-lens activations."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import torch
import zstandard as zstd

_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=1)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

# torch dtypes that numpy cannot represent natively.
_TORCH_TO_NUMPY_VIEW: dict[torch.dtype, np.dtype[Any]] = {
    torch.bfloat16: np.dtype(np.int16),
}


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Convert a single torch.Tensor to a JSON-serializable base64 dict."""
    t = tensor.detach().cpu()
    original_dtype = str(t.dtype)

    view_dtype = _TORCH_TO_NUMPY_VIEW.get(t.dtype)
    if view_dtype is not None:
        arr = t.view(torch.int16).numpy()
    else:
        arr = t.numpy()

    raw = arr.tobytes()
    compressed = _ZSTD_COMPRESSOR.compress(raw)

    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": str(arr.dtype),
        "original_dtype": original_dtype,
        "shape": list(arr.shape),
        "compression": "zstd",
    }


def deserialize_tensor(d: dict[str, Any]) -> torch.Tensor:
    """Convert a base64 dict back to a torch.Tensor."""
    raw = base64.b64decode(d["data"])

    if d.get("compression") == "zstd":
        raw = _ZSTD_DECOMPRESSOR.decompress(raw)

    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])
    t = torch.from_numpy(arr.copy())

    original_dtype = d.get("original_dtype")
    if original_dtype == "torch.bfloat16":
        t = t.view(torch.bfloat16)
    elif original_dtype == "torch.float16":
        t = t.to(torch.float16)

    return t


def serialize_activations(tensor_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a flat dict of torch tensors to a JSON-serializable form."""
    return {name: serialize_tensor(t) for name, t in tensor_dict.items()}


def decode_activations(response_json: dict[str, Any]) -> dict[str, Any]:
    """Decode base64-encoded activations from an HTTP API response."""
    raw = response_json.get("activations")
    if raw is None:
        return {}
    return {name: deserialize_tensor(encoded) for name, encoded in raw.items()}
EOF

# ── vllm_lens/_helpers/types.py (unchanged from v1.1.0) ──────────────────────
cat > "$HELPERS/types.py" << 'EOF'
"""Pydantic models for vllm-lens steering vectors."""

from __future__ import annotations

from typing import Any, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm_lens._helpers._serialize import deserialize_tensor, serialize_tensor


class SteeringVector(BaseModel):
    """A steering vector that modifies the residual stream during inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: torch.Tensor
    layer_indices: list[int]
    scale: float = 1.0
    norm_match: bool = False
    position_indices: list[int] | None = None

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize_activations(cls, v: Any) -> torch.Tensor:
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(
            f"activations must be a torch.Tensor or a base64 dict, got {type(v)}"
        )

    @field_serializer("activations")
    def _serialize_activations(self, v: torch.Tensor, _info: Any) -> dict[str, Any]:
        return serialize_tensor(v)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.activations.dim() not in (2, 3):
            raise ValueError(
                f"activations must be 2D or 3D, got {self.activations.dim()}D"
            )
        if self.activations.shape[0] != len(self.layer_indices):
            raise ValueError(
                f"activations dim 0 ({self.activations.shape[0]}) must match "
                f"len(layer_indices) ({len(self.layer_indices)})"
            )
        return self

    @property
    def layer_index_map(self) -> dict[int, int]:
        return {li: i for i, li in enumerate(self.layer_indices)}
EOF

# ── vllm_lens/_routing_ext.py (Phase 2 NEW) ──────────────────────────────────
cat > "$PKG/_routing_ext.py" << 'EOF'
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
    args: tuple,
    kwargs: dict,
) -> None:
    if not is_forward_context_available():
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


def _make_routing_hook(extension: HiddenStatesExtension, layer_idx: int) -> Callable:
    def hook(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        try:
            _routing_hook_inner(extension, layer_idx, module, args, kwargs)
        except Exception:
            logger.warning(
                "vllm-lens routing hook error on layer %d, skipping",
                layer_idx,
                exc_info=True,
            )
        return None
    return hook
EOF

# ── vllm_lens/_worker_ext.py (Phase 2 MODIFIED) ──────────────────────────────
cat > "$PKG/_worker_ext.py" << 'PYEOF'
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
PYEOF

# ── vllm_lens/_activations_plugin.py (Phase 2 MODIFIED) ──────────────────────
cat > "$PKG/_activations_plugin.py" << 'PYEOF'
"""
vLLM general plugin — Phase 2.
Adds output_routing handling alongside existing output_residual_stream.
Response gains a sibling top-level "routing" key when output_routing is set.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
import zstandard as zstd

from vllm_lens._helpers._serialize import serialize_activations, serialize_tensor
from vllm_lens._helpers.types import SteeringVector

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.v1.engine.async_llm import AsyncLLM

_WORKER_EXT = "vllm_lens._worker_ext.HiddenStatesExtension"

_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None


def _decompress(s: bytes) -> Any:
    raw = _ZSTD_DECOMPRESSOR.decompress(s) if s[:4] == _ZSTD_MAGIC else s
    return pickle.loads(raw)


def _merge_captured_states(states):
    if not states:
        return None
    parts = [_decompress(s) for s in states if s is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]["activations"]
    merged = torch.cat([p["activations"]["residual_stream"] for p in parts], dim=0)
    return {"residual_stream": merged}


def _merge_routing_data(states):
    if not states:
        return None
    parts = [_decompress(s) for s in states if s is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]["routing"]
    return {
        "layer_indices": sum([p["routing"]["layer_indices"] for p in parts], []),
        "topk_ids":        torch.cat([p["routing"]["topk_ids"] for p in parts], dim=0),
        "topk_weights":    torch.cat([p["routing"]["topk_weights"] for p in parts], dim=0),
        "routing_entropy": torch.cat([p["routing"]["routing_entropy"] for p in parts], dim=0),
    }


def _trim_activations(activations, expected_len):
    rs = activations.get("residual_stream")
    if rs is not None and rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len, :]
    ids = activations.get("input_ids")
    if ids is not None and len(ids) > expected_len:
        activations["input_ids"] = ids[:expected_len]


def _trim_routing(routing, expected_len):
    for key in ("topk_ids", "topk_weights"):
        t = routing.get(key)
        if t is not None and t.shape[1] > expected_len:
            routing[key] = t[:, :expected_len, :]
    t = routing.get("routing_entropy")
    if t is not None and t.shape[1] > expected_len:
        routing["routing_entropy"] = t[:, :expected_len]


def serialize_routing(routing: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_indices":   routing.get("layer_indices", []),
        "topk_ids":        serialize_tensor(routing["topk_ids"]),
        "topk_weights":    serialize_tensor(routing["topk_weights"]),
        "routing_entropy": serialize_tensor(routing["routing_entropy"]),
    }


def _patched_create_engine_config(self, *args, **kwargs):
    if not self.worker_extension_cls:
        self.worker_extension_cls = _WORKER_EXT
    self.enforce_eager = True
    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


async def _patched_generate(self, prompt, sampling_params, request_id, **kwargs):
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = effective_params.extra_args or {}
    wants_activations = extra.get("output_residual_stream") is not None
    wants_routing     = extra.get("output_routing") is not None
    steering_vectors  = extra.pop("apply_steering_vectors", None)
    if isinstance(steering_vectors, str):
        steering_vectors = [SteeringVector.model_validate(d) for d in json.loads(steering_vectors)]
    skip_kv_cache = extra.pop("skip_reading_prefix_cache", None)

    needs_hooks = wants_activations or wants_routing or steering_vectors is not None
    if needs_hooks or skip_kv_cache:
        effective_params.skip_reading_prefix_cache = True
    if needs_hooks and not getattr(self, "_hooks_installed", False):
        await self.collective_rpc("install_hooks")
        setattr(self, "_hooks_installed", True)
    if steering_vectors is not None:
        await self.collective_rpc("set_steering_data", args=(request_id, pickle.dumps(steering_vectors)))

    assert _original_generate is not None
    try:
        async for output in _original_generate(self, prompt, sampling_params, request_id, **kwargs):
            if output.finished:
                if wants_activations:
                    states = await self.collective_rpc("get_captured_states", args=(request_id,))
                    activations = _merge_captured_states(states)
                    if activations is not None:
                        n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                        _trim_activations(activations, n)
                        output.activations = activations
                if wants_routing:
                    routing_states = await self.collective_rpc("get_routing_data", args=(request_id,))
                    routing = _merge_routing_data(routing_states)
                    if routing is not None:
                        n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                        _trim_routing(routing, n)
                        output.routing = routing
            yield output
    finally:
        if steering_vectors is not None:
            await self.collective_rpc("clear_steering_data", args=(request_id,))
        if wants_activations:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
        if wants_routing:
            await self.collective_rpc("clear_routing_data", args=(request_id,))


def _patched_llm_generate(self, prompts, sampling_params=None, **kwargs):
    if isinstance(sampling_params, Sequence):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    wants_activations = any((sp.extra_args or {}).get("output_residual_stream") is not None for sp in params_list)
    wants_routing     = any((sp.extra_args or {}).get("output_routing") is not None for sp in params_list)

    steering_payloads = {}
    for idx, sp in enumerate(params_list):
        extra = sp.extra_args or {}
        vectors = extra.pop("apply_steering_vectors", None)
        if vectors is not None:
            sid = f"_steer_{idx}"
            steering_payloads[sid] = pickle.dumps(vectors)
            if sp.extra_args is None:
                sp.extra_args = {}
            sp.extra_args["_steering_id"] = sid

    any_skip = False
    for sp in params_list:
        if (sp.extra_args or {}).pop("skip_reading_prefix_cache", None):
            any_skip = True

    needs_hooks = wants_activations or wants_routing or bool(steering_payloads)
    if needs_hooks or any_skip:
        for sp in params_list:
            sp.skip_reading_prefix_cache = True
    if needs_hooks and not getattr(self, "_hooks_installed", False):
        self.collective_rpc("install_hooks")
        self._hooks_installed = True
    for sid, payload in steering_payloads.items():
        self.collective_rpc("set_steering_data", args=(sid, payload))

    assert _original_llm_generate is not None
    outputs = _original_llm_generate(self, prompts, sampling_params, **kwargs)

    for output in outputs:
        req_id = output.request_id
        if wants_activations:
            states = self.collective_rpc("get_captured_states", args=(req_id,))
            activations = _merge_captured_states(states)
            if activations is not None:
                n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                _trim_activations(activations, n)
                output.activations = activations
        if wants_routing:
            routing_states = self.collective_rpc("get_routing_data", args=(req_id,))
            routing = _merge_routing_data(routing_states)
            if routing is not None:
                n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                _trim_routing(routing, n)
                output.routing = routing

    for sid in steering_payloads:
        self.collective_rpc("clear_steering_data", args=(sid,))
    return outputs


def _patched_completion_response(self, final_res_batch, *args, **kwargs):
    assert _original_completion_response is not None
    response = _original_completion_response(self, final_res_batch, *args, **kwargs)
    for res in final_res_batch or ():
        if getattr(res, "activations", None) is not None:
            response.activations = serialize_activations(res.activations)
        if getattr(res, "routing", None) is not None:
            response.routing = serialize_routing(res.routing)
        break
    return response


async def _patched_chat_full_generator(self, request, result_generator, *args, **kwargs):
    assert _original_chat_full_generator is not None
    last_output = None

    async def _capturing(gen):
        nonlocal last_output
        async for output in gen:
            last_output = output
            yield output

    response = await _original_chat_full_generator(self, request, _capturing(result_generator), *args, **kwargs)
    if last_output is not None and hasattr(response, "model_dump"):
        if getattr(last_output, "activations", None) is not None:
            response.activations = serialize_activations(last_output.activations)
        if getattr(last_output, "routing", None) is not None:
            response.routing = serialize_routing(last_output.routing)
    return response


def register() -> None:
    global _original_create_engine_config, _original_generate, _original_llm_generate
    global _original_completion_response, _original_chat_full_generator

    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    _original_create_engine_config = EngineArgs.create_engine_config
    EngineArgs.create_engine_config = _patched_create_engine_config

    _original_generate = AsyncLLM.generate
    AsyncLLM.generate = _patched_generate

    _original_llm_generate = LLM.generate
    LLM.generate = _patched_llm_generate

    try:
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        _original_completion_response = OpenAIServingCompletion.request_output_to_completion_response
        OpenAIServingCompletion.request_output_to_completion_response = _patched_completion_response
    except Exception:
        pass

    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        _original_chat_full_generator = OpenAIServingChat.chat_completion_full_generator
        OpenAIServingChat.chat_completion_full_generator = _patched_chat_full_generator
    except Exception:
        pass
PYEOF

# ── Dockerfile ────────────────────────────────────────────────────────────────
cat > "$ROOT/Dockerfile" << 'EOF'
# Phase 2 derivative image: vllm-lens vendored fork with MoE routing capture.
# Base image is identical to Phase 1.
FROM vllm/vllm-openai:v0.20.0-cu130-ubuntu2404@sha256:aff65d7198dd284c37dd0a18a606544cc5e92bfb0d5eb608b77e8b8f1c6b8b0d

COPY vllm_lens_ext /vllm_lens_ext
RUN pip install --no-cache-dir -e /vllm_lens_ext
EOF

# ── scripts/phase2_smoke.py ───────────────────────────────────────────────────
cat > "$ROOT/scripts/phase2_smoke.py" << 'PYEOF'
"""
Phase 2 smoke test — MoE routing capture (Extension 1).

Usage:
  export VLLM_BASE_URL="https://cc-deep-eval.debug.pour-demain.containers.tinfoil.dev/v1"
  export VLLM_API_KEY="<key>"
  python scripts/phase2_smoke.py --n-pairs 3 --routing-layers 3 39 62 --dump-first
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from vllm_lens._helpers._serialize import deserialize_tensor
except Exception:
    import base64 as _b64, numpy as _np
    def deserialize_tensor(d):
        import zstandard as _zstd
        raw = _b64.b64decode(d["data"])
        if d.get("compression") == "zstd":
            raw = _zstd.ZstdDecompressor().decompress(raw)
        arr = _np.frombuffer(raw, dtype=_np.dtype(d["dtype"])).copy().reshape(d["shape"])
        if d.get("original_dtype") == "torch.bfloat16":
            arr = arr.view(_np.uint16).astype(_np.uint32).__lshift__(16).view(_np.float32)
        return arr


def deserialize_routing(raw: dict) -> dict | None:
    blob = raw.get("routing")
    if blob is None:
        return None
    try:
        return {
            "layer_indices":   blob.get("layer_indices", []),
            "topk_ids":        deserialize_tensor(blob["topk_ids"]),
            "topk_weights":    deserialize_tensor(blob["topk_weights"]),
            "routing_entropy": deserialize_tensor(blob["routing_entropy"]),
        }
    except Exception as e:
        print(f"[warn] routing deserialize: {e}")
        return None


def _extract_activations(raw, layers):
    blob = raw.get("activations")
    if blob is None:
        return None
    rs = blob.get("residual_stream")
    if rs is None:
        return None
    try:
        arr = np.asarray(deserialize_tensor(rs))
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return {layer_idx: arr[i] for i, layer_idx in enumerate(layers) if i < arr.shape[0]}
    except Exception as e:
        print(f"[warn] activations deserialize: {e}")
        return None


def build_pairs(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    toxic = ds.filter(lambda r: r["toxicity"] == 1).shuffle(seed=seed).select(range(n))
    benign = ds.filter(lambda r: r["toxicity"] == 0).shuffle(seed=seed).select(range(n))
    return [(toxic[i]["user_input"], benign[i]["user_input"]) for i in range(n)]


def call(client, model, prompt, residual_layers, routing_layers, max_new_tokens, dump_path=None):
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_new_tokens,
        extra_body={
            "vllm_xargs": {
                "output_residual_stream": residual_layers,
                "output_routing": routing_layers,
            },
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.perf_counter() - t0
    raw = response.model_dump()
    if dump_path:
        dump_path.write_text(json.dumps(raw, indent=2, default=str))
        print(f"  [debug] raw response → {dump_path}")
    return {
        "text":       response.choices[0].message.content or "",
        "activations": _extract_activations(raw, residual_layers),
        "routing":     deserialize_routing(raw),
        "elapsed_s":   elapsed,
        "act_bytes":   len(json.dumps(raw.get("activations", {}), default=str).encode()),
        "rout_bytes":  len(json.dumps(raw.get("routing", {}), default=str).encode()),
    }


def check_routing(r, routing_layers, top_k=8):
    routing = r.get("routing")
    if routing is None:
        print("  [FAIL] routing key absent")
        return False
    ids = routing["topk_ids"]
    weights = routing["topk_weights"]
    entropy = routing["routing_entropy"]
    n = len(routing_layers)
    print(f"  topk_ids.shape      = {ids.shape}  (expect [{n}, n_tokens, {top_k}])")
    print(f"  topk_weights.shape  = {weights.shape}")
    print(f"  routing_entropy.shape = {entropy.shape}")
    ok = True
    if ids.shape[0] != n:
        print(f"  [FAIL] expected {n} layers, got {ids.shape[0]}")
        ok = False
    if ids.shape[-1] != top_k:
        print(f"  [FAIL] expected top_k={top_k}, got {ids.shape[-1]}")
        ok = False
    wmin, wmax = float(weights.min()), float(weights.max())
    if not (0.0 <= wmin and wmax <= 1.01):
        print(f"  [FAIL] weights out of [0,1]: min={wmin:.4f} max={wmax:.4f}")
        ok = False
    if float(entropy.min()) < 0:
        print(f"  [FAIL] negative entropy: {entropy.min():.4f}")
        ok = False
    if ok:
        print("  [OK] routing checks passed")
        print(f"       expert ids (layer 0, tok 0):  {ids[0, 0, :]}")
        print(f"       weights    (layer 0, tok 0):  {weights[0, 0, :]}")
        print(f"       entropy    (layer 0, tok 0):  {entropy[0, 0]:.4f} bits")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://localhost:8001/v1"))
    p.add_argument("--api-key",  default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model",    default="glm-5-1")
    p.add_argument("--n-pairs",  type=int, default=3)
    p.add_argument("--residual-layers", type=int, nargs="+", default=[12, 39, 62])
    p.add_argument("--routing-layers",  type=int, nargs="+", default=[3, 39, 62])
    p.add_argument("--max-new-tokens",  type=int, default=32)
    p.add_argument("--out-dir",  type=Path, default=Path("./runs/phase2_smoke"))
    p.add_argument("--dump-first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(f"[smoke] residual layers: {args.residual_layers}")
    print(f"[smoke] routing layers:  {args.routing_layers}")
    pairs = build_pairs(args.n_pairs)
    all_ok, timings, log = True, [], []
    for i, (toxic, benign) in enumerate(pairs):
        for label, prompt in [("toxic", toxic), ("benign", benign)]:
            print(f"\n--- pair {i} / {label} ---")
            dump = args.out_dir / f"raw_p0_{label}.json" if args.dump_first and i == 0 else None
            try:
                r = call(client, args.model, prompt, args.residual_layers,
                         args.routing_layers, args.max_new_tokens, dump_path=dump)
            except Exception as e:
                print(f"  [ERROR] {e}")
                all_ok = False
                continue
            ok = check_routing(r, args.routing_layers)
            all_ok = all_ok and ok
            timings.append(r["elapsed_s"])
            log.append({"pair": i, "label": label, "elapsed_s": r["elapsed_s"],
                        "act_bytes": r["act_bytes"], "rout_bytes": r["rout_bytes"]})
            print(f"  elapsed: {r['elapsed_s']:.2f}s  "
                  f"act: {r['act_bytes']//1024}KB  rout: {r['rout_bytes']//1024}KB")
    print(f"\n[smoke] mean {np.mean(timings):.2f}s  min {min(timings):.2f}s  max {max(timings):.2f}s")
    (args.out_dir / "timing.json").write_text(json.dumps(log, indent=2))
    if all_ok:
        print("\n[smoke] ALL CHECKS PASSED")
    else:
        print("\n[smoke] SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
PYEOF

echo ""
echo "[setup] done. Repo now contains:"
find "$EXT" -name "*.py" -o -name "*.toml" | sort
echo ""
echo "Next:"
echo "  git add vllm_lens_ext/ Dockerfile scripts/phase2_smoke.py"
echo "  git commit -m 'phase2: MoE routing capture (Extension 1)'"
echo "  git push origin main"