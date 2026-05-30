"""
vLLM general plugin — Phase 2 + Level 2 unification.

Adds output_routing and output_attention_stats handling alongside existing
output_residual_stream. Response gains sibling top-level "routing" and
"attention_stats" keys when the respective xargs are set.

LEVEL 2 UNIFICATION (this version):
  - register() checks VLLM_LENS_BACKEND. If "gradient", returns without
    patching anything — the gradient deploy boots via the `vllm-lens-gradient`
    console script (see vllm_lens/_gradient_entry.py), not via `vllm serve`,
    so this plugin shouldn't activate.
  - _patched_generate raises a clear ValueError if a caller sends
    output_input_gradients=True against a vLLM-mode deploy. The flag is
    a no-op on the vLLM path; clients targeting that capability must
    POST /v1/saliency on a gradient-mode deploy. This makes the API
    surface uniform: every client gets a single GradientRequest schema
    and learns at request time which deploy serves it.

DIAGNOSTIC VERSION — adds four one-shot canaries to /dev/shm to localize
where attention_stats data drops out of the response pipeline. Remove the
canary blocks once the bug is fixed.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
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
_logger = logging.getLogger("vllm_lens")

_original_create_engine_config: Callable | None = None
_original_generate: Callable | None = None
_original_llm_generate: Callable | None = None
_original_completion_response: Callable | None = None
_original_chat_full_generator: Callable | None = None

# One-shot canary tracker (per-process state).
_canary_done: dict[str, bool] = {}


def _canary(name: str, payload: str) -> None:
    """Write one-shot canary to /dev/shm, tagged with pid. Never raises."""
    if _canary_done.get(name):
        return
    _canary_done[name] = True
    try:
        with open(f"/dev/shm/canary_{name}_pid{os.getpid()}.txt", "w") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] pid={os.getpid()} {payload}\n")
    except Exception:
        pass


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


def _merge_attention_stats(states):
    if not states:
        return None
    parts = [_decompress(s) for s in states if s is not None]
    if not parts:
        return None
    return parts[0]["attention_stats"]


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


def _trim_attention_stats(attn_stats, expected_len):
    for key in ("per_head_entropy", "per_head_max", "top10pct_mass"):
        t = attn_stats.get(key)
        if t is not None and t.shape[-1] > expected_len:
            attn_stats[key] = t[..., :expected_len]


def serialize_routing(routing: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_indices":   routing.get("layer_indices", []),
        "topk_ids":        serialize_tensor(routing["topk_ids"]),
        "topk_weights":    serialize_tensor(routing["topk_weights"]),
        "routing_entropy": serialize_tensor(routing["routing_entropy"]),
    }


def serialize_attention_stats(attn_stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_indices":    attn_stats.get("layer_indices", []),
        "per_head_entropy": serialize_tensor(attn_stats["per_head_entropy"]),
        "per_head_max":     serialize_tensor(attn_stats["per_head_max"]),
        "top10pct_mass":    serialize_tensor(attn_stats["top10pct_mass"]),
    }


def _patched_create_engine_config(self, *args, **kwargs):
    if not self.worker_extension_cls:
        self.worker_extension_cls = _WORKER_EXT
    self.enforce_eager = True
    assert _original_create_engine_config is not None
    return _original_create_engine_config(self, *args, **kwargs)


# ─── Level 2: shared error for gradient-on-vllm-mode misroutes ──────────────


class GradientNotSupportedError(ValueError):
    """Raised when output_input_gradients is set against a vLLM-mode deploy.

    The unified client API exposes a single GradientRequest schema; this
    error is how vLLM-mode tells callers to retarget the gradient-mode
    endpoint. The error message includes the canonical alternative path
    so misrouted clients self-correct.
    """


def _reject_gradient_on_vllm_mode(extra: dict[str, Any]) -> None:
    """Pop output_input_gradients and reject if truthy.

    Called once per request inside the patched generate paths. Keeps the
    Level-2 contract: gradient mode is opt-in via deploy variant, never
    silently degraded. Pop (not get) so the flag never propagates into
    vLLM's sampling params and confuse downstream code.
    """
    flag = extra.pop("output_input_gradients", None)
    if not flag:
        return
    raise GradientNotSupportedError(
        "output_input_gradients=True requires a gradient-mode deploy "
        "(VLLM_LENS_BACKEND=gradient). POST to /v1/saliency on that deploy "
        "instead of /v1/chat/completions on this one."
    )


async def _patched_generate(self, prompt, sampling_params, request_id, **kwargs):
    effective_params = sampling_params
    try:
        from vllm.v1.engine import EngineCoreRequest
        if isinstance(prompt, EngineCoreRequest) and prompt.sampling_params is not None:
            effective_params = prompt.sampling_params
    except ImportError:
        pass

    extra = effective_params.extra_args or {}

    # ─── Level 2: reject gradient-on-vllm-mode early ─────────────────────
    # Done before any hook install or RPC dispatch, so a misrouted client
    # gets a clean error instead of a partial side effect.
    _reject_gradient_on_vllm_mode(extra)

    wants_activations     = extra.get("output_residual_stream") is not None
    wants_routing         = extra.get("output_routing") is not None
    wants_attention_stats = extra.get("output_attention_stats") is not None
    steering_vectors      = extra.pop("apply_steering_vectors", None)
    if isinstance(steering_vectors, str):
        steering_vectors = [SteeringVector.model_validate(d) for d in json.loads(steering_vectors)]
    skip_kv_cache = extra.pop("skip_reading_prefix_cache", None)

    # CANARY 1: prove _patched_generate is the live method, and report what it sees.
    _canary("patched_generate_entry",
            f"extra_keys={list(extra.keys())} "
            f"wants_attention_stats={wants_attention_stats} "
            f"request_id={request_id!r}")

    needs_hooks = wants_activations or wants_routing or wants_attention_stats or steering_vectors is not None
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
                if wants_attention_stats:
                    attn_states = await self.collective_rpc("get_attn_stats_data", args=(request_id,))
                    # CANARY 2: did the worker's get_attn_stats_data return data?
                    nonnull = [i for i, s in enumerate(attn_states) if s is not None]
                    _canary("attn_get",
                            f"request_id={request_id!r} "
                            f"num_workers={len(attn_states)} "
                            f"nonnull_indices={nonnull} "
                            f"first_nonnull_len={len(attn_states[nonnull[0]]) if nonnull else None}")
                    attn_stats = _merge_attention_stats(attn_states)
                    if attn_stats is not None:
                        n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                        _trim_attention_stats(attn_stats, n)
                        output.attention_stats = attn_stats
                        _canary("attn_attached_to_output",
                                f"layer_indices={attn_stats.get('layer_indices')} "
                                f"entropy_shape={tuple(attn_stats['per_head_entropy'].shape)}")
            yield output
    finally:
        if steering_vectors is not None:
            await self.collective_rpc("clear_steering_data", args=(request_id,))
        if wants_activations:
            await self.collective_rpc("clear_captured_states", args=(request_id,))
        if wants_routing:
            await self.collective_rpc("clear_routing_data", args=(request_id,))
        if wants_attention_stats:
            await self.collective_rpc("clear_attn_stats_data", args=(request_id,))


def _patched_llm_generate(self, prompts, sampling_params=None, **kwargs):
    if isinstance(sampling_params, Sequence):
        params_list = list(sampling_params)
    elif sampling_params is not None:
        params_list = [sampling_params]
    else:
        params_list = []

    # ─── Level 2: same reject for the offline LLM.generate path ───────────
    # Raises early before any params mutation, so callers see one consistent
    # error regardless of whether they used async (AsyncLLM) or sync (LLM).
    for sp in params_list:
        _reject_gradient_on_vllm_mode(sp.extra_args or {})

    wants_activations     = any((sp.extra_args or {}).get("output_residual_stream") is not None for sp in params_list)
    wants_routing         = any((sp.extra_args or {}).get("output_routing") is not None for sp in params_list)
    wants_attention_stats = any((sp.extra_args or {}).get("output_attention_stats") is not None for sp in params_list)

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

    needs_hooks = wants_activations or wants_routing or wants_attention_stats or bool(steering_payloads)
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
        if wants_attention_stats:
            attn_states = self.collective_rpc("get_attn_stats_data", args=(req_id,))
            attn_stats = _merge_attention_stats(attn_states)
            if attn_stats is not None:
                n = len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1
                _trim_attention_stats(attn_stats, n)
                output.attention_stats = attn_stats

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
        if getattr(res, "attention_stats", None) is not None:
            response.attention_stats = serialize_attention_stats(res.attention_stats)
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

    # CANARY 3: prove _patched_chat_full_generator is the live method, report what it sees on last_output.
    has_attn = (last_output is not None and getattr(last_output, "attention_stats", None) is not None)
    has_routing_out = (last_output is not None and getattr(last_output, "routing", None) is not None)
    _canary("chat_full_gen_entry",
            f"last_output_has_attention_stats={has_attn} "
            f"last_output_has_routing={has_routing_out} "
            f"response_type={type(response).__name__}")

    if last_output is not None and hasattr(response, "model_dump"):
        if getattr(last_output, "activations", None) is not None:
            response.activations = serialize_activations(last_output.activations)
        if getattr(last_output, "routing", None) is not None:
            response.routing = serialize_routing(last_output.routing)
        if getattr(last_output, "attention_stats", None) is not None:
            response.attention_stats = serialize_attention_stats(last_output.attention_stats)
    return response


def register() -> None:
    """vLLM plugin entry. Installed via the `vllm.general_plugins` entry point.

    Level 2: backend selector at the top. If VLLM_LENS_BACKEND=gradient,
    skip every patch — the gradient deploy boots via the `vllm-lens-gradient`
    console script (vllm_lens._gradient_entry:main), not via `vllm serve`.
    If this register() runs anyway (e.g., someone invoked `vllm serve` in a
    gradient-mode container by mistake), log loudly and no-op rather than
    silently double-binding the process to both modes.
    """
    backend = os.environ.get("VLLM_LENS_BACKEND", "vllm").lower()
    if backend == "gradient":
        _logger.warning(
            "VLLM_LENS_BACKEND=gradient but vllm-lens plugin was loaded by "
            "vllm. Skipping vLLM-mode patches. The gradient deploy should "
            "use `vllm-lens-gradient` as the container CMD, not `vllm serve`."
        )
        _canary("register_skipped_gradient_mode", f"VLLM_LENS_BACKEND={backend}")
        return
    if backend != "vllm":
        raise ValueError(
            f"VLLM_LENS_BACKEND={backend!r}; expected 'vllm' or 'gradient'"
        )

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

    completion_patched = False
    chat_patched = False

    try:
        from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
        _original_completion_response = OpenAIServingCompletion.request_output_to_completion_response
        OpenAIServingCompletion.request_output_to_completion_response = _patched_completion_response
        completion_patched = True
    except Exception as e:
        # CANARY 4a: record any failure to patch OpenAIServingCompletion.
        _canary("completion_patch_fail", f"err={type(e).__name__}: {e}")

    try:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        _original_chat_full_generator = OpenAIServingChat.chat_completion_full_generator
        OpenAIServingChat.chat_completion_full_generator = _patched_chat_full_generator
        chat_patched = True
    except Exception as e:
        # CANARY 4b: record any failure to patch OpenAIServingChat.
        _canary("chat_patch_fail", f"err={type(e).__name__}: {e}")

    # CANARY 4: prove register() ran in this process, and which patches landed.
    _canary("register_called",
            f"completion_patched={completion_patched} "
            f"chat_patched={chat_patched}")