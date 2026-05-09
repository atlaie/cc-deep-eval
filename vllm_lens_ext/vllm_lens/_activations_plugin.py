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
