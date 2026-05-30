"""Gradient backend — HF transformers + autograd.

Lifted from the standalone Phase 2 sidecar (gradient_server.py), restructured
as a reusable class so the FastAPI app in _gradient_entry.py (and any future
co-located process) can call it without round-tripping through HTTP.

Constraint that drives the design: vLLM V1's worker runs under
torch.inference_mode(), which strictly precludes autograd graph construction.
Same-process backward inside the vLLM worker is structurally impossible.
This backend therefore runs the model independently via transformers, with
quantization handled by whichever loader the model's config declares
(FineGrainedFP8 for GLM-5.1-FP8; compressed-tensors for compressed-tensors
checkpoints; bf16 fallback). All parameters are frozen at load time — only
inputs_embeds carries gradient.

The backend is deploy-mode exclusive: a container booted with
VLLM_LENS_BACKEND=gradient loads ONLY this backend, not vLLM. The memory
math is in PHASE2_LESSONS_LEARNED.md: two simultaneous copies of GLM-5.1-FP8
do not fit on 8xH200.
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Optional

import torch
import torch.nn.functional as F
from vllm_lens._helpers._serialize import serialize_tensor


logger = logging.getLogger(__name__)


class GradientBackendError(RuntimeError):
    """Per-request failure — surface as 500 from the HTTP layer."""


class GradientLoadError(RuntimeError):
    """Load-time failure — surface as 503 / fail container startup."""


class GradientBackend:
    """Owns the HF model and runs one backward pass per call.

    Constructor blocks until the model is on GPUs and params frozen. After
    that, compute_input_gradients() is single-tenant safe; the FastAPI app
    in _gradient_entry.py runs a single worker, which is the contract.

    Notes for future extension (deliberately not in v1):
      - Layer-wise grad: would need additional hooks + a per-layer leaf.
      - Per-position target loss: trivial extension via target_token_id.
      - Batched requests: would need padding + per-row backward. The
        sidecar contract has always been single-request; keep it.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_prompt_tokens: int = 2048,
        gpu_max_memory_gib: int = 130,
        expected_gpus: int = 8,
    ) -> None:
        self.model_path = model_path
        self.max_prompt_tokens = max_prompt_tokens
        self._load(gpu_max_memory_gib, expected_gpus)

    # ---- load -----------------------------------------------------------

    def _load(self, gpu_max_memory_gib: int, expected_gpus: int) -> None:
        # Imported lazily so the vLLM-mode deploy doesn't pay for transformers.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus == 0:
            raise GradientLoadError("No CUDA GPUs visible. Check container --gpus all.")
        if n_gpus != expected_gpus:
            logger.warning("Expected %d GPUs, found %d", expected_gpus, n_gpus)

        logger.info("Loading tokenizer from %s", self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # FineGrainedFP8 rejects any cpu/disk entry in device_map; force
        # GPU-only sharding. If the quantization path didn't engage (weights
        # blow up to bf16), accelerate fails fast at load with an allocation
        # error rather than mid-shard OOM during the first backward.
        max_memory: dict[Any, str] = {
            i: f"{gpu_max_memory_gib}GiB" for i in range(n_gpus)
        }
        max_memory["cpu"] = "0GiB"
        logger.info(
            "Loading model from %s (device_map=auto, max_memory=%dGiB/GPU, "
            "expected_gpus=%d)",
            self.model_path, gpu_max_memory_gib, n_gpus,
        )

        t0 = time.time()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            raise GradientLoadError(
                f"AutoModelForCausalLM.from_pretrained failed: {e!r}"
            ) from e
        logger.info(
            "Model loaded in %.1fs. Devices: %s",
            time.time() - t0, self._summarize_device_map(),
        )

        self._log_quantization_status()
        self._log_gpu_memory("post-load")

        # Freeze every parameter — we only need inputs_embeds.grad.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()  # disable dropout; autograd remains live on leafs

        embed = self.model.get_input_embeddings()
        embed_param = next(embed.parameters())
        logger.info(
            "Embedding: %s, dtype=%s, device=%s, weight_shape=%s",
            type(embed).__name__, embed_param.dtype, embed_param.device,
            list(embed_param.shape),
        )

    # ---- diagnostics ----------------------------------------------------

    def _summarize_device_map(self) -> str:
        if not hasattr(self.model, "hf_device_map"):
            return "<no hf_device_map>"
        counts: dict[str, int] = {}
        for _, dev in self.model.hf_device_map.items():
            counts[str(dev)] = counts.get(str(dev), 0) + 1
        return ", ".join(f"{d}: {n} modules" for d, n in sorted(counts.items()))

    def _log_quantization_status(self) -> None:
        """Check both quantization paths. FineGrainedFP8 (HF native, GLM-5.1-FP8)
        and compressed-tensors. If neither engaged, the bf16 blowup will cause
        backward OOM — surface it now, before the first request.

        Diagnostic-only: replaces the version in gradient_server.py that
        checked CompressedLinear alone (cosmetic warning bug fix from
        PHASE2_REFERENCE §10).
        """
        n_fp8 = 0
        try:
            from transformers.integrations.finegrained_fp8 import (
                FP8Linear, FP8Experts,
            )
            n_fp8 = sum(
                1 for m in self.model.modules()
                if isinstance(m, (FP8Linear, FP8Experts))
            )
        except Exception:
            pass

        n_ct = 0
        try:
            from compressed_tensors.linear.compressed_linear import CompressedLinear
            n_ct = sum(
                1 for m in self.model.modules() if isinstance(m, CompressedLinear)
            )
        except Exception:
            pass

        n_linear_like = sum(
            1 for m in self.model.modules() if "Linear" in type(m).__name__
        )
        logger.info(
            "Linear-like modules: total=%d  FineGrainedFP8=%d  CompressedLinear=%d",
            n_linear_like, n_fp8, n_ct,
        )
        if n_fp8 == 0 and n_ct == 0:
            logger.warning(
                "Neither FineGrainedFP8 nor compressed-tensors engaged. "
                "Weights are bf16 (or smaller). Backward will likely OOM. "
                "Check the model's quantization_config and "
                "transformers/compressed-tensors versions."
            )

    def _log_gpu_memory(self, tag: str) -> None:
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            used = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            logger.info(
                "GPU %d %s: %.1f GiB allocated, %.1f GiB reserved",
                i, tag, used, reserved,
            )

    # ---- inference ------------------------------------------------------

    def compute_input_gradients(
        self,
        messages: list[dict],
        target_token_id: Optional[int] = None,
    ) -> dict:
        """One backward pass. Returns the Phase 2 response payload shape.

        Args:
            messages: list of {role, content} dicts in chat-template form.
            target_token_id: target for next-token NLL loss. If None, uses
                argmax(last_logits) — saliency for "what the model chose
                to say."

        Returns:
            {"gradients": {"input_embeddings": <bf16 blob>},
             "diagnostics": {loss, target_token_id, target_token,
                             prompt_tokens, fwd_seconds, bwd_seconds,
                             total_seconds}}
        """
        t_total = time.time()

        # ---- tokenize ----------------------------------------------------
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                chat_template_kwargs={"enable_thinking": False},
            )
        except Exception as e:
            raise GradientBackendError(f"tokenization failed: {e}")

        # apply_chat_template returns Tensor or BatchEncoding depending on
        # the transformers version; normalize before .shape access. This
        # was hot-patched in v0.1.15-grad and is now baked in.
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]

        n_prompt = int(input_ids.shape[-1])
        if n_prompt > self.max_prompt_tokens:
            raise GradientBackendError(
                f"prompt {n_prompt} tokens > max_prompt_tokens={self.max_prompt_tokens}"
            )

        embed = self.model.get_input_embeddings()
        embed_device = next(embed.parameters()).device
        input_ids = input_ids.to(embed_device)

        # inputs_embeds as autograd leaf. embed() runs without param grads
        # (frozen); detach() makes it a leaf; requires_grad_(True) registers
        # it with autograd so loss.backward() populates .grad.
        inputs_embeds = embed(input_ids).detach().requires_grad_(True)

        # ---- forward -----------------------------------------------------
        t_fwd = time.time()
        try:
            outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        except Exception as e:
            logger.exception("forward pass failed")
            raise GradientBackendError(
                f"forward pass failed: {e!r}\n{traceback.format_exc()}"
            )
        fwd_secs = time.time() - t_fwd

        logits = outputs.logits  # (1, seq_len, vocab_size)
        last_logits = logits[0, -1]

        target = (
            int(target_token_id)
            if target_token_id is not None
            else int(last_logits.argmax().item())
        )
        loss = -F.log_softmax(last_logits, dim=-1)[target]

        # ---- backward ----------------------------------------------------
        t_bwd = time.time()
        try:
            loss.backward()
        except Exception as e:
            logger.exception("backward pass failed")
            raise GradientBackendError(
                f"backward pass failed: {e!r}\n{traceback.format_exc()}"
            )
        bwd_secs = time.time() - t_bwd

        if inputs_embeds.grad is None:
            raise GradientBackendError(
                "inputs_embeds.grad is None after backward "
                "(autograd graph not connected through quantized modules?)"
            )
        if not torch.isfinite(inputs_embeds.grad).all():
            raise GradientBackendError(
                "inputs_embeds.grad contains non-finite values"
            )

        grad = inputs_embeds.grad[0].detach()  # (seq_len, hidden_size)
        loss_val = float(loss.detach().cpu())
        target_str = self.tokenizer.decode([target], skip_special_tokens=False)

        grad_payload = serialize_tensor(grad)

        total_secs = time.time() - t_total
        logger.info(
            "saliency: n_prompt=%d fwd=%.2fs bwd=%.2fs total=%.2fs "
            "loss=%.4f target=%r",
            n_prompt, fwd_secs, bwd_secs, total_secs, loss_val, target_str,
        )

        return {
            "gradients": {"input_embeddings": grad_payload},
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