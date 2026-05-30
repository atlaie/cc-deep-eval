"""ADD-TO-EXISTING-FILE — vllm_lens/_helpers/types.py

Append the GradientRequest class below to the existing types.py beside
SteeringVector. Do not replace the whole file; this is purely additive.

Imports needed in the host file (if not already there):
    from typing import Optional
    from pydantic import BaseModel, Field
"""

# ─── BEGIN APPEND TO _helpers/types.py ───────────────────────────────────────

class GradientRequest(BaseModel):
    """Schema for input-embedding gradient (saliency) requests.

    Served by the gradient-mode deploy via POST /v1/saliency. The vLLM-mode
    plugin (vllm_lens._activations_plugin) does NOT serve this schema; sending
    output_input_gradients=True against a vLLM-mode deploy raises
    GradientNotSupportedError with a pointer to the gradient endpoint.

    Loss: NLL on next-token logits with target = argmax(last-position logits)
    unless target_token_id overrides it.

    Returns (in the HTTP response body):
        {
          "gradients": {"input_embeddings": <bf16 blob, (seq_len, hidden)>},
          "diagnostics": {loss, target_token_id, target_token, prompt_tokens,
                          fwd_seconds, bwd_seconds, total_seconds}
        }
    The bf16 blob uses the same {data, dtype, original_dtype, shape,
    compression} schema as SteeringVector.activations and residual_stream
    payloads.

    Reserved for future versions (deliberately omitted from v1):
        - layer_indices: list[int]      # per-layer gradients
        - position_indices: list[int]   # gradient at specific positions only
        - loss_kind: str                # "nll" | "logit" | "kl_to_uniform" | ...
        - return_layer_grads: bool      # include hidden-state gradients
    Keeping v1 minimal so the FastAPI request body matches the original
    sidecar contract exactly. Existing Phase 2 clients see no schema break.
    """

    target_token_id: Optional[int] = Field(
        default=None,
        description=(
            "Token id to compute saliency for. If None, defaults to the "
            "argmax of last-position logits — saliency for what the model "
            "chose to say."
        ),
    )

# ─── END APPEND ──────────────────────────────────────────────────────────────
