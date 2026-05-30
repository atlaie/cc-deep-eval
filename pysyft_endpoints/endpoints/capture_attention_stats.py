"""
capture_attention_stats.py — Mode B endpoint for attention-statistics capture.

Wraps the attention-stats path of phase3_egress_encoder (per-head entropy +
row-max + top-mass at probe layers). Bundled with residual-stream capture
this matches the brief's "repe_bundle" condition; here it's available as
a standalone endpoint so auditors can isolate the attention payload.
"""

import syft as sy


_ENDPOINT_ID = "prepilot.capture_attention_stats"


@sy.api_endpoint_method()
def _private(
    context,
    prompt: str = "",
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    from pysyft_endpoints.endpoints import _common
    from pysyft_endpoints.endpoints._common import call_endpoint
    # TEMP hot-override of baked constant: this deploy serves the model as
    # "glm-5-1" (tinfoil-config served-model-name), not _common's baked
    # "glm-5-1-fp8". Resolved at call time. Remove once _common.py is
    # corrected and re-baked at the convergence rebuild.
    _common.DEFAULT_MODEL = "glm-5-1"

    safe_layers = layers or [12, 23, 39, 51, 62, 70]
    safe_layers = [int(L) for L in safe_layers if 0 <= int(L) <= 77]
    if not safe_layers:
        return {
            "error": "invalid_layers",
            "endpoint": "prepilot.capture_attention_stats",
            "detail": "layers must be a non-empty list of ints in [0, 77]",
        }
    xargs = {"output_attention_stats": safe_layers}

    return call_endpoint(
        context,
        endpoint_id="prepilot.capture_attention_stats",
        prompt=prompt,
        xargs=xargs,
        max_new_tokens=max_new_tokens,
    )


@sy.api_endpoint_method()
def _mock(
    context,
    prompt: str = "",
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    from pysyft_endpoints.endpoints._common import zero_filled_mock
    return zero_filled_mock(
        context,
        endpoint_id="prepilot.capture_attention_stats",
        prompt=prompt,
    )


def build_endpoint() -> sy.TwinAPIEndpoint:
    return sy.TwinAPIEndpoint(
        path=_ENDPOINT_ID,
        description=(
            "Capture per-head attention statistics (entropy, row-max, "
            "top-mass) at the GLM-5.1 probe layers ([12, 23, 39, 51, 62, 70] "
            "by default). Returns the encoder's Tier-1 bundle of bounded "
            "aggregates and the attention-entropy heatmap plot."
        ),
        mock_function=_mock,
        private_function=_private,
    )
