"""
capture_routing.py — Mode B endpoint for MoE routing capture.

Wraps the C4-equivalent routing path: top-k expert indices and weights
across all 75 MoE layers of GLM-5.1-FP8 (layers 3..77; 0..2 are dense and
have no routing payload).
"""

import syft as sy


_ENDPOINT_ID = "prepilot.capture_routing"


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

    # GLM-5.1: dense layers 0..2, MoE layers 3..77. Routing is undefined
    # on dense layers; sanitise out anything <3 or >77.
    safe_layers = layers or list(range(3, 78))
    safe_layers = [int(L) for L in safe_layers if 3 <= int(L) <= 77]
    if not safe_layers:
        return {
            "error": "invalid_layers",
            "endpoint": "prepilot.capture_routing",
            "detail": "layers must be a non-empty list of ints in [3, 77]",
        }
    xargs = {"output_routing": safe_layers}

    return call_endpoint(
        context,
        endpoint_id="prepilot.capture_routing",
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
        endpoint_id="prepilot.capture_routing",
        prompt=prompt,
    )


def build_endpoint() -> sy.TwinAPIEndpoint:
    return sy.TwinAPIEndpoint(
        path=_ENDPOINT_ID,
        description=(
            "Capture MoE top-k expert routing across the 75 MoE layers of "
            "GLM-5.1-FP8 (layers 3..77). Top-k = 8 per the model config; "
            "returned as int16 indices + bf16 weights inside the encoder "
            "bundle. Default layer set is all 75; auditor may pass a subset."
        ),
        mock_function=_mock,
        private_function=_private,
    )
