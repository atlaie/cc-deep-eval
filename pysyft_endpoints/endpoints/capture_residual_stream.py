"""
capture_residual_stream.py — Mode B endpoint for residual-stream capture.

Wraps phase3_egress_encoder's residual-stream path (the C2-equivalent
of the brief's matrix). The auditor calls:

    client.api.services.prepilot.capture_residual_stream(prompt="...")

and receives a bundle whose 'egress' field matches the
`phase3_egress_driver_v2.EgressRowV2` shape, plus a `pysyft_timings` dict
with the five-stage decomposition (workflow / approval / encoder / ledger /
bundle_return).

Default layer set is GLM-5.1's probe set [12, 23, 39, 51, 62, 70] from
captures.DEFAULT_PROBE_LAYERS. Auditor may override via the `layers` kwarg
but the layer list is sanitised — only ints in [0, 77] accepted.
"""
from __future__ import annotations

import syft as sy


# These have to be module-level constants so the endpoint function body
# (which PySyft serialises) sees them as captured globals.
_ENDPOINT_ID = "prepilot.capture_residual_stream"
_DEFAULT_LAYERS = [12, 23, 39, 51, 62, 70]


def _private(
    context,
    prompt: str,
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    """Private function — runs in the Datasite worker.

    Inline imports because PySyft executes this as a captured callable
    in the worker; module-level imports of pysyft_endpoints aren't
    guaranteed to be visible.
    """
    from pysyft_endpoints.endpoints._common import call_endpoint

    safe_layers = layers or [12, 23, 39, 51, 62, 70]
    # Sanitise: drop anything out of [0, 77] (GLM-5.1 layer index range).
    safe_layers = [int(L) for L in safe_layers if 0 <= int(L) <= 77]
    if not safe_layers:
        return {
            "error": "invalid_layers",
            "endpoint": "prepilot.capture_residual_stream",
            "detail": "layers must be a non-empty list of ints in [0, 77]",
        }
    xargs = {"output_residual_stream": safe_layers}

    return call_endpoint(
        context,
        endpoint_id="prepilot.capture_residual_stream",
        prompt=prompt,
        xargs=xargs,
        max_new_tokens=max_new_tokens,
    )


def _mock(
    context,
    prompt: str = "",
    layers: list = None,
    max_new_tokens: int = 32,
) -> dict:
    from pysyft_endpoints.endpoints._common import zero_filled_mock
    return zero_filled_mock(
        context,
        endpoint_id="prepilot.capture_residual_stream",
        prompt=prompt,
    )


def build_endpoint() -> sy.TwinAPIEndpoint:
    return sy.TwinAPIEndpoint(
        path=_ENDPOINT_ID,
        description=(
            "Capture residual-stream activations at the GLM-5.1 probe layers "
            "([12, 23, 39, 51, 62, 70] by default). Returns the encoder's "
            "Tier-1 bundle (aggregates + plots + signed tar) plus a five-stage "
            "PySyft timing decomposition."
        ),
        mock_function=_mock,
        private_function=_private,
    )
