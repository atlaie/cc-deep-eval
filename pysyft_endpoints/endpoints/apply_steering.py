"""
apply_steering.py — Mode B endpoint for EII-4 steering vector application.

Wraps the steering path of phase3_vllm_driver / vllm-lens
(`apply_steering_vectors` xarg). The auditor supplies a direction vector;
the endpoint constructs the SteeringVector payload and submits to the
encoder, which forwards through vLLM with the steering hook engaged.

Direction transport: the auditor sends a flat list of float32 values
(typically 6144 = GLM-5.1 hidden_size). For Phase 1 this is plain JSON
— ~48 KB per call, acceptable for the experimental matrix. A base64+zstd
path is left for later if call sizes become a bottleneck.

Per PHASE2_REFERENCE §5.2 / §11.1, the `apply_steering_vectors` xargs value
must be JSON-stringified (not a list-of-dicts) because vLLM 0.20.0's
`vllm_xargs` schema rejects nested dict values at the FastAPI boundary.
The endpoint handles the json.dumps inside the function body.
"""

import syft as sy


_ENDPOINT_ID = "prepilot.apply_steering"


@sy.api_endpoint_method()
def _private(
    context,
    prompt: str = "",
    direction: list = None,
    layer: int = 62,
    alpha: float = 1.0,
    sign: int = -1,
    norm_match: bool = True,
    position_indices: list = None,
    max_new_tokens: int = 64,
) -> dict:
    """Apply an auditor-supplied steering direction at one layer.

    Args:
        prompt: input text.
        direction: flat list of floats, length must equal model hidden_size
            (6144 for GLM-5.1-FP8).
        layer: layer index to steer at (default 62, the deepest separation
            point in the brief's RepE analysis).
        alpha: effective steering coefficient when norm_match=True; otherwise
            raw scale.
        sign: -1 pushes toxic-direction toward benign (default for the
            toxic→benign steering recipe); +1 reverses.
        norm_match: rescale the steering vector to match runtime residual
            norm. RepE-norm-matched recipe (recommended).
        position_indices: token positions to steer at; None = all positions.
    """
    import base64
    import json
    import struct

    from pysyft_endpoints.endpoints import _common
    from pysyft_endpoints.endpoints._common import call_endpoint
    # TEMP hot-override of baked constant: this deploy serves the model as
    # "glm-5-1" (tinfoil-config served-model-name), not _common's baked
    # "glm-5-1-fp8". Resolved at call time. Remove once _common.py is
    # corrected and re-baked at the convergence rebuild.
    _common.DEFAULT_MODEL = "glm-5-1"

    GLM51_HIDDEN_SIZE = 6144

    # ---- direction validation ----
    if not isinstance(direction, list) or len(direction) != GLM51_HIDDEN_SIZE:
        return {
            "error": "invalid_direction",
            "endpoint": "prepilot.apply_steering",
            "detail": f"direction must be a list of {GLM51_HIDDEN_SIZE} floats; "
                       f"got len={len(direction) if isinstance(direction, list) else 'non-list'}",
        }
    if not (0 <= int(layer) <= 77):
        return {
            "error": "invalid_layer",
            "endpoint": "prepilot.apply_steering",
            "detail": "layer must be in [0, 77]",
        }
    if int(sign) not in (-1, 1):
        return {
            "error": "invalid_sign",
            "endpoint": "prepilot.apply_steering",
            "detail": "sign must be -1 or +1",
        }

    # ---- build the SteeringVector wire payload ----
    # vllm-lens v1.1.0 SteeringVector schema (see d6_steering.py):
    #   activations:    bf16-as-int16 + zstd + base64 dict
    #   layer_indices:  list[int]
    #   scale:          float
    #   norm_match:     bool
    #   position_indices: Optional[list[int]]
    #
    # bf16 truncation: round-to-nearest-even on the discarded low 16 bits.
    # Same routine as serialize_tensor_bf16 in d6_steering.py.
    import zstandard as zstd

    signed = [float(sign) * float(v) for v in direction]
    # Pack as float32 little-endian, then convert to bf16-bits manually.
    f32_bytes = struct.pack(f"<{GLM51_HIDDEN_SIZE}f", *signed)

    bf16_bytes = bytearray(GLM51_HIDDEN_SIZE * 2)
    for i in range(GLM51_HIDDEN_SIZE):
        # f32 little-endian → 32-bit integer
        bits = int.from_bytes(f32_bytes[i*4 : i*4+4], "little", signed=False)
        # Round-half-to-even on the low 16 bits.
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) >> 16
        # Clamp to uint16 (overflow into exponent is fine; pack as int16 view).
        rounded &= 0xFFFF
        bf16_bytes[i*2 : i*2+2] = rounded.to_bytes(2, "little", signed=False)

    compressed = zstd.ZstdCompressor(level=1).compress(bytes(bf16_bytes))
    activations_dict = {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": [1, GLM51_HIDDEN_SIZE],     # (n_layers_in_payload, hidden_size)
        "compression": "zstd",
    }

    steering_vector = {
        "activations":    activations_dict,
        "layer_indices":  [int(layer)],
        "scale":          float(alpha),
        "norm_match":     bool(norm_match),
    }
    if position_indices is not None:
        steering_vector["position_indices"] = [int(p) for p in position_indices]

    # ---- xargs: must be JSON-stringified at the FastAPI boundary ----
    xargs = {
        "apply_steering_vectors": json.dumps([steering_vector]),
    }

    return call_endpoint(
        context,
        endpoint_id="prepilot.apply_steering",
        prompt=prompt,
        xargs=xargs,
        max_new_tokens=max_new_tokens,
        # Steering runs don't need plots/bundle by default — but keep the
        # encoder's full pipeline on for parity with the other endpoints'
        # timing decomposition. Auditor can override via a richer Phase 2
        # API if needed.
    )


@sy.api_endpoint_method()
def _mock(
    context,
    prompt: str = "",
    direction: list = None,
    layer: int = 62,
    alpha: float = 1.0,
    sign: int = -1,
    norm_match: bool = True,
    position_indices: list = None,
    max_new_tokens: int = 64,
) -> dict:
    from pysyft_endpoints.endpoints._common import zero_filled_mock
    return zero_filled_mock(
        context,
        endpoint_id="prepilot.apply_steering",
        prompt=prompt,
    )


def build_endpoint() -> sy.TwinAPIEndpoint:
    return sy.TwinAPIEndpoint(
        path=_ENDPOINT_ID,
        description=(
            "Apply an auditor-supplied steering direction at a single layer "
            "(default L62, the deepest separation point in the GLM-5.1 RepE "
            "analysis). Direction is a flat list of 6144 floats (model hidden "
            "size). norm_match=True (default) rescales to runtime residual "
            "norm so `alpha` is the effective coefficient. Returns the "
            "encoder bundle from the steered generation, plus the standard "
            "five-stage PySyft timing decomposition."
        ),
        mock_function=_mock,
        private_function=_private,
    )
