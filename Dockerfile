# Phase 1 derivative image: Tinfoil's confidential-glm5-1 base + vllm-lens.
#
# Base SHA pinned to exactly what Tinfoil ships in production
# (github.com/tinfoilsh/confidential-glm5-1). Only behavioural difference
# between their image and ours is the vllm-lens plugin, which auto-registers
# via the vllm.general_plugins entry point.
#
# vllm-lens v1.1.0 main runtime deps: datasets, pydantic, vllm, zstandard.
# No torch pin in main deps, so the prebuilt cu130 torch in the base image
# is preserved.

FROM vllm/vllm-openai:v0.20.0-cu130-ubuntu2404@sha256:aff65d7198dd284c37dd0a18a606544cc5e92bfb0d5eb608b77e8b8f1c6b8b0d

RUN pip install --no-cache-dir vllm-lens==1.1.0

# Smoke-test: build fails loudly if vllm-lens is broken against this vllm.
RUN python -c "import vllm_lens; from vllm_lens._helpers._serialize import deserialize_tensor; print('vllm-lens OK')"