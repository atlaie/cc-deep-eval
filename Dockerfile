# Phase 2 derivative image: vllm-lens vendored fork with MoE routing capture.
# Base image is identical to Phase 1.
FROM vllm/vllm-openai:v0.20.0-cu130-ubuntu2404@sha256:aff65d7198dd284c37dd0a18a606544cc5e92bfb0d5eb608b77e8b8f1c6b8b0d

COPY vllm_lens_ext /vllm_lens_ext
RUN pip install --no-cache-dir /vllm_lens_ext