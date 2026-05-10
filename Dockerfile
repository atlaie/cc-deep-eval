FROM vllm/vllm-openai:v0.20.0-cu130-ubuntu2404@sha256:aff65d7198dd284c37dd0a18a606544cc5e92bfb0d5eb608b77e8b8f1c6b8b0d

# Install vllm-lens 1.1.0 to register entry points
RUN pip install --no-cache-dir vllm-lens==1.1.0

# Overwrite with Phase 2 modified files (routing capture extension)
COPY vllm_lens_ext/vllm_lens /usr/local/lib/python3.12/dist-packages/vllm_lens# Sun May 10 23:26:53 CEST 2026
