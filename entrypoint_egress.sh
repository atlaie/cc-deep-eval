#!/usr/bin/env bash
# entrypoint_egress.sh — supervisor for vLLM + egress_service.
#
# Starts vLLM on the loopback port and egress_service on the public port.
# Exits if either child dies, so the container restarts cleanly rather
# than running half-up.
#
# Args after `--` are forwarded to vLLM verbatim (model path, --tensor-parallel-size,
# --max-model-len, --gpu-memory-utilization, etc.).  Same arg list the base
# image's entrypoint accepts — just relayed through this wrapper.

set -euo pipefail

# Forward vLLM args from the deploy command.  Pattern: `tinfoil container create
# ... -- <vllm-args>` puts them in $@ after the script's own consumed flags.
VLLM_ARGS=("$@")

cleanup() {
    echo "[entrypoint] shutting down children..."
    kill -TERM "${VLLM_PID:-}" 2>/dev/null || true
    kill -TERM "${EGRESS_PID:-}" 2>/dev/null || true
    wait
}
trap cleanup EXIT INT TERM

# ----- vLLM on loopback -----
echo "[entrypoint] starting vLLM on ${VLLM_LOOPBACK_HOST}:${VLLM_LOOPBACK_PORT}"
python3 -m vllm.entrypoints.openai.api_server \
    --host "${VLLM_LOOPBACK_HOST}" \
    --port "${VLLM_LOOPBACK_PORT}" \
    "${VLLM_ARGS[@]}" \
    &
VLLM_PID=$!
echo "[entrypoint] vLLM PID=${VLLM_PID}"

# ----- egress_service on public port -----
# Wait briefly so the egress service's first /health probe finds vLLM
# listening (otherwise the service comes up first and the laptop-side
# /health gets a 503 for a moment — harmless but noisy).
sleep 2
echo "[entrypoint] starting egress_service on ${EGRESS_HOST}:${EGRESS_PORT}"
python3 /workspace/egress_service.py \
    --host "${EGRESS_HOST}" \
    --port "${EGRESS_PORT}" \
    --vllm-url "${VLLM_LOOPBACK_URL}" \
    --signing-key "${EGRESS_SIGNING_KEY}" \
    --ledger-db "${EGRESS_LEDGER_DB}" \
    --bundle-dir "${EGRESS_BUNDLE_DIR}" \
    --plot-dir "${EGRESS_PLOT_DIR}" \
    &
EGRESS_PID=$!
echo "[entrypoint] egress_service PID=${EGRESS_PID}"

# Wait for either to exit; propagate exit status of the first dead child.
wait -n "${VLLM_PID}" "${EGRESS_PID}" || true
echo "[entrypoint] one child exited; tearing down"
exit 1
