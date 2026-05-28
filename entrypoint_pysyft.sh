#!/usr/bin/env bash
# entrypoint_pysyft.sh - three-process supervisor for the PySyft build image.
#
# Launches in order, with a health gate between each stage:
#   1. vLLM on ${VLLM_LOOPBACK_HOST}:${VLLM_LOOPBACK_PORT}  (default 127.0.0.1:8001)
#   2. egress_service on ${EGRESS_HOST}:${EGRESS_PORT}      (default 127.0.0.1:8002)
#   3. PySyft Datasite on 0.0.0.0:${SYFT_PORT}              (default 8000)
#
# Container "$@" is forwarded to vLLM (the existing tinfoil-config command
# array passes --model, --tensor-parallel-size, etc.).
#
# Shutdown: on SIGTERM/SIGINT, kill all three children and wait. If any
# child exits unexpectedly, kill the siblings and exit non-zero so Tinfoil
# marks the container failed (and the diag pattern from the spike can be
# swapped in via SYFT_DIAG=1 if needed).
set -euo pipefail

VLLM_HOST="${VLLM_LOOPBACK_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_LOOPBACK_PORT:-8001}"
EGRESS_HOST_INTERNAL="${EGRESS_HOST:-127.0.0.1}"
EGRESS_PORT_INTERNAL="${EGRESS_PORT:-8002}"
SYFT_PORT="${SYFT_PORT:-8000}"

VLLM_PID=""
EGRESS_PID=""
SYFT_PID=""

shutdown() {
    local sig="${1:-TERM}"
    echo "[entry] shutdown (sig=${sig}); terminating children"
    [[ -n "$SYFT_PID"   ]] && kill -TERM "$SYFT_PID"   2>/dev/null || true
    [[ -n "$EGRESS_PID" ]] && kill -TERM "$EGRESS_PID" 2>/dev/null || true
    [[ -n "$VLLM_PID"   ]] && kill -TERM "$VLLM_PID"   2>/dev/null || true
    wait 2>/dev/null || true
}
trap 'shutdown TERM' SIGTERM
trap 'shutdown INT'  SIGINT

wait_for_url() {
    local url="$1" name="$2" pid="$3" timeout="${4:-600}"
    local deadline=$((SECONDS + timeout))
    while ! curl -sf "$url" >/dev/null; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[entry][FATAL] $name (pid=$pid) exited before becoming ready"
            shutdown TERM
            exit 1
        fi
        if (( SECONDS >= deadline )); then
            echo "[entry][FATAL] $name did not become ready within ${timeout}s"
            shutdown TERM
            exit 1
        fi
        sleep 5
    done
}

# ===== 1. vLLM =============================================================
echo "[entry] starting vLLM on ${VLLM_HOST}:${VLLM_PORT}"
python3 -m vllm.entrypoints.openai.api_server \
    --host "$VLLM_HOST" --port "$VLLM_PORT" "$@" &
VLLM_PID=$!
echo "[entry] vLLM pid=${VLLM_PID}; waiting for /health"
# GLM-5.1-FP8 load can take ~30-60 min in CC-on; allow 90.
wait_for_url "http://${VLLM_HOST}:${VLLM_PORT}/health" "vLLM" "$VLLM_PID" 5400
echo "[entry] vLLM ready"

# ===== 2. egress_service ===================================================
echo "[entry] starting egress_service on ${EGRESS_HOST_INTERNAL}:${EGRESS_PORT_INTERNAL}"
EGRESS_HOST="$EGRESS_HOST_INTERNAL" \
EGRESS_PORT="$EGRESS_PORT_INTERNAL" \
VLLM_LOOPBACK_URL="http://${VLLM_HOST}:${VLLM_PORT}" \
python3 -u /workspace/egress_service.py &
EGRESS_PID=$!
echo "[entry] egress_service pid=${EGRESS_PID}; waiting for /health"
wait_for_url "http://${EGRESS_HOST_INTERNAL}:${EGRESS_PORT_INTERNAL}/health" "egress" "$EGRESS_PID" 300
echo "[entry] egress_service ready"

# ===== 3. PySyft Datasite ==================================================
echo "[entry] starting PySyft Datasite on 0.0.0.0:${SYFT_PORT}"
python3 -u /workspace/pysyft_datasite_server.py &
SYFT_PID=$!
echo "[entry] PySyft pid=${SYFT_PID}; waiting for /api/v2/metadata"
wait_for_url "http://127.0.0.1:${SYFT_PORT}/api/v2/metadata" "PySyft" "$SYFT_PID" 600
echo "[entry] PySyft Datasite ready; all 3 processes up"

# ===== supervisor loop =====================================================
# wait -n returns when any child exits. Whichever dies first → kill the others
# and propagate exit code. Container goes to Failed in Tinfoil, prompting
# either a relaunch or (more usefully) a diag-mode rebuild.
echo "[entry] supervising; pid_vllm=${VLLM_PID} pid_egress=${EGRESS_PID} pid_syft=${SYFT_PID}"
set +e
wait -n
EXIT=$?
echo "[entry] a child exited (rc=${EXIT}); shutting down siblings"
shutdown TERM
exit "$EXIT"
