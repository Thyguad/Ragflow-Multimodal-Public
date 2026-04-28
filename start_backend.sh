#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/logs"
SERVER_LOG="$LOG_DIR/ragflow_server.log"
IMAGE_LOG="$LOG_DIR/image_server.log"
TASK_LOG="$LOG_DIR/task_executor_0.log"

mkdir -p "$LOG_DIR"

clear_proxy_env() {
  # Keep Codex's proxy configuration from leaking into project services.
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping stale process on port ${port}: ${pids}"
    kill -9 $pids 2>/dev/null || true
  fi
}

stop_task_executor() {
  local pids
  pids="$(pgrep -f "rag/svr/task_executor.py" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping stale task_executor.py process: ${pids}"
    kill -9 $pids 2>/dev/null || true
  fi
}

is_running() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

wait_for_port() {
  local port="$1"
  local name="$2"
  local pid="$3"
  local log_path="$4"
  local timeout="${5:-180}"

  for ((i = 1; i <= timeout; i++)); do
    if lsof -ti "tcp:${port}" >/dev/null 2>&1; then
      echo "${name} is ready on port ${port}."
      return 0
    fi

    if ! is_running "$pid"; then
      echo "${name} exited before port ${port} became ready. Check log: ${log_path}" >&2
      tail -n 80 "$log_path" >&2 || true
      exit 1
    fi

    sleep 1
  done

  echo "${name} did not become ready on port ${port} within ${timeout}s. Check log: ${log_path}" >&2
  tail -n 80 "$log_path" >&2 || true
  exit 1
}

wait_for_process() {
  local pid="$1"
  local name="$2"
  local log_path="$3"
  local seconds="${4:-5}"

  sleep "$seconds"
  if is_running "$pid"; then
    echo "${name} is running with PID ${pid}."
    return 0
  fi

  echo "${name} exited during startup. Check log: ${log_path}" >&2
  tail -n 80 "$log_path" >&2 || true
  exit 1
}

cd "$REPO_ROOT"

echo "Preparing backend services..."
stop_port 9380
stop_port 8000
stop_task_executor

if [[ -f "$REPO_ROOT/load_dev_env.sh" ]]; then
  # Load bash-compatible local toolchain environment without sourcing zsh-only config.
  # shellcheck disable=SC1091
  source "$REPO_ROOT/load_dev_env.sh"
fi

clear_proxy_env

source .venv/bin/activate

export PYTHONPATH="$REPO_ROOT"
export NLTK_DATA="$REPO_ROOT/nltk_data"
export HF_ENDPOINT="https://hf-mirror.com"
export PY="$REPO_ROOT/.venv/bin/python"
if [[ -z "${RAGFLOW_VLM_API_KEY:-}" ]]; then
  echo "RAGFLOW_VLM_API_KEY is required for VLM parsing. Export it before running start_backend.sh." >&2
  exit 1
fi
export RAGFLOW_VLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RAGFLOW_VLM_MODEL="qwen2.5-vl-72b-instruct"
export RAGFLOW_MD_API_KEY="${RAGFLOW_MD_API_KEY:-$RAGFLOW_VLM_API_KEY}"
export RAGFLOW_MD_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export RAGFLOW_MD_MODEL="qwen-plus"
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-$RAGFLOW_VLM_API_KEY}"
export RAGFLOW_IMAGE_EMBEDDING_API_KEY="${RAGFLOW_IMAGE_EMBEDDING_API_KEY:-$DASHSCOPE_API_KEY}"
export RAGFLOW_IMAGE_EMBEDDING_MODEL="${RAGFLOW_IMAGE_EMBEDDING_MODEL:-qwen3-vl-embedding}"
export RAGFLOW_IMAGE_EMBEDDING_DIMENSION="${RAGFLOW_IMAGE_EMBEDDING_DIMENSION:-1024}"
export SERVER_IP="${SERVER_IP:-http://127.0.0.1:8000}"

: >"$SERVER_LOG"
: >"$IMAGE_LOG"
: >"$TASK_LOG"

echo "Starting ragflow_server.py..."
nohup "$PY" api/ragflow_server.py >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Starting image_server.py..."
nohup "$PY" image_server.py >"$IMAGE_LOG" 2>&1 &
IMAGE_PID=$!

echo "Starting task_executor.py..."
nohup "$PY" rag/svr/task_executor.py 0 >"$TASK_LOG" 2>&1 &
TASK_PID=$!

echo "Waiting for backend readiness..."
wait_for_port 9380 "RAGFlow backend" "$SERVER_PID" "$SERVER_LOG" 240
wait_for_port 8000 "Image server" "$IMAGE_PID" "$IMAGE_LOG" 60
wait_for_process "$TASK_PID" "Task executor" "$TASK_LOG" 5

cat <<EOF

Backend is ready.
- API server: http://127.0.0.1:9380
- Image server: http://127.0.0.1:8000
- Task executor PID: ${TASK_PID}

Now it is safe to start the frontend:
  bash ./start_frontend.sh

Logs:
- ${SERVER_LOG}
- ${IMAGE_LOG}
- ${TASK_LOG}
EOF
