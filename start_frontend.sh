#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WEB_ROOT="$REPO_ROOT/web"

clear_proxy_env() {
  # Keep Codex's proxy configuration from leaking into project commands.
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY
  unset http_proxy https_proxy all_proxy no_proxy
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti "tcp:${port}" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping stale frontend process on port ${port}: ${pids}"
    kill -9 $pids 2>/dev/null || true
  fi
}

require_backend_ready() {
  local missing=0

  if ! lsof -ti "tcp:9380" >/dev/null 2>&1; then
    echo "Backend API is not ready on port 9380." >&2
    missing=1
  fi

  if ! lsof -ti "tcp:8000" >/dev/null 2>&1; then
    echo "Image server is not ready on port 8000." >&2
    missing=1
  fi

  if ! pgrep -f "rag/svr/task_executor.py" >/dev/null 2>&1; then
    echo "Task executor is not running." >&2
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    cat >&2 <<'EOF'

Please start the backend first and wait for the success message:
  bash ./start_backend.sh

Start the frontend only after you see:
  Backend is ready.
EOF
    exit 1
  fi

  echo "Backend is ready; starting frontend setup..."
}

cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/load_dev_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/load_dev_env.sh"
fi

clear_proxy_env

if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found in PATH." >&2
  exit 1
fi

require_backend_ready

stop_port 9222

rm -rf "$WEB_ROOT/src/.umi" \
       "$WEB_ROOT/node_modules/.cache/mfsu" \
       "$WEB_ROOT/node_modules/.cache/.mfsu"

cd "$WEB_ROOT"
npm run setup

if [[ ! -f "$WEB_ROOT/src/.umi/umi.ts" || ! -f "$WEB_ROOT/src/.umi/exports.ts" ]]; then
  echo "umi setup did not generate src/.umi correctly." >&2
  exit 1
fi

echo "Generating Tailwind CSS..."
mkdir -p "$WEB_ROOT/src/.umi/plugin-tailwindcss"
node "$WEB_ROOT/node_modules/tailwindcss/lib/cli.js" \
  -c "$WEB_ROOT/tailwind.config.js" \
  -i "$WEB_ROOT/tailwind.css" \
  -o "$WEB_ROOT/src/.umi/plugin-tailwindcss/tailwind.css"

echo "Starting frontend dev server at http://localhost:9222"
echo "Keep this terminal open while you develop."

exec npm run dev -- --host 127.0.0.1 --port 9222
