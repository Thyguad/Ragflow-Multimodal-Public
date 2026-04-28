#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DOCKER_ROOT="$REPO_ROOT/docker"

if [[ -f "$REPO_ROOT/load_dev_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/load_dev_env.sh"
fi

ensure_docker_ready() {
  if docker info >/dev/null 2>&1; then
    return
  fi

  if [[ -d /Applications/Docker.app ]]; then
    echo "Docker Desktop is not ready. Launching Docker.app..."
    open -a Docker
  fi

  for _ in $(seq 1 60); do
    sleep 2
    if docker info >/dev/null 2>&1; then
      return
    fi
  done

  echo "Docker daemon is not ready. Please start Docker Desktop and retry." >&2
  exit 1
}

echo "Starting RAGFlow base services (MySQL / Elasticsearch / MinIO / Redis)..."

ensure_docker_ready

cd "$DOCKER_ROOT"
docker compose -f docker-compose-base.yml up -d

echo
echo "Services started:"
docker compose -f docker-compose-base.yml ps
echo
echo "Ports:"
echo "  MySQL:         localhost:5455"
echo "  Elasticsearch: localhost:1200"
echo "  MinIO:         localhost:9000"
echo "  Redis:         localhost:6380"
