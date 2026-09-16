#!/usr/bin/env bash
# Server-side deployment for the Docker Compose stack (run on the EC2 instance by the
# GitHub Actions "deploy" job, or by hand).
#
#   ./deploy.sh deploy <image-tag>   pull <tag> from ECR, migrate, restart, health-check; rolls back on failure
#   ./deploy.sh rollback             redeploy the previously running tag (manual rollback)
#   ./deploy.sh status               show the current and previous tags and container health
#
# Expects, in the same directory: docker-compose.yml and a filled-in .env (see .env.example).
# Environment: ECR_REGISTRY (account.dkr.ecr.region.amazonaws.com), AWS_REGION and, unless the
# instance has an IAM role allowed to pull from ECR, AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
# Optional: ECR_BACKEND_REPO (voice-backend), ECR_FRONTEND_REPO (voice-frontend),
# HEALTH_URL (http://localhost:8000/health), HEALTH_RETRIES (5), HEALTH_INTERVAL (15 s).

set -euo pipefail

cd "$(dirname "$0")"

COMPOSE=(docker compose -f docker-compose.yml) # base file only: no source mounts, no --reload
STATE_DIR=".deploy"
CURRENT_FILE="$STATE_DIR/current_tag"
PREVIOUS_FILE="$STATE_DIR/previous_tag"
ECR_BACKEND_REPO="${ECR_BACKEND_REPO:-voice-backend}"
ECR_FRONTEND_REPO="${ECR_FRONTEND_REPO:-voice-frontend}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_RETRIES="${HEALTH_RETRIES:-5}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-15}"

log() { printf '%s | %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

require_env() {
  local missing=0
  for name in "$@"; do
    if [ -z "${!name:-}" ]; then
      log "ERROR: $name is not set"
      missing=1
    fi
  done
  [ "$missing" -eq 0 ]
}

ecr_login() {
  require_env ECR_REGISTRY AWS_REGION
  aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
}

# Point docker-compose.yml at the given tag by rewriting BACKEND_IMAGE / FRONTEND_IMAGE in .env.
set_image_tags() {
  local tag="$1"
  local backend="$ECR_REGISTRY/$ECR_BACKEND_REPO:$tag"
  local frontend="$ECR_REGISTRY/$ECR_FRONTEND_REPO:$tag"
  touch .env
  for pair in "BACKEND_IMAGE=$backend" "FRONTEND_IMAGE=$frontend"; do
    local key="${pair%%=*}"
    if grep -q "^${key}=" .env; then
      sed -i "s#^${key}=.*#${pair}#" .env
    else
      printf '%s\n' "$pair" >> .env
    fi
  done
  log "docker-compose.yml now uses $backend and $frontend"
}

pull_images() {
  local tag="$1"
  docker pull "$ECR_REGISTRY/$ECR_BACKEND_REPO:$tag"
  docker pull "$ECR_REGISTRY/$ECR_FRONTEND_REPO:$tag"
}

run_migrations() {
  log "Running database migrations"
  "${COMPOSE[@]}" up -d postgres redis
  "${COMPOSE[@]}" run --rm backend alembic upgrade head
}

restart_stack() {
  log "Restarting the stack"
  "${COMPOSE[@]}" down
  "${COMPOSE[@]}" up -d
}

health_check() {
  local attempt
  for attempt in $(seq 1 "$HEALTH_RETRIES"); do
    if curl -fsS --max-time 10 "$HEALTH_URL" | grep -q '"status":"ok"'; then
      log "Health check passed (attempt $attempt/$HEALTH_RETRIES)"
      return 0
    fi
    log "Health check not ready (attempt $attempt/$HEALTH_RETRIES); waiting ${HEALTH_INTERVAL}s"
    sleep "$HEALTH_INTERVAL"
  done
  log "ERROR: health check failed after $HEALTH_RETRIES attempts"
  "${COMPOSE[@]}" ps || true
  "${COMPOSE[@]}" logs --tail=100 backend || true
  return 1
}

# Old images are kept locally (never pruned here) so a rollback needs no registry access.
activate() {
  local tag="$1"
  set_image_tags "$tag"
  run_migrations
  restart_stack
  health_check
}

deploy() {
  local tag="${1:?usage: deploy.sh deploy <image-tag>}"
  mkdir -p "$STATE_DIR"
  local previous=""
  [ -f "$CURRENT_FILE" ] && previous="$(cat "$CURRENT_FILE")"

  ecr_login
  log "Pulling images for tag $tag"
  pull_images "$tag"

  if activate "$tag"; then
    [ -n "$previous" ] && printf '%s\n' "$previous" > "$PREVIOUS_FILE"
    printf '%s\n' "$tag" > "$CURRENT_FILE"
    log "Deployed tag $tag (previous: ${previous:-none})"
    return 0
  fi

  log "Deployment of $tag failed"
  if [ -n "$previous" ]; then
    log "Rolling back to $previous"
    if activate "$previous"; then
      log "Rollback to $previous succeeded"
    else
      log "ERROR: rollback to $previous also failed; manual intervention required"
    fi
  else
    log "No previous tag recorded; nothing to roll back to"
  fi
  return 1
}

rollback() {
  [ -f "$PREVIOUS_FILE" ] || { log "ERROR: no previous tag recorded in $PREVIOUS_FILE"; exit 1; }
  local previous current
  previous="$(cat "$PREVIOUS_FILE")"
  current="$(cat "$CURRENT_FILE" 2>/dev/null || true)"
  log "Manual rollback from ${current:-unknown} to $previous"
  activate "$previous"
  printf '%s\n' "$previous" > "$CURRENT_FILE"
  [ -n "$current" ] && printf '%s\n' "$current" > "$PREVIOUS_FILE"
  log "Rollback complete"
}

status() {
  log "current tag:  $(cat "$CURRENT_FILE" 2>/dev/null || echo none)"
  log "previous tag: $(cat "$PREVIOUS_FILE" 2>/dev/null || echo none)"
  "${COMPOSE[@]}" ps
}

case "${1:-}" in
  deploy) deploy "${2:-}" ;;
  rollback) rollback ;;
  status) status ;;
  *)
    echo "usage: $0 deploy <image-tag> | rollback | status" >&2
    exit 2
    ;;
esac
