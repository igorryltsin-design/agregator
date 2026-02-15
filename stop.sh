#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"

log() {
  echo "[stop] $*" >&2
}

DEFAULT_PORT="${AGREGATOR_PORT:-5050}"

stop_by_app_path() {
  local app_path="$1"
  local pids
  pids="$(pgrep -f "$app_path" || true)"
  if [ -n "$pids" ]; then
    log "Stopping app processes: $pids"
    if ! kill $pids 2>/dev/null; then
      log "kill failed, trying sudo (may ask for password)"
      sudo kill $pids 2>/dev/null || true
    fi
  fi
}

stop_by_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    log "Stopping processes on port $port: $pids"
    if ! kill $pids 2>/dev/null; then
      log "kill failed, trying sudo (may ask for password)"
      sudo kill $pids 2>/dev/null || true
    fi
  fi
}

stop_by_app_path "$here/app.py"
stop_by_port "$DEFAULT_PORT"

log "Done."
