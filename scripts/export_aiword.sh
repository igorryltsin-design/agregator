#!/usr/bin/env bash
set -euo pipefail

# Creates a clean standalone copy of the AiWord frontend that can be moved
# to another machine. By default the copy is placed under backups/ with a
# timestamped directory name. Pass a custom destination path as the first
# argument. Add --zip to also produce a .tar.gz archive next to the copy.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/AiWord"

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "[export-aiword] Source directory ${SOURCE_DIR} not found."
  exit 1
fi

usage() {
  cat <<'EOF'
Usage: scripts/export_aiword.sh [destination] [--zip]

destination  Optional target directory for the standalone copy.
             Defaults to backups/aiword-standalone-<timestamp> inside the repo.
--zip        Additionally create <destination>.tar.gz next to the copy.
EOF
}

ZIP_ARCHIVE=false
DEST_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)
      ZIP_ARCHIVE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [ -n "$DEST_PATH" ]; then
        echo "[export-aiword] Unexpected argument: $1"
        usage
        exit 1
      fi
      DEST_PATH="$1"
      shift
      ;;
  esac
done

timestamp="$(date +%Y%m%d-%H%M%S)"
if [ -z "$DEST_PATH" ]; then
  DEST_PATH="${ROOT_DIR}/backups/aiword-standalone-${timestamp}"
else
  DEST_PATH="$(python3 - <<'PY' "$DEST_PATH"
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[export-aiword] rsync is required but not found. Install it and retry."
  exit 1
fi

echo "[export-aiword] Creating standalone copy at ${DEST_PATH}"
mkdir -p "${DEST_PATH}"

RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.idea"
  "--exclude=.vscode"
  "--exclude=node_modules"
  "--exclude=dist/node"
  "--exclude=.DS_Store"
  "--exclude=*.log"
)

rsync -a "${RSYNC_EXCLUDES[@]}" "${SOURCE_DIR}/" "${DEST_PATH}/"

if [ "${ZIP_ARCHIVE}" = true ]; then
  ARCHIVE_PATH="${DEST_PATH}.tar.gz"
  echo "[export-aiword] Creating archive ${ARCHIVE_PATH}"
  tar -C "$(dirname "${DEST_PATH}")" -czf "${ARCHIVE_PATH}" "$(basename "${DEST_PATH}")"
fi

cat <<'EOF'
[export-aiword] Done.
Copy includes package.json, src/, configuration and other assets.
Run npm install && npm run dev (or npm run build) inside the copied folder on the target machine.
EOF
