#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
RELEASE_DIR="$DIST_DIR/release"

command -v git >/dev/null 2>&1 || { echo "git not found"; exit 1; }

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(git describe --tags --always --dirty || date +%Y%m%d%H%M%S)"
fi

echo "==> Preparing release $VERSION"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

echo "==> Building frontend bundles"
(
  cd "$ROOT_DIR/frontend"
  npm install >/dev/null
  npm run build
)
(
  cd "$ROOT_DIR/AiWord"
  npm install >/dev/null
  npm run build
)

echo "==> Collecting sources"
tarball="$DIST_DIR/agregator-$VERSION.tar.gz"
tmpdir="$RELEASE_DIR/agregator-$VERSION"
mkdir -p "$tmpdir"

rsync -a \
  --exclude '.git' \
  --exclude 'dist' \
  --exclude 'library' \
  --exclude 'logs' \
  --exclude 'backups' \
  "$ROOT_DIR/" "$tmpdir/"

echo "$VERSION" > "$tmpdir/VERSION"

echo "==> Creating archive $tarball"
mkdir -p "$DIST_DIR"
tar -C "$RELEASE_DIR" -czf "$tarball" "agregator-$VERSION"

cat <<EOF
Release artifact created:
  $tarball

You can upload this file to the target server and run the Ansible playbook:
  ansible-playbook -i deploy/ansible/inventory sample deploy/ansible/site.yml \
    -e agregator_release_tarball=/path/to/agregator-$VERSION.tar.gz \
    -e agregator_release_id=$VERSION
EOF
