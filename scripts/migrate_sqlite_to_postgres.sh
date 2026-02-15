#!/usr/bin/env bash
set -euo pipefail

# Helper script for SQLite -> PostgreSQL migration:
# - optional data transfer via pgloader
# - schema alignment via alembic
#
# Usage:
#   scripts/migrate_sqlite_to_postgres.sh --sqlite ./catalogue.db --pg "$DATABASE_URL" --run
#   scripts/migrate_sqlite_to_postgres.sh --sqlite ./catalogue.db --pg "$DATABASE_URL" --dry-run

SQLITE_PATH="catalogue.db"
PG_URL="${DATABASE_URL:-}"
MODE="dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sqlite)
      SQLITE_PATH="${2:-}"
      shift 2
      ;;
    --pg)
      PG_URL="${2:-}"
      shift 2
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --dry-run)
      MODE="dry-run"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PG_URL" ]]; then
  echo "PostgreSQL URL is required. Pass --pg or set DATABASE_URL." >&2
  exit 1
fi

if [[ ! -f "$SQLITE_PATH" ]]; then
  echo "SQLite file not found: $SQLITE_PATH" >&2
  exit 1
fi

echo "SQLite source: $SQLITE_PATH"
echo "PostgreSQL target: $PG_URL"
echo "Mode: $MODE"
echo

echo "Step 1/4: backup SQLite"
BACKUP_DIR="backups"
mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="$BACKUP_DIR/catalogue_before_pg_${STAMP}.db"
if [[ "$MODE" == "run" ]]; then
  cp "$SQLITE_PATH" "$BACKUP_PATH"
fi
echo "  backup: $BACKUP_PATH"

echo "Step 2/4: connectivity check"
if [[ "$MODE" == "run" ]]; then
  psql "$PG_URL" -c "select 1;" >/dev/null
fi
echo "  ok"

echo "Step 3/4: data transfer (pgloader)"
if command -v pgloader >/dev/null 2>&1; then
  echo "  pgloader found"
  if [[ "$MODE" == "run" ]]; then
    pgloader "sqlite:///${SQLITE_PATH}" "$PG_URL"
  fi
else
  echo "  pgloader is not installed; skipping data transfer"
  echo "  install pgloader and rerun with --run"
fi

echo "Step 4/4: alembic upgrade head"
if [[ "$MODE" == "run" ]]; then
  DATABASE_URL="$PG_URL" alembic upgrade head
fi
echo "  done"

echo
echo "Migration helper finished."
echo "Next: run smoke checks from docs/postgres_migration.md"
