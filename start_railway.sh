#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DB_PATH="${FARM_DB_PATH:-data/farm.db}"

mkdir -p "$(dirname "$DB_PATH")"

exec python app.py --host "$HOST" --port "$PORT" --db "$DB_PATH"
