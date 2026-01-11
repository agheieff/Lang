#!/usr/bin/env bash
set -euo pipefail

# Default port
PORT=8000

usage() {
  echo "Usage: $0 [--port N | -p N]" >&2
}

# Parse args
while [ $# -gt 0 ]; do
  case "$1" in
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --port|-p)
      if [ $# -lt 2 ]; then usage; exit 1; fi
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage; exit 0
      ;;
    *)
      usage; exit 1
      ;;
  esac
done

# Run from project root (script directory)
cd "$(dirname "${BASH_SOURCE[0]}")"

# Ensure deps/env are synced before running
uv sync

# Set default environment variables if not set
export ARC_STARTUP_LANGS="${ARC_STARTUP_LANGS:-zh-CN}"
export ARC_STARTUP_TEXTS_PER_LANG="${ARC_STARTUP_TEXTS_PER_LANG:-5}"

exec uv run uvicorn server.main:app --reload --port "$PORT"
