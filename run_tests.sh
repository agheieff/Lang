#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
uv run ruff check server tests
uv run ruff format --check server tests
uv run mypy server
uv run pytest
pnpm check
pnpm test
