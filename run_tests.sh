#!/bin/bash
export ARC_LANG_ENVIRONMENT=test
export ARC_LANG_JWT_SECRET=test-secret
uv run pytest "$@"
