from __future__ import annotations

import os
import re

# A reverse proxy (Personal on the Pi) may strip a fixed prefix such as `/lang` before forwarding.
# Routes stay unprefixed; only URLs sent back to the browser need the public prefix.
_BASE_PATH_PATTERN = re.compile(r"(/[A-Za-z0-9._~-]+)*")


def parse_base_path(value: str) -> str:
    if not _BASE_PATH_PATTERN.fullmatch(value):
        raise ValueError("ARC_LANG_BASE_PATH must be empty or like /lang, without a trailing slash")
    return value


BASE_PATH = parse_base_path(os.getenv("ARC_LANG_BASE_PATH", ""))


def app_path(path: str) -> str:
    if not path.startswith("/"):
        raise ValueError("application paths must be absolute")
    return f"{BASE_PATH}{path}"
