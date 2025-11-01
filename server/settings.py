from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class _Settings:
    NEXT_READY_MAX_WAIT_SEC: float = float(os.getenv("ARC_NEXT_READY_MAX_WAIT_SEC", "8"))
    NEXT_READY_GRACE_SEC: float = float(os.getenv("ARC_NEXT_READY_GRACE_SEC", "3"))


_SETTINGS = _Settings()


def get_settings() -> _Settings:
    return _SETTINGS
