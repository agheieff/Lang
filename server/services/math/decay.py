from __future__ import annotations

import math


def decay_factor(dt_days: float, hl_days: float) -> float:
    if hl_days <= 0:
        return 1.0
    try:
        return math.exp(-math.log(2.0) * (dt_days / hl_days))
    except Exception:
        return 1.0

