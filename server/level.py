from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .models import Profile
from .services.level_service import (
    update_level_for_profile,
    update_level_if_stale,
    get_level_summary,
    HL_DAYS,
)
from .utils.math_utils import gaussian_kernel_weights as _gaussian_kernel_weights, decay_factor as _decay_factor_impl

# Re-export types for callers
__all__ = [
    "update_level_for_profile",
    "update_level_if_stale",
    "get_level_summary",
    "_gaussian_kernel_weights",
    "HL_DAYS",
]


def _decay_factor(dt_days: float) -> float:
    return _decay_factor_impl(dt_days, HL_DAYS)
