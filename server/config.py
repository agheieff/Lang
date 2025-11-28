from __future__ import annotations

import json
import os
from typing import Dict


def _b(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v2 = v.strip().lower()
    return v2 in ("1", "true", "yes", "on", "y")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


# Feature flags
MSP_ENABLE: bool = _b("ARC_LANG_MSP_ENABLE", False)


# SRS weights and half-lives (days)
_W_CLICK: float = _f("ARC_SRS_W_CLICK", 0.2)
_W_NONLOOK: float = _f("ARC_SRS_W_NONLOOK", 1.0)
_W_EXPOSURE: float = _f("ARC_SRS_W_EXPOSURE", 0.02)

_HL_CLICK_D: float = _f("ARC_SRS_HL_CLICK_D", 14.0)
_HL_NONLOOK_D: float = _f("ARC_SRS_HL_NONLOOK_D", 30.0)
_HL_EXPOSURE_D: float = _f("ARC_SRS_HL_EXPOSURE_D", 7.0)

# FSRS-like schedule parameters
_FSRS_TARGET_R: float = _f("ARC_SRS_TARGET_R", 0.9)
_FSRS_FAIL_F: float = _f("ARC_SRS_FAIL_F", 0.5)
_G_PASS_WEAK: float = _f("ARC_SRS_G_PASS_WEAK", 1.0)
_G_PASS_NORM: float = _f("ARC_SRS_G_PASS_NORM", 2.0)
_G_PASS_STRONG: float = _f("ARC_SRS_G_PASS_STRONG", 3.0)

# Exposure gating
_SESSION_MIN: float = _f("ARC_SRS_SESSION_MIN_MINUTES", 1.0)
_EXPOSURE_WEAK_W: float = _f("ARC_SRS_EXPOSURE_WEAK_W", 0.01)
_DISTINCT_PROMOTE: int = _i("ARC_SRS_DISTINCT_PROMOTE", 3)
_FREQ_LOW_THRESH: int = _i("ARC_SRS_FREQ_LOW_THRESH", 10000)
_DIFF_HIGH: float = _f("ARC_SRS_DIFF_HIGH", 1.5)

# Synthetic nonlookup promotions
_SYN_NL_ENABLE: bool = _b("ARC_SRS_SYN_NL_ENABLE", False)
_SYN_NL_MIN_DISTINCT: int = _i("ARC_SRS_SYN_NL_MIN_DISTINCT", 3)
_SYN_NL_MIN_DAYS: int = _i("ARC_SRS_SYN_NL_MIN_DAYS", 3)
_SYN_NL_COOLDOWN_DAYS: int = _i("ARC_SRS_SYN_NL_COOLDOWN_DAYS", 7)


# Rate limiting (per window per key)
RATE_WINDOW_SEC: int = _i("ARC_RATE_WINDOW_SEC", 60)
_DEFAULT_RATE_LIMITS: Dict[str, int] = {
    # New product tiers
    "Free": _i("ARC_RATE_FREE", 60),
    "Standard": _i("ARC_RATE_STANDARD", 300),
    "Pro": _i("ARC_RATE_PRO", 600),
    "Pro+": _i("ARC_RATE_PRO_PLUS", 1200),
    "BYOK": _i("ARC_RATE_BYOK", 100000),
    # Ops/internal
    "admin": _i("ARC_RATE_ADMIN", 1000000),  # effectively unlimited; middleware bypasses anyway
    # Back-compat keys (lowercase legacy)
    "free": _i("ARC_RATE_FREE", 60),
    "pro": _i("ARC_RATE_PRO", 600),
    "premium": _i("ARC_RATE_STANDARD", 300),
}
try:
    _rl_env = os.getenv("ARC_RATE_LIMITS_JSON")
    RATE_LIMITS: Dict[str, int] = json.loads(_rl_env) if _rl_env else _DEFAULT_RATE_LIMITS
    if not isinstance(RATE_LIMITS, dict):
        RATE_LIMITS = _DEFAULT_RATE_LIMITS
except Exception:
    RATE_LIMITS = _DEFAULT_RATE_LIMITS


# OpenRouter per-user key management
OPENROUTER_PROVISIONING_KEY: str = os.getenv("OPENROUTER_PROVISIONING_KEY", "")
OPENROUTER_KEY_ENCRYPTION_SECRET: str = os.getenv("OPENROUTER_KEY_ENCRYPTION_SECRET", "")

# Per-tier monthly spending limits (USD) for OpenRouter sub-keys
# None means no limit (BYOK users bring their own key)
TIER_SPENDING_LIMITS: Dict[str, float | None] = {
    "Free": None,  # Free tier uses shared pool, no individual key
    "Standard": _f("ARC_TIER_LIMIT_STANDARD", 25.0),
    "Pro": _f("ARC_TIER_LIMIT_PRO", 100.0),
    "Pro+": _f("ARC_TIER_LIMIT_PRO_PLUS", 250.0),
    "BYOK": None,  # User provides their own key
    "admin": None,  # No limit for admins
}

# Tiers that get their own OpenRouter sub-key
PAID_TIERS: set[str] = {"Standard", "Pro", "Pro+"}


# Text pool settings
POOL_SIZE: int = _i("ARC_POOL_SIZE", 4)  # Number of pre-generated texts to maintain per user/lang
POOL_CI_VARIANCE: float = _f("ARC_POOL_CI_VARIANCE", 0.05)  # How much to vary CI target in pool

# Topic categories for text generation (path-based strings, hierarchy-ready)
TOPICS: list[str] = [
    "fiction",
    "news", 
    "science",
    "technology",
    "history",
    "daily_life",
    "culture",
    "sports",
    "business",
]

# Default topic weights (all equal)
DEFAULT_TOPIC_WEIGHTS: dict[str, float] = {t: 1.0 for t in TOPICS}


# Free tier usage limits (monthly)
FREE_TIER_CHAR_LIMIT: int = _i("ARC_FREE_CHAR_LIMIT", 100_000)  # ~25-30 texts worth
FREE_TIER_TEXT_LIMIT: int = _i("ARC_FREE_TEXT_LIMIT", 50)  # Hard cap on text count
