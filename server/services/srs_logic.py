from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any


@dataclass
class SRSParams:
    target_retention: float = 0.9
    fail_factor: float = 0.5
    g_pass_weak: float = 1.0
    g_pass_norm: float = 2.0
    g_pass_strong: float = 3.0


def _days(dt: float) -> float:
    return max(0.0, float(dt) / 86400.0)


def retention_now(lexeme: Any, now: datetime) -> float:
    S = float(getattr(lexeme, "stability", 1.0) or 1.0)
    ref = getattr(lexeme, "last_seen_at", None) or getattr(lexeme, "first_seen_at", None) or getattr(lexeme, "created_at", None)
    if not ref:
        return 0.5
    try:
        dt = (now - ref).total_seconds()
        return math.exp(-_days(dt) / max(0.1, S))
    except Exception:
        return 0.5


def decay_posterior(lexeme: Any, now: datetime, hl_days: float) -> None:
    if hl_days <= 0:
        return
    last = getattr(lexeme, "last_decay_at", None) or getattr(lexeme, "created_at", None) or now
    try:
        dt = (now - last).total_seconds()
    except Exception:
        dt = 0.0
    k = math.exp(-math.log(2.0) * (_days(dt) / float(hl_days)))
    try:
        lexeme.alpha = float(getattr(lexeme, "alpha", 1.0) or 1.0) * k
        lexeme.beta = float(getattr(lexeme, "beta", 9.0) or 9.0) * k
    except Exception:
        pass
    try:
        lexeme.last_decay_at = now
    except Exception:
        pass


def schedule_next(lexeme: Any, quality: int, now: datetime, params: SRSParams) -> None:
    # Very small, safe scheduler: bump stability based on quality and posterior mean
    a = float(getattr(lexeme, "alpha", 1.0) or 1.0)
    b = float(getattr(lexeme, "beta", 9.0) or 9.0)
    mu = a / (a + b) if (a + b) > 0 else 0.0
    S = float(getattr(lexeme, "stability", 0.5) or 0.5)
    g = params.g_pass_norm
    if quality <= 0:
        g = -params.fail_factor
    elif quality == 1:
        g = params.g_pass_weak
    elif quality >= 3:
        g = params.g_pass_strong
    S2 = max(0.1, S + 0.3 * g * (0.2 + mu))
    try:
        lexeme.stability = float(S2)
    except Exception:
        pass
    try:
        lexeme.next_due_at = now + timedelta(days=S2)
    except Exception:
        pass