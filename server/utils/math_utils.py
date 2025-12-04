from __future__ import annotations

import math
from typing import List, Tuple


def decay_factor(dt_days: float, hl_days: float) -> float:
    """Calculate exponential decay factor based on half-life."""
    if hl_days <= 0:
        return 1.0
    try:
        return math.exp(-math.log(2.0) * (dt_days / hl_days))
    except Exception:
        return 1.0


def gaussian_kernel_weights(center: float, sigma: float, bins: int = 6) -> List[float]:
    """Generate Gaussian kernel weights for level estimation."""
    if sigma <= 0:
        sigma = 1.0
    out: List[float] = []
    for i in range(1, bins + 1):
        d = (i - center)
        out.append(math.exp(-0.5 * (d / sigma) ** 2))
    s = sum(out) or 1.0
    return [v / s for v in out]


def estimate_level(mu: List[float], var: List[float], p_target: float) -> Tuple[float, float]:
    """Estimate user level from probability distributions."""
    jstar = 0
    for j in range(6):
        if mu[j] >= p_target:
            jstar = j + 1
    if jstar == 0:
        frac = max(0.0, min(1.0, mu[0] / max(1e-6, p_target)))
        lvl = frac
        v = var[0] + 1.0
        return (lvl, v)
    if jstar >= 6:
        return (6.0, var[5] + 1.0 / (1.0 + sum(mu)))
    j1 = jstar - 1
    j2 = j1 + 1
    denom = abs(mu[j1] - mu[j2]) or 1e-6
    frac = max(0.0, min(1.0, (mu[j1] - p_target) / denom))
    lvl = jstar + frac
    sigma_p = (max(0.0, var[j1]) + max(0.0, var[j2])) ** 0.5
    slope = denom
    level_var = (sigma_p / max(1e-6, slope)) ** 2
    return (max(0.0, min(6.0, lvl)), max(0.0, level_var))


def compute_level_from_counts(a: List[float], b: List[float], prior_a: float, prior_b: float, p_target: float) -> Tuple[float, float]:
    """Compute level from success/failure counts using Bayesian estimation."""
    mu: List[float] = []
    vr: List[float] = []
    for j in range(6):
        alpha = prior_a + max(0.0, a[j])
        beta = prior_b + max(0.0, b[j])
        s = alpha + beta
        mu.append(alpha / s)
        vr.append((alpha * beta) / (s * s * (s + 1.0)))
    return estimate_level(mu, vr, p_target)
