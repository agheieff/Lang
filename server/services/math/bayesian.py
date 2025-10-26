from __future__ import annotations

from typing import List, Tuple


def estimate_level(mu: List[float], var: List[float], p_target: float) -> Tuple[float, float]:
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
    mu: List[float] = []
    vr: List[float] = []
    for j in range(6):
        alpha = prior_a + max(0.0, a[j])
        beta = prior_b + max(0.0, b[j])
        s = alpha + beta
        mu.append(alpha / s)
        vr.append((alpha * beta) / (s * s * (s + 1.0)))
    return estimate_level(mu, vr, p_target)

