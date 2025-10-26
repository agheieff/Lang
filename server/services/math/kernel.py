from __future__ import annotations

import math
from typing import List


def gaussian_kernel_weights(center: float, sigma: float, bins: int = 6) -> List[float]:
    if sigma <= 0:
        sigma = 1.0
    out: List[float] = []
    for i in range(1, bins + 1):
        d = (i - center)
        out.append(math.exp(-0.5 * (d / sigma) ** 2))
    s = sum(out) or 1.0
    return [v / s for v in out]

