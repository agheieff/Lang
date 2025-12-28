from __future__ import annotations

"""
Common utility functions merged from math_utils.py, file_lock.py, and other misc utilities.
"""

import math
import json
import fcntl
import os
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)


# Math utilities
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
        d = i - center
        out.append(math.exp(-0.5 * (d / sigma) ** 2))
    s = sum(out) or 1.0
    return [v / s for v in out]


def estimate_level(
    mu: List[float], var: List[float], p_target: float
) -> Tuple[float, float]:
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
    v = (sigma_p / max(1e-6, slope)) ** 2
    return (max(0.0, min(6.0, lvl)), max(0.0, v))


def compute_level_from_counts(
    a: List[float], b: List[float], prior_a: float, prior_b: float, p_target: float
) -> Tuple[float, float]:
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


# File lock utilities
class FileLock:
    """Simple file-based lock using fcntl."""

    def __init__(self, path: Path, timeout: float = 30.0):
        self.path = path
        self.lock_file = Path(str(path) + ".lock")
        self.timeout = timeout
        self.fd: Optional[int] = None

    def __enter__(self):
        import time

        start_time = time.time()

        while True:
            try:
                self.fd = os.open(
                    str(self.lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC
                )
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (OSError, IOError):
                if self.fd:
                    os.close(self.fd)
                    self.fd = None

                if time.time() - start_time > self.timeout:
                    raise TimeoutError(f"Could not acquire lock on {self.path}")

                # Sleep and retry
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
                os.close(self.fd)
            except OSError:
                pass

            try:
                os.unlink(str(self.lock_file))
            except OSError:
                pass


# Misc utilities
def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON, returning default on failure."""
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}")
        return default


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def safe_file_read(path: Path, default: str = "") -> str:
    """Safely read file, returning default on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return default


def safe_file_write(path: Path, content: str) -> bool:
    """Safely write file, returning success status."""
    try:
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.error(f"Failed to write {path}: {e}")
        return False
