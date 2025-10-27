from __future__ import annotations

from typing import Optional, Any
import re


def validate_password(password: str, settings: Any) -> Optional[str]:
    """Validate password against settings.

    Returns:
      None if OK, else a human‑readable rejection message string.

    The settings object is expected to expose attributes:
      pwd_min_len: Optional[int]
      pwd_max_len: Optional[int]
      require_upper: bool
      require_lower: bool
      require_digit: bool
      require_special: bool
    Missing attributes fall back to permissive defaults.
    """
    pw = password or ""
    min_len: Optional[int] = getattr(settings, "pwd_min_len", 8)
    max_len: Optional[int] = getattr(settings, "pwd_max_len", 256)
    req_up: bool = bool(getattr(settings, "require_upper", False))
    req_lo: bool = bool(getattr(settings, "require_lower", False))
    req_di: bool = bool(getattr(settings, "require_digit", False))
    req_sp: bool = bool(getattr(settings, "require_special", False))

    if isinstance(min_len, int) and min_len > 0 and len(pw) < min_len:
        return f"Password must be at least {min_len} characters"
    if isinstance(max_len, int) and max_len > 0 and len(pw) > max_len:
        return f"Password must be at most {max_len} characters"
    if req_up and not re.search(r"[A-Z]", pw):
        return "Password must include an uppercase letter"
    if req_lo and not re.search(r"[a-z]", pw):
        return "Password must include a lowercase letter"
    if req_di and not re.search(r"\d", pw):
        return "Password must include a number"
    if req_sp and not re.search(r"[^0-9A-Za-z]", pw):
        return "Password must include a special character"
    return None

