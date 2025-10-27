from __future__ import annotations

from typing import Optional, Sequence, Any

from .security import decode_token


def parse_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract raw token from an Authorization header value.

    Accepts values like "Bearer <token>" (case-insensitive). Returns None when
    header is missing or malformed.
    """
    if not authorization:
        return None
    val = authorization.strip()
    if not val.lower().startswith("bearer "):
        return None
    return val.split(" ", 1)[1]


def extract_subject(token: Optional[str], secret_key: str, algorithms: Sequence[str]) -> Optional[Any]:
    """Decode the token and return the subject claim (sub) when valid.

    Returns None if token is missing or invalid.
    """
    if not token:
        return None
    data = decode_token(token, secret_key, list(algorithms))
    return (data.get("sub") if data else None)

