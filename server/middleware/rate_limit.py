from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import HTTPException

from server.auth import decode_token  # type: ignore

from ..config import RATE_LIMITS as _RATE_LIMITS, RATE_WINDOW_SEC as _RATE_WINDOW_SEC


_RATE_BUCKETS: dict[str, deque] = defaultdict(deque)
_JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")


async def rate_limit(request, call_next):
    path = request.url.path
    if not (
        path.startswith("/api/lookup")
        or path.startswith("/api/parse")
        or path.startswith("/translate")
    ):
        return await call_next(request)

    tier = "Free"
    key: Optional[str] = None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1]
        try:
            data = decode_token(token, _JWT_SECRET, ["HS256"])  # type: ignore
            uid = (data or {}).get("sub")
            if uid:
                key = f"u:{uid}:{path}"
        except Exception:
            pass
    if not key:
        client = request.client.host if request.client else "unknown"
        key = f"ip:{client}:{path}"

    if tier == "admin":
        return await call_next(request)

    limit = _RATE_LIMITS.get(tier, _RATE_LIMITS.get("Free", 60))
    now = time.time()
    window = float(_RATE_WINDOW_SEC)
    dq = _RATE_BUCKETS[key]
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        raise HTTPException(429, "Rate limit exceeded")
    dq.append(now)
    return await call_next(request)


def install_rate_limit(app) -> None:
    app.middleware("http")(rate_limit)

