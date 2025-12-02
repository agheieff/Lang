from __future__ import annotations

import os

from server.auth import CookieUserMiddleware, Account
from server.db import GlobalSessionLocal


def install_auth(app) -> None:
    """Install auth middleware that loads full user object including role/tier."""
    try:
        secret = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
        app.add_middleware(
            CookieUserMiddleware,
            session_factory=GlobalSessionLocal,
            UserModel=Account,
            secret_key=secret,
            algorithm="HS256",
            cookie_name="access_token",
        )
    except Exception:
        # Non-fatal during dev
        pass

