from __future__ import annotations

import os

from server.auth import mount_cookie_agent_middleware  # type: ignore


def install_auth(app) -> None:
    try:
        secret = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
        mount_cookie_agent_middleware(app, secret_key=secret)
    except Exception:
        # Non-fatal during dev
        pass

