from __future__ import annotations

from typing import Optional, Callable
import logging

import os
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from .db import GlobalSessionLocal, get_global_db
from server.auth import Account, decode_token

_JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
logger = logging.getLogger(__name__)


def get_current_account(
    request: Request,
    db: Session = Depends(get_global_db),
    authorization: Optional[str] = Header(default=None),
) -> Account:
    token: Optional[str] = None

    # Extract token from Authorization header or cookie
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    else:
        token = request.cookies.get("access_token")

    if not token:
        logger.warning("Authentication failed: No token provided")
        raise HTTPException(401, "Not authenticated")

    # Validate token format
    if not isinstance(token, str) or not token.strip():
        logger.warning("Authentication failed: Invalid token format")
        raise HTTPException(401, "Invalid token format")

    try:
        # Decode and validate JWT token
        data = decode_token(token, _JWT_SECRET, ["HS256"])
        if not data or "sub" not in data:
            logger.warning("Authentication failed: Invalid token payload")
            raise HTTPException(401, "Invalid token payload")

        account_id = int(data.get("sub"))
        if account_id <= 0:
            logger.warning(f"Authentication failed: Invalid account ID: {account_id}")
            raise HTTPException(401, "Invalid account ID")

    except ValueError as e:
        logger.warning(f"Authentication failed: Invalid account ID format: {e}")
        raise HTTPException(401, "Invalid account ID format")
    except Exception as e:
        logger.warning(f"Authentication failed: Token validation error: {e}")
        raise HTTPException(401, "Invalid token")

    # Retrieve account from database
    account = db.get(Account, account_id)
    if not account:
        logger.warning(f"Authentication failed: Account not found: {account_id}")
        raise HTTPException(401, "Account not found")

    # Check if account is active
    if not getattr(account, "is_active", True):
        logger.warning(f"Authentication failed: Account inactive: {account_id}")
        raise HTTPException(401, "Account inactive")

    # Ensure a default tier is assigned for newly registered users
    # We do it here because registration happens in a shared router we don't control.
    if not getattr(account, "subscription_tier", None):
        try:
            account.subscription_tier = "Free"
            db.commit()
        except Exception as e:
            logger.warning(f"Could not assign default tier to account {account_id}: {e}")

    return account


def require_tier(allowed: set[str]) -> Callable[[Account], Account]:
    def dep(account: Account = Depends(get_current_account)) -> Account:
        if not allowed:
            logger.error("Authorization misconfiguration: No allowed tiers specified")
            raise HTTPException(500, "Authorization configuration error")

        if not hasattr(account, "subscription_tier") or not account.subscription_tier:
            logger.warning(f"Authorization failed: Account {account.id} has no subscription tier")
            raise HTTPException(403, "No subscription tier assigned")

        if account.subscription_tier not in allowed:
            logger.warning(
                f"Authorization failed: Account {account.id} with tier '{account.subscription_tier}' "
                f"not in allowed tiers: {allowed}"
            )
            raise HTTPException(403, "Insufficient subscription tier")

        return account
    return dep
