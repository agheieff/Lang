"""
OpenRouter key provisioning and management service.
"""

from __future__ import annotations

import logging
import os
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from server.models import Account

logger = logging.getLogger(__name__)

# OpenRouter API endpoints
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


def get_provisioning_key() -> Optional[str]:
    """Get master provisioning key from environment."""
    return os.getenv("OPENROUTER_PROVISIONING_KEY")


def get_encryption_key() -> Optional[str]:
    """Get encryption secret for storing user keys."""
    return os.getenv("OPENROUTER_KEY_ENCRYPTION_SECRET")


def provision_user_key(
    db: Session,
    account_id: int,
    budget_usd: float = 25.0,
    duration_days: int = 30,
) -> Dict[str, Any]:
    """
    Provision a sub-key for a user using:master provisioning key.

    Returns:
        Dict with 'key_id', 'key', 'limits', 'created_at'
    """
    try:
        master_key = get_provisioning_key()
        if not master_key:
            raise ValueError("OPENROUTER_PROVISIONING_KEY not set")

        account = db.query(Account).filter(Account.id == account_id).first()
        if not account:
            raise ValueError(f"Account {account_id} not found")

        if account.openrouter_key_id:
            logger.info(
                f"Account {account_id} already has a sub-key: {account.openrouter_key_id}"
            )
            return {
                "key_id": account.openrouter_key_id,
                "key": None,
                "message": "User already has a provisioned key",
            }

        # Call OpenRouter provisioning API
        url = f"{OPENROUTER_API_BASE}/auth/keys"

        headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        }

        data = {
            "label": f"Arcadia Lang - User {account_id}",
            "budget": budget_usd,
            "period": "monthly",
            "limits": {
                "budget": budget_usd,
                "period": "monthly",
            },
            "models": [],
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()

        # Extract key info
        key_data = result.get("data", {})
        sub_key_id = key_data.get("id")
        sub_key = key_data.get("key")
        limits = key_data.get("limits", {})

        # Encrypt and store in database
        encryption_secret = get_encryption_key()
        if not encryption_secret:
            raise ValueError("OPENROUTER_KEY_ENCRYPTION_SECRET not set")

        from cryptography.fernet import Fernet

        fernet = Fernet(encryption_secret.encode())
        encrypted_key = fernet.encrypt(sub_key.encode()).decode()

        # Update account instance
        account.openrouter_key_id = sub_key_id
        account.openrouter_key_encrypted = encrypted_key
        account.updated_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"Provisioned sub-key {sub_key_id} for account {account_id}")

        return {
            "key_id": sub_key_id,
            "key": sub_key,
            "limits": limits,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    except httpx.HTTPStatusError as e:
        logger.error(
            f"OpenRouter API error: {e.response.status_code} - {e.response.text}"
        )
        raise RuntimeError(f"Failed to provision key: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Error provisioning key: {e}", exc_info=True)
        raise


def get_user_openrouter_key(db: Session, account_id: int) -> Optional[str]:
    """
    Decrypt and return user's OpenRouter sub-key.

    Returns:
        Decrypted API key string, or None if not found
    """
    try:
        account = db.query(Account).filter(Account.id == account_id).first()
        if not account or not account.openrouter_key_encrypted:
            return None

        encryption_secret = get_encryption_key()
        if not encryption_secret:
            logger.error("OPENROUTER_KEY_ENCRYPTION_SECRET not set")
            return None

        from cryptography.fernet import Fernet

        fernet = Fernet(encryption_secret.encode())
        decrypted_key = fernet.decrypt(
            account.openrouter_key_encrypted.encode()
        ).decode()

        return decrypted_key

    except Exception as e:
        logger.error(f"Error decrypting user key: {e}", exc_info=True)
        return None


def revoke_user_key(
    db: Session,
    account_id: int,
) -> bool:
    """
    Revoke a user's OpenRouter sub-key.

    Returns:
        True if revoked successfully, False otherwise
    """
    try:
        master_key = get_provisioning_key()
        if not master_key:
            logger.warning("OPENROUTER_PROVISIONING_KEY not set, cannot revoke via API")

        account = db.query(Account).filter(Account.id == account_id).first()
        if not account or not account.openrouter_key_id:
            return False

        # Try to revoke via API if we have master key
        if master_key:
            url = f"{OPENROUTER_API_BASE}/auth/keys/{account.openrouter_key_id}"

            headers = {
                "Authorization": f"Bearer {master_key}",
                "Content-Type": "application/json",
            }

            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.delete(url, headers=headers)
                    # Don't raise on 404 - key may already be deleted
                    if response.status_code != 404:
                        response.raise_for_status()
            except Exception as e:
                logger.warning(
                    f"Failed to revoke via API: {e}, clearing from DB anyway"
                )

        # Clear from database
        account.openrouter_key_id = None
        account.openrouter_key_encrypted = None
        account.updated_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"Revoked/cleared sub-key for account {account_id}")
        return True

    except Exception as e:
        logger.error(f"Error revoking key: {e}", exc_info=True)
        return False


def get_user_key_usage(db: Session, account_id: int) -> Dict[str, Any]:
    """
    Get usage stats for a user's OpenRouter sub-key.

    Returns:
        Dict with 'budget', 'used', 'remaining', 'reset_date'
    """
    try:
        master_key = get_provisioning_key()
        if not master_key:
            return {"error": "Provisioning key not set"}

        account = db.query(Account).filter(Account.id == account_id).first()
        if not account or not account.openrouter_key_id:
            return {"error": "No key provisioned for this user"}

        # Call OpenRouter API to get key info
        url = f"{OPENROUTER_API_BASE}/auth/keys/{account.openrouter_key_id}"

        headers = {
            "Authorization": f"Bearer {master_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

        key_data = result.get("data", {})
        limits = key_data.get("limits", {})
        usage = key_data.get("usage", {})

        budget = limits.get("budget", 0)
        used = usage.get("spend", 0)
        remaining = max(0, budget - used)

        reset_date = key_data.get("refreshed_at") or key_data.get("created_at")

        return {
            "budget_usd": budget,
            "used_usd": used,
            "remaining_usd": remaining,
            "reset_date": reset_date,
            "key_id": account.openrouter_key_id,
        }

    except Exception as e:
        logger.error(f"Error getting key usage: {e}", exc_info=True)
        return {"error": str(e)}


def ensure_user_has_key(
    db: Session,
    account_id: int,
    tier: str,
) -> bool:
    """
    Ensure a user has an OpenRouter sub-key based on their subscription tier.

    Returns:
        True if user has a key (existing or newly provisioned)
    """
    try:
        account = db.query(Account).filter(Account.id == account_id).first()
        if not account:
            return False

        # Check if already has a key
        if account.openrouter_key_id:
            return True

        # Determine budget based on tier
        from server.config import TIER_SPENDING_LIMITS

        budget = TIER_SPENDING_LIMITS.get(tier)

        if budget is None:
            # Free tier or BYOK - no key needed
            return False

        # Provision key
        provision_user_key(
            db=db,
            account_id=account_id,
            budget_usd=budget,
            duration_days=30,
        )

        return True

    except Exception as e:
        logger.error(f"Error ensuring user has key: {e}", exc_info=True)
        return False
