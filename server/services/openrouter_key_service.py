from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import httpx
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from server.auth import Account
from server.config import (
    OPENROUTER_PROVISIONING_KEY,
    OPENROUTER_KEY_ENCRYPTION_SECRET,
    TIER_SPENDING_LIMITS,
    PAID_TIERS,
)

logger = logging.getLogger(__name__)

_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_KEYS_ENDPOINT = f"{_OPENROUTER_API_BASE}/keys"


class OpenRouterKeyError(Exception):
    """Base exception for OpenRouter key operations"""
    pass


class OpenRouterKeyService:
    """Manages per-user OpenRouter API keys for paid tiers"""

    def __init__(self):
        self._fernet: Optional[Fernet] = None
        self._init_encryption()

    def _init_encryption(self) -> None:
        secret = OPENROUTER_KEY_ENCRYPTION_SECRET
        if not secret:
            logger.warning("OPENROUTER_KEY_ENCRYPTION_SECRET not set - key encryption disabled")
            return
        # Derive a valid Fernet key from the secret
        key = hashlib.sha256(secret.encode()).digest()
        import base64
        fernet_key = base64.urlsafe_b64encode(key)
        self._fernet = Fernet(fernet_key)

    def _encrypt_key(self, api_key: str) -> str:
        if not self._fernet:
            raise OpenRouterKeyError("Encryption not configured")
        return self._fernet.encrypt(api_key.encode()).decode()

    def _decrypt_key(self, encrypted: str) -> str:
        if not self._fernet:
            raise OpenRouterKeyError("Encryption not configured")
        try:
            return self._fernet.decrypt(encrypted.encode()).decode()
        except InvalidToken:
            raise OpenRouterKeyError("Failed to decrypt key - invalid token")

    async def create_user_key(
        self,
        db: Session,
        account: Account,
        tier: str,
    ) -> str:
        """
        Create a new OpenRouter API key for a paid-tier user.
        Returns the API key (shown once, then encrypted in DB).
        """
        if tier not in PAID_TIERS:
            raise OpenRouterKeyError(f"Tier '{tier}' does not get individual keys")

        if not OPENROUTER_PROVISIONING_KEY:
            raise OpenRouterKeyError("OPENROUTER_PROVISIONING_KEY not configured")

        # Revoke existing key if present
        if account.openrouter_key_id:
            try:
                await self.revoke_user_key(db, account)
            except Exception as e:
                logger.warning(f"Failed to revoke old key for account {account.id}: {e}")

        limit = TIER_SPENDING_LIMITS.get(tier)
        payload = {
            "name": f"arcadia-user-{account.id}-{tier}",
        }
        if limit is not None:
            payload["limit"] = limit

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                _OPENROUTER_KEYS_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_PROVISIONING_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if resp.status_code not in (200, 201):
                logger.error(f"OpenRouter key creation failed: {resp.status_code} {resp.text}")
                raise OpenRouterKeyError(f"Failed to create key: {resp.status_code}")

            response_data = resp.json()

        # API returns {"data": {...}, "key": "..."}
        api_key = response_data.get("key")
        key_data = response_data.get("data", {})
        key_hash = key_data.get("hash")

        if not api_key or not key_hash:
            raise OpenRouterKeyError("Invalid response from OpenRouter API")

        # Store encrypted key and hash in DB
        account.openrouter_key_encrypted = self._encrypt_key(api_key)
        account.openrouter_key_id = key_hash
        db.commit()

        logger.info(f"Created OpenRouter key for account {account.id} (tier: {tier})")
        return api_key

    async def revoke_user_key(self, db: Session, account: Account) -> bool:
        """Revoke and remove a user's OpenRouter key"""
        if not account.openrouter_key_id:
            return False

        if not OPENROUTER_PROVISIONING_KEY:
            raise OpenRouterKeyError("OPENROUTER_PROVISIONING_KEY not configured")

        key_hash = account.openrouter_key_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{_OPENROUTER_KEYS_ENDPOINT}/{key_hash}",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_PROVISIONING_KEY}",
                },
            )

            if resp.status_code not in (200, 204, 404):
                logger.error(f"OpenRouter key revocation failed: {resp.status_code} {resp.text}")
                raise OpenRouterKeyError(f"Failed to revoke key: {resp.status_code}")

        # Clear from DB
        account.openrouter_key_encrypted = None
        account.openrouter_key_id = None
        db.commit()

        logger.info(f"Revoked OpenRouter key for account {account.id}")
        return True

    async def update_key_limit(
        self,
        db: Session,
        account: Account,
        new_limit: Optional[float],
    ) -> bool:
        """Update the spending limit on a user's OpenRouter key"""
        if not account.openrouter_key_id:
            raise OpenRouterKeyError("Account has no OpenRouter key")

        if not OPENROUTER_PROVISIONING_KEY:
            raise OpenRouterKeyError("OPENROUTER_PROVISIONING_KEY not configured")

        key_hash = account.openrouter_key_id
        payload = {}
        if new_limit is not None:
            payload["limit"] = new_limit

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.patch(
                f"{_OPENROUTER_KEYS_ENDPOINT}/{key_hash}",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_PROVISIONING_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if resp.status_code not in (200, 201):
                logger.error(f"OpenRouter key update failed: {resp.status_code} {resp.text}")
                raise OpenRouterKeyError(f"Failed to update key: {resp.status_code}")

        logger.info(f"Updated OpenRouter key limit for account {account.id} to ${new_limit}")
        return True

    def get_user_key(self, account: Account) -> Optional[str]:
        """Get the decrypted API key for a user (if they have one)"""
        if not account.openrouter_key_encrypted:
            return None
        try:
            return self._decrypt_key(account.openrouter_key_encrypted)
        except OpenRouterKeyError:
            logger.error(f"Failed to decrypt key for account {account.id}")
            return None

    def user_has_key(self, account: Account) -> bool:
        """Check if user has an individual OpenRouter key"""
        return account.openrouter_key_id is not None


# Singleton instance
_key_service: Optional[OpenRouterKeyService] = None


def get_openrouter_key_service() -> OpenRouterKeyService:
    global _key_service
    if _key_service is None:
        _key_service = OpenRouterKeyService()
    return _key_service
