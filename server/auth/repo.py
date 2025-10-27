from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class AuthRepository(ABC):
    """Abstract repository to be implemented per project/DB.

    IDs are opaque (int or str). Implementations should normalize email casing.
    """

    @abstractmethod
    def find_account_by_email(self, email: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def get_account_credentials(self, email: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def create_account(self, email: str, password_hash: str) -> Dict[str, Any]: ...

    @abstractmethod
    def get_account_by_id(self, account_id: str | int) -> Optional[Dict[str, Any]]: ...


class MutableAuthRepository(AuthRepository, ABC):
    """Optional extension for repositories that support updates.

    Implementations may choose to provide update methods; callers can use
    isinstance(repo, MutableAuthRepository) to detect support.
    """

    @abstractmethod
    def update_account(self, account_id: str | int, **updates) -> Optional[Dict[str, Any]]: ...


class InMemoryRepo(AuthRepository):
    """Simple in-memory store for testing.

    Not persistent; IDs are ints.
    """

    def __init__(self) -> None:
        self._acc_id = 0
        self.accounts: Dict[int, Dict[str, Any]] = {}
        self.accounts_by_email: Dict[str, int] = {}

    def _next_acc(self) -> int:
        self._acc_id += 1
        return self._acc_id

    def _public_acc(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        # Return a sanitized copy without password_hash
        out = {k: v for k, v in acc.items() if k != "password_hash"}
        return out

    def find_account_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        key = email.strip().lower()
        acc_id = self.accounts_by_email.get(key)
        acc = self.accounts.get(acc_id) if acc_id else None
        return (self._public_acc(acc) if acc else None)

    def get_account_credentials(self, email: str) -> Optional[Dict[str, Any]]:
        key = email.strip().lower()
        acc_id = self.accounts_by_email.get(key)
        acc = self.accounts.get(acc_id) if acc_id else None
        if not acc:
            return None
        return {
            "id": acc["id"],
            "password_hash": acc.get("password_hash", ""),
            "is_active": bool(acc.get("is_active", True)),
            "is_verified": bool(acc.get("is_verified", True)),
        }

    def create_account(self, email: str, password_hash: str) -> Dict[str, Any]:
        if self.find_account_by_email(email):
            raise ValueError("email already registered")
        aid = self._next_acc()
        acc = {
            "id": aid,
            "email": email.strip().lower(),
            "password_hash": password_hash,
            "is_active": True,
            "is_verified": True,
            "role": None,
            "subscription_tier": None,
            "extras": None,
        }
        self.accounts[aid] = acc
        self.accounts_by_email[acc["email"]] = aid
        return self._public_acc(acc)

    def get_account_by_id(self, account_id: int) -> Optional[Dict[str, Any]]:
        acc = self.accounts.get(int(account_id))
        return (self._public_acc(acc) if acc else None)

