from __future__ import annotations

from typing import Any, Callable, Optional, Dict, Type

from sqlalchemy.orm import Session
from sqlalchemy import func

from ..repo import AuthRepository


class SQLAlchemyRepo(AuthRepository):
    """Generic SQLAlchemy adapter.

    Configure with model classes and field names used by the project.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        UserModel: Type[Any],
        ProfileModel: Type[Any],
        user_email_field: str = "email",
        user_password_hash_field: str = "password_hash",
        user_is_active_field: str = "is_active",
        user_is_verified_field: str = "is_verified",
        user_role_field: Optional[str] = None,
        user_subscription_field: Optional[str] = None,
        profile_account_fk_field: str = "user_id",
        profile_display_name_field: Optional[str] = "display_name",
        profile_prefs_field: Optional[str] = "prefs",
        profile_extras_field: Optional[str] = None,
    ) -> None:
        self._sf = session_factory
        self.U = UserModel
        self.P = ProfileModel
        self.uf_email = user_email_field
        self.uf_pwd = user_password_hash_field
        self.uf_active = user_is_active_field
        self.uf_verified = user_is_verified_field
        self.uf_role = user_role_field
        self.uf_sub = user_subscription_field
        self.pf_acc = profile_account_fk_field
        self.pf_name = profile_display_name_field
        self.pf_prefs = profile_prefs_field
        self.pf_extras = profile_extras_field

    # ---- helpers ----
    def _acc_to_dict(self, u: Any) -> Dict[str, Any]:
        return {
            "id": getattr(u, "id"),
            "email": getattr(u, self.uf_email),
            "is_active": bool(getattr(u, self.uf_active, True)),
            "is_verified": bool(getattr(u, self.uf_verified, True)),
            "role": (getattr(u, self.uf_role) if self.uf_role else None),
            "subscription_tier": (getattr(u, self.uf_sub) if self.uf_sub else None),
            "name": getattr(u, "name", None),
            "extras": None,
        }

    def _acc_to_credentials(self, u: Any) -> Dict[str, Any]:
        return {
            "id": getattr(u, "id"),
            "password_hash": getattr(u, self.uf_pwd),
            "is_active": bool(getattr(u, self.uf_active, True)),
            "is_verified": bool(getattr(u, self.uf_verified, True)),
        }


    # ---- repo API ----
    def find_account_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        s = self._sf()
        try:
            key = email.strip().lower()
            u = (
                s.query(self.U)
                .filter(func.lower(getattr(self.U, self.uf_email)) == key)
                .first()
            )
            return (self._acc_to_dict(u) if u else None)
        finally:
            s.close()

    def get_account_credentials(self, email: str) -> Optional[Dict[str, Any]]:
        s = self._sf()
        try:
            key = email.strip().lower()
            u = (
                s.query(self.U)
                .filter(func.lower(getattr(self.U, self.uf_email)) == key)
                .first()
            )
            return (self._acc_to_credentials(u) if u else None)
        finally:
            s.close()

    def create_account(self, email: str, password_hash: str) -> Dict[str, Any]:
        s = self._sf()
        try:
            u = self.U()
            setattr(u, self.uf_email, email.strip().lower())
            setattr(u, self.uf_pwd, password_hash)
            if hasattr(u, self.uf_active):
                setattr(u, self.uf_active, True)
            if hasattr(u, self.uf_verified):
                setattr(u, self.uf_verified, True)
            s.add(u)
            s.commit()
            s.refresh(u)
            return self._acc_to_dict(u)
        finally:
            s.close()

    def get_account_by_id(self, account_id: str | int) -> Optional[Dict[str, Any]]:
        s = self._sf()
        try:
            u = s.get(self.U, account_id)
            return (self._acc_to_dict(u) if u else None)
        finally:
            s.close()

    # Profile management has been removed - applications should implement their own profile systems

