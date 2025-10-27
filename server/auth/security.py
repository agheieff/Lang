from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext


def _argon2_available() -> bool:
    try:
        # Prefer passlib's detection; returns False if backend missing
        from passlib.handlers.argon2 import argon2  # type: ignore

        try:
            return bool(getattr(argon2, "has_backend", lambda: False)())
        except Exception:
            return False
    except Exception:
        return False


def _build_pwd_context() -> CryptContext:
    if _argon2_available():
        # Use argon2 when the backend is present; PBKDF2 for legacy hashes
        return CryptContext(schemes=["argon2", "pbkdf2_sha256"], default="argon2", deprecated="auto")
    # Fallback: only PBKDF2 to avoid runtime MissingBackendError
    return CryptContext(schemes=["pbkdf2_sha256"], default="pbkdf2_sha256", deprecated="auto")


# Password hashing context: prefer argon2id when available, fallback to PBKDF2
pwd_context = _build_pwd_context()


def set_password_context(context: CryptContext) -> None:
    """Allow applications to replace the CryptContext at runtime."""
    global pwd_context
    pwd_context = context


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(plain_password, password_hash)
    except Exception:
        return False


def create_access_token(subject: str | int, secret_key: str, algorithm: str = "HS256", expires_minutes: int = 60 * 24 * 7) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=expires_minutes)
    payload = {"sub": str(subject), "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_token(token: str, secret_key: str, algorithms: list[str] = ["HS256"]) -> Optional[dict]:
    try:
        return jwt.decode(token, secret_key, algorithms=algorithms)
    except JWTError:
        return None

