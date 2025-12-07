"""
Flattened authentication module consolidating models, JWT, password handling,
and routing into a single file for better modularity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union, Sequence, Type
from dataclasses import dataclass
import re

from sqlalchemy import Column, Integer, String, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Header, status, Request, Cookie
from jose import jwt, JWTError
from passlib.context import CryptContext


# =============================================================================
# Database Models
# =============================================================================

Base = declarative_base()


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Core fields from original schema
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False) 
    # Unified tier for both access control and features: Free|Standard|Pro|Pro+|BYOK|admin|system
    subscription_tier = Column(String(50), default="Free", nullable=False)
    
    # OpenRouter per-user key management (for paid tiers)
    openrouter_key_encrypted = Column(Text, nullable=True)  # Fernet-encrypted API key
    openrouter_key_id = Column(String(128), nullable=True)  # OpenRouter key hash/identifier for management
    
    # Extended fields - apps can add their own here
    # Note: Profile fields (name, timezone, avatar_url) have been removed
    # Applications should implement their own profile systems
    
    # Metadata with JSON fallback for truly flexible data
    extras = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by AuthRepository interface"""
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "subscription_tier": self.subscription_tier,
            "extras": self.extras,
            "has_openrouter_key": self.openrouter_key_id is not None,
        }


def create_sqlite_engine(database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False):
    """Create SQLite engine with proper configuration"""
    engine = create_engine(database_url, echo=echo, connect_args={"check_same_thread": False})
    return engine


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


# =============================================================================
# Pydantic Schemas
# =============================================================================

class AccountCreate(BaseModel):
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class AccountOut(BaseModel):
    id: Union[str, int]
    email: str
    is_active: bool = True
    is_verified: bool = True
    subscription_tier: str = "Free"  # Free|Standard|Pro|Pro+|BYOK|admin|system
    extras: Optional[Dict[str, Any]] = None


# =============================================================================
# Password & JWT Security
# =============================================================================

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


# =============================================================================
# Auth Utilities & Policy
# =============================================================================

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


def validate_password(password: str, settings: Any) -> Optional[str]:
    """Validate password against settings.

    Returns:
      None if OK, else a human‑readable rejection message string.

    The settings object is expected to expose attributes:
      pwd_min_len: Optional[int]
      pwd_max_len: Optional[int]
      require_upper: bool
      require_lower: bool
      require_digit: bool
      require_special: bool
    Missing attributes fall back to permissive defaults.
    """
    pw = password or ""
    min_len: Optional[int] = getattr(settings, "pwd_min_len", 8)
    max_len: Optional[int] = getattr(settings, "pwd_max_len", 256)
    req_up: bool = bool(getattr(settings, "require_upper", False))
    req_lo: bool = bool(getattr(settings, "require_lower", False))
    req_di: bool = bool(getattr(settings, "require_digit", False))
    req_sp: bool = bool(getattr(settings, "require_special", False))

    if isinstance(min_len, int) and min_len > 0 and len(pw) < min_len:
        return f"Password must be at least {min_len} characters"
    if isinstance(max_len, int) and max_len > 0 and len(pw) > max_len:
        return f"Password must be at most {max_len} characters"
    if req_up and not re.search(r"[A-Z]", pw):
        return "Password must include an uppercase letter"
    if req_lo and not re.search(r"[a-z]", pw):
        return "Password must include a lowercase letter"
    if req_di and not re.search(r"\d", pw):
        return "Password must include a number"
    if req_sp and not re.search(r"[^0-9A-Za-z]", pw):
        return "Password must include a special character"
    return None


# =============================================================================
# Auth Repository
# =============================================================================

class AuthRepository(ABC):
    """Abstract repository to be implemented per project/DB.

    IDs are opaque (int or str). Implementations should normalize email casing.
    """

    @abstractmethod
    def find_account_by_email(self, email: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def get_account_credentials(self, email: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def create_account(self, email: str, password_hash: str, subscription_tier: str = "Free") -> Dict[str, Any]: ...

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

    def create_account(self, email: str, password_hash: str, subscription_tier: str = "Free") -> Dict[str, Any]:
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
            "subscription_tier": subscription_tier,
            "extras": None,
        }
        self.accounts[aid] = acc
        self.accounts_by_email[acc["email"]] = aid
        return self._public_acc(acc)

    def get_account_by_id(self, account_id: int) -> Optional[Dict[str, Any]]:
        acc = self.accounts.get(int(account_id))
        return (self._public_acc(acc) if acc else None)


# =============================================================================
# FastAPI Router
# =============================================================================

@dataclass
class AuthSettings:
    secret_key: str
    algorithm: str = "HS256"
    access_expire_minutes: int = 60 * 24 * 7
    # Password policy (optional)
    pwd_min_len: int = 8
    pwd_max_len: int = 256
    require_upper: bool = False
    require_lower: bool = False
    require_digit: bool = False
    require_special: bool = False  # non-alnum


def _auth_header(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    access_token: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    # Prefer Bearer token from Authorization; fallback to access_token cookie
    token = parse_bearer_token(authorization)
    if token:
        return token
    if access_token:
        return access_token
    try:
        ck = request.cookies.get("access_token")
        if ck:
            return ck
    except Exception:
        pass
    return None


def create_auth_router(
    repo: AuthRepository,
    settings: AuthSettings,
    *,
    AccountPublic: Type[AccountOut] = AccountOut,
) -> APIRouter:
    r = APIRouter(prefix="/auth", tags=["auth"])

    def _to_account_out(acc: dict[str, Any]) -> AccountOut:
        return AccountPublic.model_validate({
            "id": acc.get("id"),
            "email": acc.get("email"),
            "is_active": bool(acc.get("is_active", True)),
            "is_verified": bool(acc.get("is_verified", True)),
            "subscription_tier": acc.get("subscription_tier", "Free"),
            "extras": acc.get("extras"),
        })

    @r.post("/register", response_model=AccountOut, status_code=status.HTTP_201_CREATED)
    def register(payload: AccountCreate):
        email = payload.email.strip().lower()
        if repo.find_account_by_email(email):
            raise HTTPException(status_code=409, detail="Email already registered")
        # Validate password policy; None means OK
        msg = validate_password(payload.password, settings)
        if msg:
            raise HTTPException(status_code=422, detail=msg)
        # Default all new accounts to admin for development
        # TODO: Change to "Free" in production
        acc = repo.create_account(
            email, 
            hash_password(payload.password),
            subscription_tier="admin",
        )
        return _to_account_out(acc)

    @r.post("/login", response_model=TokenOut)
    def login(payload: LoginIn):
        email = payload.email.strip().lower()
        creds = repo.get_account_credentials(email)
        if not creds or not verify_password(payload.password, creds.get("password_hash", "")):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not creds.get("is_active", True):
            raise HTTPException(status_code=403, detail="Inactive account")
        token = create_access_token(creds["id"], settings.secret_key, settings.algorithm, settings.access_expire_minutes)
        return TokenOut(access_token=token)

    @r.get("/logout")
    def logout():
        from fastapi.responses import RedirectResponse
        resp = RedirectResponse(url="/", status_code=302)
        try:
            resp.delete_cookie("access_token", path="/")
        except Exception:
            pass
        return resp

    @r.get("/me", response_model=AccountOut)
    def me(authorization: Optional[str] = Depends(_auth_header)):
        if not authorization:
            raise HTTPException(status_code=401, detail="Not authenticated")
        sub = extract_subject(authorization, settings.secret_key, [settings.algorithm])
        if sub is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        acc = repo.get_account_by_id(sub)  # type: ignore[arg-type]
        if not acc:
            raise HTTPException(status_code=401, detail="User not found")
        return r


# =============================================================================
# CookieUserMiddleware
# =============================================================================

from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import HTTPException


class CookieUserMiddleware(BaseHTTPMiddleware):
    """Middleware that loads user from JWT cookie and adds to request state."""
    
    def __init__(
        self,
        app,
        session_factory,
        UserModel,
        secret_key: str,
        algorithm: str = "HS256",
        cookie_name: str = "access_token",
    ):
        super().__init__(app)
        self.session_factory = session_factory
        self.UserModel = UserModel
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.cookie_name = cookie_name
    
    async def dispatch(self, request: Request, call_next):
        # Try to get token from cookie
        token = request.cookies.get(self.cookie_name)
        
        if token:
            try:
                # Decode token
                data = decode_token(token, self.secret_key, [self.algorithm])
                if data and "sub" in data:
                    user_id = data["sub"]
                    
                    # Load user from database
                    with self.session_factory() as session:
                        user = session.query(self.UserModel).filter(
                            self.UserModel.id == user_id
                        ).first()
                        
                        if user:
                            # Add user to request state
                            request.state.user = user
                            # Support both dict and attribute access
                            request.state.user_id = getattr(user, 'id', user.get('id') if isinstance(user, dict) else None)
                            request.state.account_id = request.state.user_id
                        else:
                            # Invalid user - clear cookie
                            response = await call_next(request)
                            response.delete_cookie(self.cookie_name, path="/")
                            return response
                            
            except Exception as e:
                # Invalid token - continue without user
                pass
        
        # Continue processing
        response = await call_next(request)
        return response
