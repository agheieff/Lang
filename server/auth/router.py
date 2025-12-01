from __future__ import annotations

from typing import Optional, Type, Any
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException, Header, status, Request, Cookie

from .schemas import AccountCreate, AccountOut, LoginIn, TokenOut
from .security import hash_password, verify_password, create_access_token
from .auth_utils import parse_bearer_token, extract_subject
from .repo import AuthRepository
from .policy import validate_password


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
            "role": acc.get("role"),
            "subscription_tier": acc.get("subscription_tier"),
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
        # TODO: Remove this in production
        acc = repo.create_account(
            email, 
            hash_password(payload.password),
            role="admin",
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
        return _to_account_out(acc)

    return r

