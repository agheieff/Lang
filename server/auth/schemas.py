from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, Dict, Any, Union


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

