from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db, get_global_db
from ..models import SubscriptionTier
from ..repos.tiers import ensure_default_tiers
from ..deps import get_current_account
from arcadia_auth import Account


router = APIRouter()


class TierOut(BaseModel):
    name: str
    description: str | None = None


@router.get("/tiers", response_model=List[TierOut])
def list_tiers(db: Session = Depends(get_global_db)):
    ensure_default_tiers(db)
    rows = db.query(SubscriptionTier).order_by(SubscriptionTier.name.asc()).all()
    return [TierOut(name=r.name, description=r.description) for r in rows]


class TierIn(BaseModel):
    name: str


@router.get("/me/tier", response_model=TierIn)
def get_my_tier(account: Account = Depends(get_current_account)):
    return TierIn(name=(account.subscription_tier or "free"))


@router.post("/me/tier", response_model=TierIn)
def set_my_tier(
    payload: TierIn,
    tiers_db: Session = Depends(get_global_db),
    auth_db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    ensure_default_tiers(tiers_db)
    allowed = {t.name for t in tiers_db.query(SubscriptionTier).all()}
    if payload.name not in allowed:
        raise HTTPException(400, "Unknown tier")
    acc = auth_db.get(Account, account.id)
    if not acc:
        raise HTTPException(401, "Account not found")
    acc.subscription_tier = payload.name
    auth_db.commit()
    return TierIn(name=payload.name)
