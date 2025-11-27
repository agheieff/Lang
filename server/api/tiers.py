from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_global_db
from ..account_db import get_db
from ..models import SubscriptionTier
from ..repos.tiers import ensure_default_tiers
from ..deps import get_current_account
from ..config import PAID_TIERS, TIER_SPENDING_LIMITS
from ..services.openrouter_key_service import (
    get_openrouter_key_service,
    OpenRouterKeyError,
)
from server.auth import Account

logger = logging.getLogger(__name__)

router = APIRouter()


class TierOut(BaseModel):
    name: str
    description: str | None = None


@router.get("/tiers", response_model=List[TierOut])
def list_tiers(db: Session = Depends(get_global_db)):
    ensure_default_tiers(db)
    rows = db.query(SubscriptionTier).all()
    order = ["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"]
    rows.sort(key=lambda r: (order.index(r.name) if r.name in order else len(order), r.name))
    return [TierOut(name=r.name, description=r.description) for r in rows if r.name != "admin"]


class TierIn(BaseModel):
    name: str


@router.get("/me/tier", response_model=TierIn)
def get_my_tier(account: Account = Depends(get_current_account)):
    # Default to productized baseline tier when unset
    return TierIn(name=(account.subscription_tier or "Free"))


class TierChangeOut(BaseModel):
    name: str
    has_openrouter_key: bool = False
    message: Optional[str] = None


async def _provision_key_for_tier(db: Session, account: Account, tier: str) -> None:
    """Background task to provision OpenRouter key for paid tiers"""
    if tier not in PAID_TIERS:
        return
    try:
        key_service = get_openrouter_key_service()
        await key_service.create_user_key(db, account, tier)
    except OpenRouterKeyError as e:
        logger.error(f"Failed to provision key for account {account.id}: {e}")


async def _revoke_key_if_exists(db: Session, account: Account) -> None:
    """Revoke OpenRouter key when downgrading from paid tier"""
    if not account.openrouter_key_id:
        return
    try:
        key_service = get_openrouter_key_service()
        await key_service.revoke_user_key(db, account)
    except OpenRouterKeyError as e:
        logger.error(f"Failed to revoke key for account {account.id}: {e}")


@router.post("/me/tier", response_model=TierChangeOut)
async def set_my_tier(
    payload: TierIn,
    background_tasks: BackgroundTasks,
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

    old_tier = acc.subscription_tier or "Free"
    new_tier = payload.name

    # Update tier
    acc.subscription_tier = new_tier
    auth_db.commit()

    message = None

    # Handle key provisioning/revocation
    if new_tier in PAID_TIERS and old_tier not in PAID_TIERS:
        # Upgrading to paid tier - provision key
        await _provision_key_for_tier(auth_db, acc, new_tier)
        # Refresh to get updated key status
        auth_db.refresh(acc)
        if acc.openrouter_key_id:
            limit = TIER_SPENDING_LIMITS.get(new_tier)
            message = f"OpenRouter key provisioned with ${limit}/month limit"
        else:
            message = "Key provisioning in progress"

    elif old_tier in PAID_TIERS and new_tier not in PAID_TIERS:
        # Downgrading from paid tier - revoke key
        await _revoke_key_if_exists(auth_db, acc)
        auth_db.refresh(acc)
        message = "OpenRouter key revoked"

    elif new_tier in PAID_TIERS and old_tier in PAID_TIERS and new_tier != old_tier:
        # Changing between paid tiers - update limit
        key_service = get_openrouter_key_service()
        if acc.openrouter_key_id:
            new_limit = TIER_SPENDING_LIMITS.get(new_tier)
            try:
                await key_service.update_key_limit(auth_db, acc, new_limit)
                message = f"OpenRouter key limit updated to ${new_limit}/month"
            except OpenRouterKeyError as e:
                logger.error(f"Failed to update key limit: {e}")
                message = "Tier updated, key limit update pending"

    return TierChangeOut(
        name=new_tier,
        has_openrouter_key=acc.openrouter_key_id is not None,
        message=message,
    )
