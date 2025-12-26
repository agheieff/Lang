from __future__ import annotations

from fastapi import APIRouter, Depends
from typing import Optional
from server.deps import get_current_account
from server.db import get_db
from server.models import Lexeme, Profile
from server.auth import Account
from sqlalchemy.orm import Session

router = APIRouter(tags=["srs"])


@router.get("/srs/urgent")
def srs_urgent():
    """Get urgent SRS words - placeholder for now."""
    return {"words": [], "items": []}


@router.get("/srs/stats")
def get_srs_stats():
    """Get SRS statistics - placeholder for now."""
    return {"total": 0, "by_p": {}, "by_S": {}, "by_D": {}}


@router.get("/srs/level")
def get_srs_level():
    """Get user level - placeholder for now."""
    return {
        "level_value": 0.0,
        "level_var": 1.0,
        "last_update_at": None,
        "last_activity_at": None,
        "ess": 0.0,
        "bins": {},
    }


@router.get("/srs/words")
def get_srs_words(
    lang: Optional[str] = None,
    min_S: Optional[float] = None,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Get SRS words for the current user."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return []

    # For now, return empty list - SRS implementation pending
    return []
