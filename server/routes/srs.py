from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from typing import Optional, List
from pydantic import BaseModel
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
def get_srs_stats(
    request: Request = None,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Get SRS statistics."""
    from server.models import Profile, Lexeme, ProfileTextRead, ReadingText

    # Get active profile from cookie
    active_profile_id = request.cookies.get("active_profile_id") if request else None
    profile = None

    if active_profile_id:
        try:
            profile = db.query(Profile).filter(
                Profile.id == int(active_profile_id),
                Profile.account_id == account.id
            ).first()
        except ValueError:
            profile = None

    # Fall back to first profile if no active profile specified
    if profile is None:
        profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    if not profile:
        return {
            "level_value": 0.0,
            "level_var": 1.0,
            "level_code": None,
            "total_words": 0,
            "words_by_level": {},
            "texts_read": 0,
        }

    # Word statistics
    lexemes = (
        db.query(Lexeme)
        .filter(
            Lexeme.account_id == account.id,
            Lexeme.profile_id == profile.id,
        )
        .all()
    )

    total_words = len(lexemes)

    words_by_level = {}
    for lex in lexemes:
        level = lex.level_code or "Unknown"
        words_by_level[level] = words_by_level.get(level, 0) + 1

    # Texts read
    texts_read = (
        db.query(ProfileTextRead)
        .filter(ProfileTextRead.profile_id == profile.id)
        .count()
    )

    # Words per text average
    words_per_text = round(total_words / texts_read, 1) if texts_read > 0 else 0

    return {
        "level_value": profile.level_value,
        "level_var": profile.level_var,
        "level_code": profile.level_code,
        "total_words": total_words,
        "words_by_level": words_by_level,
        "texts_read": texts_read,
        "words_per_text": words_per_text,
    }


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
    min_stability: Optional[float] = None,
    request: Request = None,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Get SRS words for current user."""
    from server.models import Lexeme

    # Get active profile from cookie
    active_profile_id = request.cookies.get("active_profile_id") if request else None
    profile = None

    if active_profile_id:
        try:
            profile = db.query(Profile).filter(
                Profile.id == int(active_profile_id),
                Profile.account_id == account.id
            ).first()
        except ValueError:
            profile = None

    # Fall back to first profile if no active profile specified
    if profile is None:
        profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    if not profile:
        return []

    query = db.query(Lexeme).filter(
        Lexeme.account_id == account.id,
        Lexeme.profile_id == profile.id,
    )

    if lang:
        query = query.filter(Lexeme.lang == lang)

    if min_stability is not None:
        query = query.filter(Lexeme.stability >= min_stability)

    lexemes = query.order_by(Lexeme.next_due_at.asc()).all()

    items = []
    for lex in lexemes:
        items.append(
            {
                "id": lex.id,
                "lemma": lex.lemma,
                "pos": lex.pos,
                "n": lex.exposures,
                "click_count": lex.clicks,
                "stability": lex.stability,
                "level_code": lex.level_code,
                "freq_rank": lex.frequency_rank,
                "next_due_at": lex.next_due_at.isoformat() if lex.next_due_at else None,
                "familiarity": lex.familiarity,
                "familiarity_variance": lex.familiarity_variance,
                "decay_rate": lex.decay_rate,
                "decay_variance": lex.decay_variance,
            }
        )

    return items


class DeleteWordsRequest(BaseModel):
    word_ids: List[int]


@router.delete("/srs/words")
def delete_words(
    request: DeleteWordsRequest,
    account: Account = Depends(get_current_account),
    db: Session = Depends(get_db),
):
    """Delete selected words from user's vocabulary."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return {"deleted": 0, "error": "No profile found"}

    # Delete words that belong to this account and profile
    deleted = (
        db.query(Lexeme)
        .filter(
            Lexeme.id.in_(request.word_ids),
            Lexeme.account_id == account.id,
            Lexeme.profile_id == profile.id,
        )
        .delete(synchronize_session=False)
    )

    db.commit()

    return {"deleted": deleted}
