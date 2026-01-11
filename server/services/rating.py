"""Rating service for managing text ratings."""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from server.models import TextRating, ReadingText

logger = logging.getLogger(__name__)


def get_user_rating(db: Session, account_id: int, text_id: int) -> Optional[int]:
    """Get user's rating (-1, 1, or None)."""
    rating_obj = db.query(TextRating).filter(
        TextRating.account_id == account_id,
        TextRating.text_id == text_id
    ).first()
    return rating_obj.rating if rating_obj else None


def save_rating(
    db: Session,
    account_id: int,
    text_id: int,
    rating: int,
    profile_id: Optional[int] = None,
) -> TextRating:
    """Save/update user rating and recalculate aggregates."""
    if rating not in (-1, 1):
        raise ValueError(f"Rating must be -1 or 1, got {rating}")

    existing = db.query(TextRating).filter(
        TextRating.account_id == account_id,
        TextRating.text_id == text_id
    ).first()

    if existing:
        existing.rating = rating
        existing.updated_at = datetime.now(timezone.utc)
        if profile_id:
            existing.profile_id = profile_id
    else:
        existing = TextRating(
            account_id=account_id,
            text_id=text_id,
            rating=rating,
            profile_id=profile_id
        )
        db.add(existing)

    db.flush()
    _update_text_aggregates(db, text_id)
    return existing


def _update_text_aggregates(db: Session, text_id: int) -> None:
    """Recalculate rating_avg and rating_count."""
    ratings = db.query(TextRating).filter(TextRating.text_id == text_id).all()

    if ratings:
        text = db.query(ReadingText).filter(ReadingText.id == text_id).first()
        if text:
            text.rating_count = len(ratings)
            text.rating_avg = sum(r.rating for r in ratings) / text.rating_count
    else:
        # No ratings - clear aggregates
        text = db.query(ReadingText).filter(ReadingText.id == text_id).first()
        if text:
            text.rating_count = 0
            text.rating_avg = None
