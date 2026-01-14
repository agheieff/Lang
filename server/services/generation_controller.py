"""Hybrid generation controller with continuous desire scoring."""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session

from server.models import Profile, ReadingText
from server.config import (
    POOL_MIN_SIZE,
    POOL_MAX_SIZE,
    GEN_QUALITY_THRESHOLD,
    GEN_DESIRE_URGENT,
    GEN_DESIRE_NORMAL,
)

logger = logging.getLogger(__name__)


def calculate_generation_desire(
    db: Session,
    profile: Profile,
) -> float:
    """Calculate continuous desire to generate (0-1, higher = more need)."""
    from server.services.recommendation import get_unread_texts_for_profile

    # Get pool size for this language pair
    ready_count = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == profile.lang,
            ReadingText.target_lang == profile.target_lang,
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
            ReadingText.is_hidden == False,
        )
        .count()
    )

    # Get scored texts to assess quality
    scored_texts = get_unread_texts_for_profile(db, profile, limit=50)

    if not scored_texts:
        # No texts at all - maximum desire
        logger.info(f"GenDesire: No texts for profile {profile.id}, desire=1.0")
        return 1.0

    # Extract scores
    scores = [score for _, score in scored_texts[:10]]  # Top 10
    best_score = scores[0]
    score_variance = (max(scores) - min(scores)) if len(scores) > 1 else 0.0

    # Base desire from pool size (non-linear decay)
    if ready_count <= POOL_MIN_SIZE:
        base_desire = 1.0
    elif ready_count >= POOL_MAX_SIZE:
        base_desire = 0.0
    else:
        # Linear interpolation between MIN and MAX with power curve
        ratio = (ready_count - POOL_MIN_SIZE) / (POOL_MAX_SIZE - POOL_MIN_SIZE)
        base_desire = 1.0 - ratio ** 1.5  # Curve for diminishing returns

    # Quality factor (penalty for poor matches)
    quality_factor = max(0.0, (best_score - GEN_QUALITY_THRESHOLD) / 5.0)

    # Variance bonus (reward diversity)
    variance_bonus = min(0.2, score_variance / 10.0)

    # Combined desire
    desire = base_desire + quality_factor - variance_bonus
    desire = max(0.0, min(1.0, desire))  # Clamp to [0, 1]

    logger.info(
        f"GenDesire: profile={profile.id}, pool={ready_count}, "
        f"best_score={best_score:.2f}, variance={score_variance:.2f}, desire={desire:.2f}"
    )

    return desire


def should_generate_for_profile(
    db: Session,
    profile: Profile,
) -> Tuple[bool, int, str]:
    """Determine if generation needed and how many texts.

    Returns:
        (should_generate, count, reason)
    """
    if profile.preferences_updating:
        return False, 0, "prefs_updating"

    desire = calculate_generation_desire(db, profile)

    if desire > GEN_DESIRE_URGENT:
        return True, 3, f"urgent_desire_{desire:.2f}"
    elif desire > GEN_DESIRE_NORMAL:
        return True, 1, f"normal_desire_{desire:.2f}"
    else:
        return False, 0, f"saturated_desire_{desire:.2f}"


def mark_profile_queue_dirty(
    db: Session,
    profile_id: int,
    reason: str = "manual",
) -> None:
    """Mark a profile's queue as needing rebuild."""
    profile = db.query(Profile).filter(Profile.id == profile_id).first()
    if profile:
        profile.queue_dirty = True
        profile.queue_dirty_reason = reason
        db.commit()
        logger.info(f"Marked queue dirty for profile {profile_id}: {reason}")


def rebuild_dirty_queues(
    db: Session,
    limit: int = 20,
) -> int:
    """Rebuild queues for profiles marked as dirty.

    Returns number of queues rebuilt.
    """
    from server.services.recommendation import update_text_queue
    from server.config import QUEUE_DIRTY_REBUILD_LIMIT

    dirty_profiles = (
        db.query(Profile)
        .filter(
            Profile.queue_dirty == True,
            Profile.preferences_updating == False,
        )
        .limit(limit or QUEUE_DIRTY_REBUILD_LIMIT)
        .all()
    )

    if not dirty_profiles:
        return 0

    logger.info(f"Rebuilding {len(dirty_profiles)} dirty queues")

    rebuilt = 0
    for profile in dirty_profiles:
        try:
            update_text_queue(db, profile)
            profile.queue_dirty = False
            profile.queue_built_at = datetime.now(timezone.utc)
            profile.queue_dirty_reason = None
            rebuilt += 1
        except Exception as e:
            logger.error(f"Error rebuilding queue for profile {profile.id}: {e}")

    db.commit()
    return rebuilt
