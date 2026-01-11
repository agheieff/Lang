"""SRS (Spaced Repetition System) service for tracking word familiarity.

Uses Bayesian-style updating with:
- familiarity: How well user knows the word (0-1)
- familiarity_variance: Uncertainty in familiarity estimate (0-0.25)
- decay_rate: How quickly familiarity decays per day (0-1)
- decay_variance: Uncertainty in decay rate (0-0.25)
"""

import logging
from datetime import datetime, timezone, timedelta
from math import sqrt, exp
from typing import Optional

from sqlalchemy.orm import Session

from server.models import Lexeme

logger = logging.getLogger(__name__)


# Constants
DEFAULT_FAMILIARITY = 0.5  # Start with 50% familiarity
DEFAULT_FAMILIARITY_VARIANCE = 0.25  # High initial uncertainty
DEFAULT_DECAY_RATE = 0.1  # 10% decay per day
DEFAULT_DECAY_VARIANCE = 0.25  # High initial uncertainty

LEARNING_RATE_CLICK = 0.3  # How much a click affects familiarity
LEARNING_RATE_NO_CLICK = 0.1  # How much a no-click affects familiarity
VARIANCE_REDUCTION = 0.05  # How much variance decreases per interaction


def initialize_lexeme_srs(lexeme: Lexeme) -> None:
    """Initialize SRS values for a new lexeme based on existing click/exposure data."""
    # Calculate initial familiarity from historical data
    if lexeme.exposures > 0:
        click_ratio = lexeme.clicks / lexeme.exposures
        lexeme.familiarity = max(0.0, min(1.0, 1.0 - click_ratio))

        # Variance decreases with more data points
        data_confidence = min(1.0, sqrt(lexeme.exposures) * 0.1)
        lexeme.familiarity_variance = max(0.05, 0.25 * (1.0 - data_confidence))
    else:
        lexeme.familiarity = DEFAULT_FAMILIARITY
        lexeme.familiarity_variance = DEFAULT_FAMILIARITY_VARIANCE

    # Set default decay values
    lexeme.decay_rate = DEFAULT_DECAY_RATE
    lexeme.decay_variance = DEFAULT_DECAY_VARIANCE

    logger.debug(
        f"Initialized SRS for lexeme {lexeme.id}: "
        f"familiarity={lexeme.familiarity:.2f}±{lexeme.familiarity_variance:.2f}"
    )


def apply_time_decay(lexeme: Lexeme, current_time: Optional[datetime] = None) -> None:
    """Apply exponential decay to familiarity based on time since last seen."""
    if not current_time:
        current_time = datetime.now(timezone.utc)

    if not lexeme.last_seen_at:
        return

    # Ensure both datetimes are timezone-aware
    last_seen = lexeme.last_seen_at
    if last_seen.tzinfo is None:
        # If naive, assume it's UTC
        last_seen = last_seen.replace(tzinfo=timezone.utc)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    # Calculate days since last exposure
    days_since_seen = (current_time - last_seen).total_seconds() / 86400.0

    if days_since_seen <= 0:
        return

    # Apply exponential decay: familiarity *= exp(-decay_rate * days)
    if lexeme.decay_rate and lexeme.familiarity is not None:
        decay_factor = exp(-lexeme.decay_rate * days_since_seen)
        lexeme.familiarity = max(0.0, lexeme.familiarity * decay_factor)

        logger.debug(
            f"Applied {days_since_seen:.1f}d decay to lexeme {lexeme.id}: "
            f"{decay_factor:.3f}x, new familiarity={lexeme.familiarity:.2f}"
        )


def update_lexeme_from_click(
    lexeme: Lexeme,
    db: Session,
    click_time: Optional[datetime] = None,
) -> None:
    """Update SRS values when user clicks on a word (negative signal).

    Clicking means user doesn't know the word well:
    - Decrease familiarity
    - Increase variance (we're less certain now)
    - May increase decay_rate (word is harder than expected)
    """
    if not click_time:
        click_time = datetime.now(timezone.utc)

    # Apply time decay first
    apply_time_decay(lexeme, click_time)

    # Decrease familiarity (they don't know it)
    if lexeme.familiarity is not None:
        old_fam = lexeme.familiarity
        lexeme.familiarity = max(0.0, lexeme.familiarity - LEARNING_RATE_CLICK)

        # Increase variance (we're less certain after unexpected click)
        if lexeme.familiarity_variance is not None:
            lexeme.familiarity_variance = min(0.25, lexeme.familiarity_variance + VARIANCE_REDUCTION)

        logger.debug(
            f"Click on lexeme {lexeme.id}: {old_fam:.2f}→{lexeme.familiarity:.2f}, "
            f"var={lexeme.familiarity_variance:.2f}"
        )

    # Update interaction tracking
    lexeme.clicks = (lexeme.clicks or 0) + 1
    lexeme.last_clicked_at = click_time
    lexeme.last_seen_at = click_time

    db.flush()


def update_lexeme_from_exposure(
    lexeme: Lexeme,
    db: Session,
    exposure_time: Optional[datetime] = None,
) -> None:
    """Update SRS values when user sees word but doesn't click (positive signal).

    Not clicking means user knows the word:
    - Increase familiarity
    - Decrease variance (we're more confident)
    - May decrease decay_rate (word is easier than expected)
    """
    if not exposure_time:
        exposure_time = datetime.now(timezone.utc)

    # Apply time decay first
    apply_time_decay(lexeme, exposure_time)

    # Increase familiarity (they know it)
    if lexeme.familiarity is not None:
        old_fam = lexeme.familiarity
        lexeme.familiarity = min(1.0, lexeme.familiarity + LEARNING_RATE_NO_CLICK)

        # Decrease variance (we're more confident after successful non-click)
        if lexeme.familiarity_variance is not None:
            lexeme.familiarity_variance = max(0.05, lexeme.familiarity_variance - VARIANCE_REDUCTION * 0.5)

        logger.debug(
            f"No-click on lexeme {lexeme.id}: {old_fam:.2f}→{lexeme.familiarity:.2f}, "
            f"var={lexeme.familiarity_variance:.2f}"
        )

    # Update interaction tracking
    lexeme.exposures = (lexeme.exposures or 0) + 1
    lexeme.last_seen_at = exposure_time

    db.flush()


def calculate_next_review(lexeme: Lexeme) -> Optional[datetime]:
    """Calculate when this word should be reviewed based on SRS state.

    Words with low familiarity should be shown more often.
    Words with high familiarity and low variance are shown less often.
    """
    if lexeme.familiarity is None or lexeme.decay_rate is None:
        return None

    # Base interval: 1 day
    # Adjust by familiarity: higher familiarity = longer interval
    # Adjust by variance: higher variance = shorter interval (review uncertain words)

    familiarity_factor = max(0.1, lexeme.familiarity)  # Avoid 0
    variance_factor = 1.0 + (lexeme.familiarity_variance or 0.0)

    # Days until next review
    days = 1.0 * (familiarity_factor / 0.5) * variance_factor

    # Cap at reasonable range
    days = max(0.1, min(days, 30.0))

    next_review = datetime.now(timezone.utc) + timedelta(days=days)
    return next_review


def batch_update_lexemes_from_text_state(
    db: Session,
    account_id: int,
    profile_id: int,
    text_id: int,
    words: list[dict],
    current_time: Optional[datetime] = None,
) -> dict:
    """Update all lexemes from a completed text state.

    Args:
        db: Database session
        account_id: User account ID
        profile_id: User profile ID
        text_id: Text ID
        words: List of word objects with clicks arrays
        current_time: Timestamp for updates

    Returns:
        Summary dict with updated/clicked/total counts
    """
    if not current_time:
        current_time = datetime.now(timezone.utc)

    updated = 0
    clicked = 0
    total = 0

    for word_data in words:
        surface = word_data.get("surface", "")
        lemma = word_data.get("lemma") or surface
        pos = word_data.get("pos", "UNKNOWN")
        lang = word_data.get("lang", "zh-CN")

        # Find or create lexeme
        lexeme = (
            db.query(Lexeme)
            .filter(
                Lexeme.account_id == account_id,
                Lexeme.profile_id == profile_id,
                Lexeme.lang == lang,
                Lexeme.lemma == lemma,
                Lexeme.pos == pos,
            )
            .first()
        )

        if not lexeme:
            logger.warning(f"Lexeme not found for {lemma}/{pos}, skipping SRS update")
            continue

        total += 1

        # Initialize SRS values if needed
        if lexeme.familiarity is None:
            initialize_lexeme_srs(lexeme)

        # Apply time decay ONCE per word (not per click)
        apply_time_decay(lexeme, current_time)

        # Process clicks
        click_count = len(word_data.get("clicks", []))

        if click_count > 0:
            # User clicked - apply negative signal scaled by click count
            old_fam = lexeme.familiarity

            # Scale the learning effect by number of clicks
            # Multiple clicks on same word = stronger negative signal
            learning_effect = LEARNING_RATE_CLICK * min(click_count, 3)  # Cap at 3x

            if lexeme.familiarity is not None:
                lexeme.familiarity = max(0.0, lexeme.familiarity - learning_effect)

                # Increase variance (we're less certain after unexpected clicks)
                if lexeme.familiarity_variance is not None:
                    lexeme.familiarity_variance = min(0.25, lexeme.familiarity_variance + VARIANCE_REDUCTION * click_count)

                logger.debug(
                    f"Click ({click_count}x) on lexeme {lexeme.id}: {old_fam:.2f}→{lexeme.familiarity:.2f}, "
                    f"var={lexeme.familiarity_variance:.2f}"
                )

            # Update interaction tracking
            lexeme.clicks = (lexeme.clicks or 0) + click_count
            lexeme.last_clicked_at = current_time
            clicked += 1
        else:
            # No clicks - apply positive signal (they know it)
            old_fam = lexeme.familiarity

            if lexeme.familiarity is not None:
                lexeme.familiarity = min(1.0, lexeme.familiarity + LEARNING_RATE_NO_CLICK)

                # Decrease variance (we're more confident after successful non-click)
                if lexeme.familiarity_variance is not None:
                    lexeme.familiarity_variance = max(0.05, lexeme.familiarity_variance - VARIANCE_REDUCTION * 0.5)

                logger.debug(
                    f"No-click on lexeme {lexeme.id}: {old_fam:.2f}→{lexeme.familiarity:.2f}, "
                    f"var={lexeme.familiarity_variance:.2f}"
                )

        # Always update exposure tracking
        lexeme.exposures = (lexeme.exposures or 0) + 1
        lexeme.last_seen_at = current_time

        # Calculate next review time
        lexeme.next_due_at = calculate_next_review(lexeme)

        updated += 1

    db.commit()

    logger.info(
        f"SRS batch update: account={account_id}, profile={profile_id}, "
        f"text={text_id}, updated={updated}, clicked={clicked}, total={total}"
    )

    return {
        "updated": updated,
        "clicked": clicked,
        "total": total,
    }
