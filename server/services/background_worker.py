"""Background worker for text pool management and generation."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Set

from sqlalchemy.orm import Session
from sqlalchemy import not_, exists
from sqlalchemy.sql import select, update

from server.db import SessionLocal
from server.services.recommendation import detect_pool_gaps
from server.services.content import generate_text_content, generate_translations

logger = logging.getLogger(__name__)

# Worker configuration
WORKER_INTERVAL_SECONDS = 60  # 1 minute (reduced from 5 minutes for faster generation)
MAX_GENERATION_CONCURRENCY = 3  # Max parallel text generations
MAX_RETRIES = 3  # Max retry attempts for failed translations
URGENT_POOL_THRESHOLD = 2  # If pool has <= this many texts, generate multiple per gap

# In-memory lock tracking for active generations
# Format: {profile_id: lock_acquired_timestamp}
_generating_locks: Dict[int, float] = {}
GENERATION_LOCK_TIMEOUT = 600  # 10 minutes - prevent stuck locks


def _acquire_generation_lock(profile_id: int) -> bool:
    """Try to acquire a generation lock for a profile.

    Returns True if lock was acquired, False if already locked.
    """
    global _generating_locks

    now = time.time()

    # Clean up expired locks
    expired = [
        pid for pid, timestamp in _generating_locks.items()
        if now - timestamp > GENERATION_LOCK_TIMEOUT
    ]
    for pid in expired:
        del _generating_locks[pid]
        logger.warning(f"Cleaned up expired generation lock for profile {pid}")

    # Check if profile is already being generated
    if profile_id in _generating_locks:
        logger.debug(f"Profile {profile_id} is already being generated, skipping")
        return False

    # Acquire lock
    _generating_locks[profile_id] = now
    logger.debug(f"Acquired generation lock for profile {profile_id}")
    return True


def _release_generation_lock(profile_id: int) -> None:
    """Release a generation lock for a profile."""
    global _generating_locks

    if profile_id in _generating_locks:
        del _generating_locks[profile_id]
        logger.debug(f"Released generation lock for profile {profile_id}")


def _get_locked_profile_ids() -> Set[int]:
    """Get set of profile IDs that are currently locked."""
    global _generating_locks

    now = time.time()

    # Clean up expired locks first
    expired = [
        pid for pid, timestamp in _generating_locks.items()
        if now - timestamp > GENERATION_LOCK_TIMEOUT
    ]
    for pid in expired:
        del _generating_locks[pid]

    return set(_generating_locks.keys())


async def background_worker():
    """Main background worker loop."""
    logger.info("Background worker started")

    while True:
        try:
            await maintenance_cycle()
        except Exception as e:
            logger.error(f"Error in maintenance cycle: {e}", exc_info=True)

        logger.debug(f"Worker sleeping for {WORKER_INTERVAL_SECONDS} seconds")
        await asyncio.sleep(WORKER_INTERVAL_SECONDS)


async def maintenance_cycle():
    """Single maintenance cycle."""
    logger.info("Starting maintenance cycle")

    with SessionLocal() as db:
        # Check if pool is already saturated before detecting gaps
        from server.models import ReadingText, Profile

        total_ready = (
            db.query(ReadingText)
            .filter(
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
            .count()
        )
        active_profiles = db.query(Profile).filter(Profile.preferences_updating == False).count()

        # If we have 20+ ready texts per active profile, don't generate more
        if active_profiles > 0 and total_ready >= (20 * active_profiles):
            logger.info(
                f"Pool saturated: {total_ready} ready texts for {active_profiles} profiles, skipping generation"
            )
        else:
            # Step 1: Detect pool gaps
            gaps = detect_pool_gaps(db, threshold=3.0)

            if gaps:
                logger.info(f"Found {len(gaps)} pool gaps")
                await fill_gaps(db, gaps)
            else:
                logger.debug("No pool gaps detected")

        # Step 2: Retry failed translations
        await retry_failed_translations(db)

        # Step 3: Clean up old texts
        cleanup_old_texts(db)

    logger.info("Maintenance cycle completed")


async def fill_gaps(db: Session, gaps: List) -> None:
    """Generate texts to fill detected pool gaps.

    Filters out profiles that are already being generated to prevent duplicates.
    When pool is very low (<= URGENT_POOL_THRESHOLD texts), generates 2-3 texts per gap.
    """
    if not gaps:
        return

    # Get currently locked profile IDs
    locked_profile_ids = _get_locked_profile_ids()

    # Filter out gaps for locked profiles
    available_gaps = [
        gap for gap in gaps
        if gap.profile_id not in locked_profile_ids
    ]

    if len(available_gaps) < len(gaps):
        logger.info(
            f"Filtered out {len(gaps) - len(available_gaps)} gaps due to active generation"
        )

    if not available_gaps:
        logger.debug("No available gaps to fill (all locked)")
        return

    # Check pool size for each gap to determine how many texts to generate
    generation_tasks = []

    for gap in available_gaps[:MAX_GENERATION_CONCURRENCY]:
        # Count existing ready texts for this language pair
        from server.models import ReadingText

        ready_count = (
            db.query(ReadingText)
            .filter(
                ReadingText.lang == gap.lang,
                ReadingText.target_lang == gap.target_lang,
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
            .count()
        )

        # Determine how many texts to generate for this gap
        if ready_count <= URGENT_POOL_THRESHOLD:
            # Pool is very low - generate 2-3 texts
            texts_to_generate = min(3, MAX_GENERATION_CONCURRENCY)
            logger.info(
                f"Urgent gap for {gap.lang}->{gap.target_lang}: "
                f"only {ready_count} texts, generating {texts_to_generate}"
            )
        else:
            # Normal generation - 1 text per gap
            texts_to_generate = 1
            logger.debug(
                f"Normal gap for {gap.lang}->{gap.target_lang}: "
                f"{ready_count} texts, generating 1"
            )

        # Add generation tasks
        for i in range(texts_to_generate):
            generation_tasks.append((gap, i))

    # Limit total concurrent generations
    semaphore = asyncio.Semaphore(MAX_GENERATION_CONCURRENCY)
    tasks = []

    for gap, index in generation_tasks[:MAX_GENERATION_CONCURRENCY]:
        task = _generate_text_for_gap(db, gap, semaphore, index)
        tasks.append(task)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Gap fill results: {success_count}/{len(results)} successful")


async def _generate_text_for_gap(
    db: Session, gap, semaphore: asyncio.Semaphore, index: int = 0
) -> None:
    """Generate a text for a specific pool gap."""
    profile_id = gap.profile_id

    # Try to acquire generation lock
    if not _acquire_generation_lock(profile_id):
        logger.debug(f"Could not acquire lock for profile {profile_id}, skipping")
        return

    try:
        async with semaphore:
            try:
                # Get profile for this gap
                from server.models import Profile

                profile = db.query(Profile).filter(Profile.id == profile_id).first()
                if not profile:
                    logger.warning(f"Profile {profile_id} not found for gap generation")
                    return

                logger.info(
                    f"Generating text for profile {profile.id}: {profile.lang}->{profile.target_lang}"
                )

                # Extract target words from gap if available
                target_words = gap.target_words if hasattr(gap, 'target_words') else None

                # Select topic with diversity consideration
                from server.services.recommendation import (
                    select_topic_for_profile,
                    select_diverse_topic,
                )

                preferred_topic = select_topic_for_profile(db, profile)
                selected_topic = select_diverse_topic(db, profile, preferred_topic)

                # Generate text content
                text_obj = await generate_text_content(
                    account_id=profile.account_id,
                    profile_id=profile.id,
                    lang=profile.lang,
                    target_lang=profile.target_lang,
                    profile=profile,
                    target_words=target_words,
                    topic=selected_topic,
                )

                if not text_obj:
                    logger.error(f"Failed to generate text for profile {profile.id}")
                    return

                # Generate translations
                success = await generate_translations(
                    text_id=text_obj.id,
                    lang=profile.lang,
                    target_lang=profile.target_lang,
                )

                if success:
                    logger.info(
                        f"Successfully generated text {text_obj.id} for profile {profile.id}"
                    )
                else:
                    logger.warning(f"Generated text {text_obj.id} but translations failed")

            except Exception as e:
                logger.error(f"Error generating text for gap: {e}", exc_info=True)
    finally:
        # Always release the lock when done
        _release_generation_lock(profile_id)


async def retry_failed_translations(db: Session) -> None:
    """Retry translations for texts with failed/attempts."""
    from server.models import ReadingText

    # Find texts with content but incomplete translations
    failed_texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.content.is_not(None),
            ReadingText.content != "",
            ReadingText.translation_attempts < MAX_RETRIES,
            ~ReadingText.words_complete | ~ReadingText.sentences_complete,
        )
        .limit(10)
        .all()
    )

    if not failed_texts:
        logger.debug("No failed translations to retry")
        return

    logger.info(f"Retrying translations for {len(failed_texts)} texts")

    tasks = []
    for text in failed_texts:
        task = _retry_translations(db, text)
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(1 for r in results if r)
    logger.info(f"Retry results: {success_count}/{len(results)} successful")


async def _retry_translations(db: Session, text) -> bool:
    """Retry translations for a single text."""
    try:
        text.translation_attempts += 1
        text.last_translation_attempt = datetime.now(timezone.utc)
        db.commit()

        success = await generate_translations(
            text_id=text.id,
            lang=text.lang,
            target_lang=text.target_lang,
        )

        if success:
            logger.info(f"Successfully retried translations for text {text.id}")
        else:
            logger.warning(f"Retry failed for text {text.id}")

        return success

    except Exception as e:
        logger.error(f"Error retrying translations for text {text.id}: {e}")
        return False


def cleanup_old_texts(db: Session) -> None:
    """Clean up old or low-quality texts."""
    from server.models import ReadingText, ProfileTextRead

    # Mark very old texts as hidden to keep pool fresh
    # 30 days old threshold
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

    # Find texts that have never been read
    # Subquery to check if any reading history exists
    from sqlalchemy import not_, exists

    unread_texts = (
        db.query(ReadingText.id)
        .outerjoin(ProfileTextRead, ProfileTextRead.text_id == ReadingText.id)
        .filter(
            ReadingText.created_at < cutoff_date,
            ProfileTextRead.id.is_(None),  # No reading history
            ReadingText.is_hidden == False,
        )
        .limit(50)
        .all()
    )

    if unread_texts:
        for text_id in unread_texts:
            text = db.query(ReadingText).filter(ReadingText.id == text_id[0]).first()
            if text:
                text.is_hidden = True
        db.commit()
        logger.info(f"Hidden {len(unread_texts)} old unread texts")

    # Remove very low-rated texts permanently
    # Binary scale: < -0.5 means 75%+ disliked
    low_rated_texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.rating_avg < -0.5,  # was 2.0 for 5-star scale
            ReadingText.rating_count >= 3,  # At least 3 ratings
        )
        .all()
    )

    if low_rated_texts:
        for text in low_rated_texts:
            db.delete(text)
        db.commit()
        logger.info(f"Deleted {len(low_rated_texts)} low-rated texts")


async def startup_generation(
    langs: Optional[List[str]] = None, texts_per_lang: int = 2
) -> None:
    """Generate initial texts on startup for specified languages."""
    if not langs:
        # Default to enabled languages if none specified
        from server.models import Language

        with SessionLocal() as db:
            enabled_langs = (
                db.query(Language)
                .filter(Language.is_enabled == True)
                .filter(Language.code != "en")
                .all()
            )
            langs = [lang.code for lang in enabled_langs]

    logger.info(f"Startup generation: {texts_per_lang} texts for {', '.join(langs)}")

    with SessionLocal() as db:
        for lang in langs:
            # Find first profile for this language
            from server.models import Profile, Account

            profiles = (
                db.query(Profile)
                .filter(
                    Profile.lang == lang,
                    Profile.lang != Profile.target_lang,  # Learning a new language
                )
                .limit(texts_per_lang)
                .all()
            )

            if not profiles:
                logger.warning(f"No profiles found for language {lang}")
                continue

            for profile in profiles:
                try:
                    logger.info(f"Startup generation for profile {profile.id}")

                    # Get urgent lexemes for target words
                    from server.services.recommendation import (
                        get_urgent_lexemes_for_profile,
                        select_topic_for_profile,
                        select_diverse_topic,
                    )
                    urgent_lexemes = get_urgent_lexemes_for_profile(db, profile, limit=20)
                    target_words = {lex.lemma for lex, _ in urgent_lexemes[:5]} if urgent_lexemes else None

                    if target_words:
                        logger.info(f"Startup generation with target words: {target_words}")

                    # Select topic with diversity consideration
                    preferred_topic = select_topic_for_profile(db, profile)
                    selected_topic = select_diverse_topic(db, profile, preferred_topic)
                    logger.info(f"Startup generation with topic: {selected_topic}")

                    text_obj = await generate_text_content(
                        account_id=profile.account_id,
                        profile_id=profile.id,
                        lang=profile.lang,
                        target_lang=profile.target_lang,
                        profile=profile,
                        target_words=target_words,
                        topic=selected_topic,
                    )

                    if text_obj:
                        await generate_translations(
                            text_id=text_obj.id,
                            lang=profile.lang,
                            target_lang=profile.target_lang,
                        )
                        logger.info(f"Startup generated text {text_obj.id}")
                    else:
                        logger.warning(
                            f"Startup generation failed for profile {profile.id}"
                        )

                except Exception as e:
                    logger.error(f"Error in startup generation: {e}", exc_info=True)
