"""Background worker for text pool management and generation."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import not_, exists

from server.db import SessionLocal
from server.services.recommendation import detect_pool_gaps
from server.services.content import generate_text_content, generate_translations

logger = logging.getLogger(__name__)

# Worker configuration
WORKER_INTERVAL_SECONDS = 300  # 5 minutes
MAX_GENERATION_CONCURRENCY = 3  # Max parallel text generations
MAX_RETRIES = 3  # Max retry attempts for failed translations


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
    """Generate texts to fill detected pool gaps."""
    if not gaps:
        return

    # Limit concurrent generations
    semaphore = asyncio.Semaphore(MAX_GENERATION_CONCURRENCY)
    tasks = []

    for gap in gaps[:MAX_GENERATION_CONCURRENCY]:
        task = _generate_text_for_gap(db, gap, semaphore)
        tasks.append(task)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Gap fill results: {success_count}/{len(results)} successful")


async def _generate_text_for_gap(
    db: Session, gap, semaphore: asyncio.Semaphore
) -> None:
    """Generate a text for a specific pool gap."""
    async with semaphore:
        try:
            # Get profile for this gap
            from server.models import Profile

            profile = db.query(Profile).filter(Profile.id == gap.profile_id).first()
            if not profile:
                logger.warning(f"Profile {gap.profile_id} not found for gap generation")
                return

            logger.info(
                f"Generating text for profile {profile.id}: {profile.lang}->{profile.target_lang}"
            )

            # Extract target words from gap if available
            target_words = gap.target_words if hasattr(gap, 'target_words') else None

            # Generate text content
            text_obj = await generate_text_content(
                account_id=profile.account_id,
                profile_id=profile.id,
                lang=profile.lang,
                target_lang=profile.target_lang,
                profile=profile,
                target_words=target_words,
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
    low_rated_texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.rating_avg < 2.0,
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
                    from server.services.recommendation import get_urgent_lexemes_for_profile
                    urgent_lexemes = get_urgent_lexemes_for_profile(db, profile, limit=20)
                    target_words = {lex.lemma for lex, _ in urgent_lexemes[:5]} if urgent_lexemes else None

                    if target_words:
                        logger.info(f"Startup generation with target words: {target_words}")

                    text_obj = await generate_text_content(
                        account_id=profile.account_id,
                        profile_id=profile.id,
                        lang=profile.lang,
                        target_lang=profile.target_lang,
                        profile=profile,
                        target_words=target_words,
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
