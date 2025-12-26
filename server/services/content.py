"""
Content services consolidating reading generation, LLM calls, and text processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

from sqlalchemy.orm import Session
from server.db import SessionLocal
from server.llm.client import chat_complete_with_raw
from server.llm.prompts import (
    PromptSpec,
    build_reading_prompt,
    build_translation_contexts,
    build_word_translation_prompt,
)
from server.utils.nlp import (
    extract_text_from_llm_response,
    extract_word_translations,
    extract_structured_translation,
    split_sentences,
    compute_spans,
)
from server.models import (
    Profile,
    ProfileTextRead,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    GenerationLog,
    TranslationLog,
)

logger = logging.getLogger(__name__)


# Text Generation Functions
async def generate_text_content(
    account_id: int,
    profile_id: int,
    lang: str,
    target_lang: str,
    profile: Profile,
) -> Optional[ReadingText]:
    """Generate text content using LLM."""
    try:
        ci_target = _get_ci_target(profile.level_value, profile.level_var)
        words = _get_word_list_from_profile(profile)
        level_hint = _compose_level_hint(profile.level_value, profile.level_code)

        spec = PromptSpec(
            lang=lang,
            unit="text",
            approx_len=profile.text_length or 200,
            user_level_hint=level_hint,
            include_words=None,
            ci_target=ci_target,
        )

        prompt = build_reading_prompt(spec)

        model = _pick_openrouter_model()

        logger.info(f"Generating text for account {account_id}, profile {profile_id}")

        response = chat_complete_with_raw(
            messages=prompt,
            model=model,
        )

        content = extract_text_from_llm_response(response[0] if response else "")

        if not content:
            logger.error(f"Failed to extract content from LLM response: {response}")
            return None

        with SessionLocal() as db:
            rt = ReadingText(
                generated_for_account_id=account_id,
                lang=lang,
                target_lang=target_lang,
                content=content,
                source="llm",
                ci_target=ci_target,
                request_sent_at=datetime.now(timezone.utc),
                generated_at=datetime.now(timezone.utc),
                prompt_words={},
                prompt_level_hint=level_hint,
                translation_attempts=0,
                words_complete=False,
                sentences_complete=False,
            )

            db.add(rt)
            db.commit()
            db.refresh(rt)

            # Extract title
            title = _extract_title(content, lang)
            if title:
                rt.title = title
                db.commit()

            logger.info(f"Generated text content {rt.id} for account {account_id}")
            return rt

    except Exception as e:
        logger.error(f"Error generating text content: {e}")
        return None


async def generate_translations(
    text_id: int,
    lang: str,
    target_lang: str,
) -> bool:
    """Generate word and sentence translations for a text."""
    try:
        with SessionLocal() as db:
            rt = db.get(ReadingText, text_id)
            if not rt or not rt.content:
                return False

            # Generate word translations
            word_success = await _generate_word_translations(db, rt, lang, target_lang)

            # Generate sentence translations
            sentence_success = await _generate_sentence_translations(
                db, rt, lang, target_lang
            )

            if word_success and sentence_success:
                rt.words_complete = True
                rt.sentences_complete = True
                db.commit()

                logger.info(f"Completed translations for text {text_id}")
                return True
            else:
                logger.error(f"Failed translations for text {text_id}")
                return False

    except Exception as e:
        logger.error(f"Error generating translations for text {text_id}: {e}")
        return False


async def _generate_word_translations(
    db: Session,
    rt: ReadingText,
    lang: str,
    target_lang: str,
) -> bool:
    """Generate word translations."""
    try:
        content_str = rt.content or ""
        if not content_str:
            return True

        spans = compute_spans(content_str, [{"word": w} for w in content_str.split()])

        if not spans:
            return True

        batch_size = 10
        for i in range(0, len(spans), batch_size):
            batch = spans[i : i + batch_size]

            word_list = []
            for j, span in enumerate(batch):
                if span and len(span) == 2:
                    word_list.append(
                        {
                            "surface": content_str[span[0] : span[1]],
                            "span_start": span[0],
                            "span_end": span[1],
                        }
                    )

            if not word_list:
                continue

            prompt = build_word_translation_prompt(
                source_lang=lang,
                target_lang=target_lang,
                text=content_str,
            )

            model = _pick_openrouter_model()
            response = chat_complete_with_raw(
                messages=prompt,
                model=model,
            )

            translations = extract_word_translations(response[0] if response else "")

            for word_data in word_list:
                surface = word_data["surface"]
                start = word_data["span_start"]
                end = word_data["span_end"]

                translation = None
                for trans in translations:
                    if trans.get("surface") == surface:
                        translation = trans.get("translation")
                        break

                gloss = ReadingWordGloss(
                    text_id=rt.id,
                    lang=lang,
                    target_lang=target_lang,
                    surface=surface,
                    lemma=surface,
                    pos="UNKNOWN",
                    pinyin=None,
                    translation=translation,
                    grammar={},
                    span_start=start,
                    span_end=end,
                )

                db.add(gloss)

            db.commit()

        return True

    except Exception as e:
        logger.error(f"Error generating word translations: {e}")
        return False


async def _generate_sentence_translations(
    db: Session,
    rt: ReadingText,
    lang: str,
    target_lang: str,
) -> bool:
    """Generate sentence translations."""
    try:
        content_str = rt.content or ""
        if not content_str:
            return True

        sentences = split_sentences(content_str, lang)

        if not sentences:
            return True

        contexts = build_translation_contexts(
            reading_messages=[],
            source_lang=lang,
            target_lang=target_lang,
            text=content_str,
        )

        batch_size = 5
        word_translations = contexts.get("words", [])

        model = _pick_openrouter_model()
        response = chat_complete_with_raw(
            messages=word_translations,
            model=model,
        )

        translations = extract_structured_translation(response[0] if response else "")

        for i, (start, end, seg) in enumerate(sentences):
            trans = ReadingTextTranslation(
                text_id=rt.id,
                target_lang=target_lang,
                unit="sentence",
                segment_index=i,
                span_start=start,
                span_end=end,
                source_text=seg,
                translated_text=str(translations.get(str(i), seg))
                if isinstance(translations, dict)
                else seg,
                provider="llm",
                model=model,
            )
            db.add(trans)

        db.commit()

        return True

    except Exception as e:
        logger.error(f"Error generating sentence translations: {e}")
        return False


# Pool Selection Functions
def get_available_texts(
    db: Session,
    profile: Profile,
    limit: int = 10,
) -> List[Tuple[ReadingText, float]]:
    """Get available texts for this profile, scored by match."""
    # Get texts this profile hasn't read
    read_texts = {
        ptr.text_id
        for ptr in db.query(ProfileTextRead)
        .filter(ProfileTextRead.profile_id == profile.id)
        .all()
    }

    available_texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == profile.lang,
            ReadingText.target_lang == profile.target_lang,
            ReadingText.content.is_not(None),
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
            ~ReadingText.id.in_(read_texts),
        )
        .all()
    )

    scored_texts = []
    for text in available_texts:
        score = _calculate_match_score(profile, text)
        scored_texts.append((text, score))

    scored_texts.sort(key=lambda x: x[1])

    return scored_texts[:limit]


def select_next_text(
    db: Session,
    profile: Profile,
) -> Optional[ReadingText]:
    """Select the next text for the profile."""
    scored_texts = get_available_texts(db, profile, limit=1)

    if scored_texts:
        return scored_texts[0][0]

    return None

    return None


async def ensure_next_text_ready(
    account_id: int,
    profile_id: int,
    profile: Profile,
) -> Optional[ReadingText]:
    """Ensure there's a ready text for the profile."""
    with SessionLocal() as db:
        next_text = select_next_text(db, profile)

        if next_text:
            profile.current_text_id = next_text.id
            db.commit()

            return next_text

        # No ready text, start generation
        rt = await generate_text_content(
            account_id,
            profile_id,
            profile.lang,
            profile.target_lang,
            profile,
        )

        if rt:
            success = await generate_translations(
                rt.id,
                profile.lang,
                profile.target_lang,
            )

            if success:
                profile.current_text_id = rt.id
                db.commit()
                return rt

        return None


# Helper Functions
def _get_ci_target(level_value: float, level_var: float = 1.0) -> float:
    """Get target comprehension index based on user level."""
    if level_value < 2.0:
        return 0.95
    elif level_value < 4.0:
        return 0.92
    elif level_value < 6.0:
        return 0.88
    elif level_value < 8.0:
        return 0.86
    else:
        return 0.85


def _pick_openrouter_model(requested: Optional[str] = None) -> str:
    """Prefer non-thinking model variants by default."""
    if requested:
        return requested
    m = os.getenv("OPENROUTER_MODEL_NONREASONING")
    if m:
        return m
    m2 = os.getenv("OPENROUTER_MODEL")
    return m2 or "x-ai/grok-4.1-fast:free"


def _compose_level_hint(
    level_value: float, level_code: Optional[str] = None
) -> Optional[str]:
    """Compose level hint string for text generation."""
    if level_code:
        return f"CEFR {level_code} (approx. level {level_value:.1f})"
    return f"Level {level_value:.1f}"


def _get_word_list_from_profile(profile: Profile) -> List[Dict]:
    """Get word list for text generation from profile."""
    # TODO: Implement vocabulary selection based on profile
    return []


def _extract_title(content: str, lang: str) -> Optional[str]:
    """Extract title from generated text content."""
    # Simple title extraction - first sentence or first 50 chars
    sentences = content.split(".")
    if sentences and len(sentences[0].strip()) > 10:
        return sentences[0].strip()
    return content[:50].strip() + ("..." if len(content) > 50 else "")


def _calculate_match_score(profile: Profile, text: ReadingText) -> float:
    """Calculate how well text matches profile preferences."""
    if text.ci_target and profile.ci_preference:
        ci_diff = abs(text.ci_target - profile.ci_preference)
        return ci_diff

    return 0.5  # Default middle score


def track_generation_usage(
    db: Session,
    account_id: int,
    text_id: int,
    generation_type: str,
    tokens_used: Optional[int] = None,
    model: Optional[str] = None,
    success: bool = True,
) -> None:
    """Track usage from text generation."""
    try:
        log = GenerationLog(
            account_id=account_id,
            profile_id=None,
            text_id=text_id,
            model=model or "unknown",
            prompt={},
            words={},
            level_hint=None,
            approx_len=None,
            unit=None,
            created_at=datetime.now(timezone.utc),
        )
        db.add(log)
        db.commit()

    except Exception as e:
        logger.error(f"Error tracking generation usage: {e}")
        db.rollback()


def track_translation_usage(
    db: Session,
    account_id: int,
    text_id: int,
    translation_type: str,
    tokens_used: Optional[int] = None,
    model: Optional[str] = None,
    success: bool = True,
) -> None:
    """Track usage from translation requests."""
    try:
        log = TranslationLog(
            account_id=account_id,
            text_id=text_id,
            unit=translation_type,
            target_lang=None,
            provider=None,
            model=model or "unknown",
            prompt={},
            segments={},
            response=None,
            created_at=datetime.now(timezone.utc),
        )
        db.add(log)
        db.commit()

    except Exception as e:
        logger.error(f"Error tracking translation usage: {e}")
        db.rollback()
