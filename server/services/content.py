"""
Content services consolidating reading generation, LLM calls, and text processing.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    parse_csv_word_translations,
    parse_csv_translation,
    parse_csv_title_translation,
    split_sentences,
    compute_word_spans,
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

        raw_content = response[0] if response else ""

        # Parse CSV response: title|text
        title = None
        content = ""

        lines = raw_content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if "|" in line and not line.startswith("#"):
                parts = line.split("|")
                if len(parts) >= 2:
                    header = parts[0].strip().lower()
                    if header == "title":
                        continue  # Skip header
                    # Check if this looks like title|text format
                    first_part = parts[0].strip()
                    if not title and len(first_part) < 100:  # Likely a title
                        title = first_part
                        content = parts[1].strip() if len(parts) > 1 else ""
                    elif content and len(parts) > 1:  # Append to content
                        content += "\n" + parts[1].strip()
                    elif not content:  # Just text content
                        content = parts[0].strip()

        # Fallback: treat entire response as content
        if not content:
            content = extract_text_from_llm_response(raw_content)

        if not content:
            logger.error(f"Failed to extract content from LLM response: {response}")
            return None

        with SessionLocal() as db:
            rt = ReadingText(
                generated_for_account_id=account_id,
                lang=lang,
                target_lang=target_lang,
                content=content,
                title=title,
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

            # Fallback title extraction if not in CSV
            if not title:
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

        word_data = parse_csv_word_translations(response[0] if response else "")

        if not word_data:
            logger.warning(f"No word data parsed from LLM response")
            return True

        # Compute spans for all words
        all_spans = compute_word_spans(content_str, word_data)

        for word, spans in zip(word_data, all_spans):
            if not spans:
                continue

            surface = word["surface"]
            lemma = word.get("lemma") or surface
            pos = word.get("pos") or "UNKNOWN"
            pinyin = word.get("pinyin")
            translation = word.get("translation")

            # For non-continuous words, use the first span for span_start/end
            # Store full spans in grammar field for frontend use
            span_start, span_end = spans[0]

            gloss = ReadingWordGloss(
                text_id=rt.id,
                lang=lang,
                target_lang=target_lang,
                surface=surface,
                lemma=lemma,
                pos=pos,
                pinyin=pinyin,
                translation=translation,
                grammar={"spans": spans} if len(spans) > 1 else {},
                span_start=span_start,
                span_end=span_end,
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

        translations = parse_csv_translation(response[0] if response else "")

        # Create mapping from source text to translation
        trans_map = {t["source"]: t["translation"] for t in translations}

        for i, (start, end, seg) in enumerate(sentences):
            trans_text = trans_map.get(seg, seg)

            trans = ReadingTextTranslation(
                text_id=rt.id,
                target_lang=target_lang,
                unit="sentence",
                segment_index=i,
                source_text=seg,
                translated_text=trans_text,
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
from server.services.learning import get_ci_target as _get_ci_target
from server.llm.client import _pick_openrouter_model


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
    """Extract title from generated text content (CSV format)."""
    # Try CSV format first: title|text
    lines = content.strip().split("\n")
    for line in lines:
        if "|" in line and not line.startswith("#"):
            parts = line.split("|")
            if len(parts) >= 2 and parts[0].strip().lower() == "title":
                # Header row, get next line
                continue
            if len(parts) >= 2 and parts[0].strip():
                # This is the title line
                return parts[0].strip()

    # Fallback to simple extraction
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
