from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from .gen_queue import ensure_text_available
from ..llm.prompts import build_reading_prompt
from .llm_common import build_reading_prompt_spec


def should_generate_new_text(db: Session, account_id: int, lang: str) -> bool:
    """Return True when no unopened texts exist for (account, lang)."""
    unopened = (
        db.query(ReadingText)
        .filter(ReadingText.account_id == account_id, ReadingText.lang == lang, ReadingText.opened_at.is_(None))
        .count()
    )
    return unopened == 0


def generate_reading(
    db: Session,
    *,
    account_id: int,
    lang: str,
    length: Optional[int] = None,
    include_words: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: str = "",
) -> Dict[str, Any]:
    """Unified wrapper: schedule background generation and return the newly created reading.

    This avoids direct LLM calls here and relies on gen_queue's single implementation
    (with provider fallback and logging). It long-polls the DB for up to ~25s.
    """

    # Build prompt spec to return prompt/words/level_hint (kept for API parity)
    spec, words, level_hint = build_reading_prompt_spec(
        db,
        account_id=account_id,
        lang=lang,
        length=length,
        include_words=include_words,
    )
    messages = build_reading_prompt(spec)

    # Kick background generation if needed
    try:
        ensure_text_available(db, account_id, lang)
    except Exception:
        pass

    # Long-poll until a new unopened text is available
    deadline = time.time() + 25.0
    rt: Optional[ReadingText] = None
    while time.time() < deadline and rt is None:
        try:
            db.expire_all()
        except Exception:
            pass
        rt = (
            db.query(ReadingText)
            .filter(ReadingText.account_id == account_id, ReadingText.lang == lang, ReadingText.opened_at.is_(None))
            .order_by(ReadingText.created_at.desc())
            .first()
        )
        if rt is not None:
            break
        time.sleep(0.5)

    if rt is None:
        # Return only prompt metadata; caller can retry later
        return {
            "prompt": messages,
            "text": "",
            "level_hint": level_hint,
            "words": words,
            "text_id": None,
            "structured_translations": None,
            "word_translations": None,
        }

    # Optionally attach translations/words if already available
    # Structured translations (sentence unit)
    prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    target_lang = prof.target_lang if prof and getattr(prof, "target_lang", None) else "en"
    rtts = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.account_id == account_id,
            ReadingTextTranslation.text_id == rt.id,
            ReadingTextTranslation.unit == "sentence",
            ReadingTextTranslation.target_lang == target_lang,
        )
        .order_by(ReadingTextTranslation.segment_index.asc().nullsfirst())
        .all()
    )
    structured = None
    if rtts:
        structured = {
            "target_lang": target_lang,
            "paragraphs": [
                {
                    "sentences": [
                        {"text": r.source_text, "translation": r.translated_text} for r in rtts
                    ]
                }
            ],
        }

    # Word translations
    wgs = (
        db.query(ReadingWordGloss)
        .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == rt.id)
        .order_by(ReadingWordGloss.span_start.asc().nullsfirst(), ReadingWordGloss.span_end.asc().nullsfirst())
        .all()
    )
    word_struct = None
    if wgs:
        word_struct = {
            "words": [
                {
                    "word": w.surface,
                    "lemma": w.lemma,
                    "part_of_speech": w.pos,
                    "pinyin": w.pinyin,
                    "translation": w.translation,
                    "lemma_translation": w.lemma_translation,
                    "grammar": w.grammar,
                }
                for w in wgs
            ]
        }

    return {
        "prompt": messages,
        "text": rt.content,
        "level_hint": level_hint,
        "words": words,
        "text_id": rt.id,
        "structured_translations": structured,
        "word_translations": word_struct,
    }
