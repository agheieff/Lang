"""
LLM module - API client and prompt building for language learning.
"""

from typing import Any, Dict, List, Optional, Tuple

from .client import chat_complete, resolve_model
from .prompts import (
    build_reading_prompt,
    build_translation_prompt,
    build_structured_translation_prompt,
    build_word_translation_prompt,
    PromptSpec,
    TranslationSpec,
)
from ..services.word_selection import urgent_words_detailed
from sqlalchemy.orm import Session
from server.auth import Account as User
from ..models import Profile, Lexeme


def _profile_for_lang(db: Session, user: User, lang: str) -> Optional[Profile]:
    """Get profile for user and language."""
    return db.query(Profile).filter(Profile.account_id == user.id, Profile.lang == lang).first()


def _bucket_zh(level_value: float) -> Tuple[str, str]:
    """Bucket Chinese level value into descriptive categories."""
    if level_value < 0.2:
        return "A1", "Beginner"
    elif level_value < 0.4:
        return "A2", "Elementary"
    elif level_value < 0.6:
        return "B1", "Intermediate"
    elif level_value < 0.8:
        return "B2", "Upper Intermediate"
    else:
        return "C1", "Advanced"


def pick_words(db: Session, user: User, lang: str, count: int = 12, *, new_ratio: Optional[float] = None) -> List[str]:
    """Return only forms for prompt inclusion.
    new_ratio: desired fraction of new words among picked (0..1). If None, default from urgent_words_detailed.
    """
    kwargs: Dict[str, Any] = {"total": count}
    if isinstance(new_ratio, (int, float)):
        kwargs["new_ratio"] = float(new_ratio)
    return [it["form"] for it in urgent_words_detailed(db, user, lang, **kwargs)]


def estimate_level(db: Session, user: User, lang: str) -> Optional[str]:
    # For zh, approximate HSK by taking the most common level among user lexemes with moderate stability
    if not lang.startswith("zh"):
        return None
    prof = _profile_for_lang(db, user, lang)
    if not prof:
        return None
    pid = prof.id
    rows = (
        db.query(Lexeme)
        .filter(
            Lexeme.account_id == user.id,
            Lexeme.profile_id == pid,
            Lexeme.stability >= 0.3,
            Lexeme.level_code != None,
        )
        .limit(100)
        .all()
    )
    if not rows:
        return None
    counts: Dict[str, int] = {}
    for lx in rows:
        if lx and lx.level_code:
            counts[lx.level_code] = counts.get(lx.level_code, 0) + 1
    if not counts:
        return None
    most_common = max(counts.items(), key=lambda x: x[1])
    return most_common[0]


def compose_level_hint(db: Session, user: User, lang: str) -> Optional[str]:
    prof = _profile_for_lang(db, user, lang)
    if prof and lang.startswith("zh"):
        code, desc = _bucket_zh(getattr(prof, "level_value", 0.0) or 0.0)
        return f"{code}: {desc}"
    # Fallback to inferred level codes
    return estimate_level(db, user, lang)


__all__ = [
    "chat_complete",
    "resolve_model",
    "build_reading_prompt",
    "build_translation_prompt",
    "build_structured_translation_prompt",
    "build_word_translation_prompt",
    "PromptSpec",
    "TranslationSpec",
    "pick_words",
    "compose_level_hint",
]