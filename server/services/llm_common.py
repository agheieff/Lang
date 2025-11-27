from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models import Profile
from ..llm.prompts import PromptSpec
from ..services.level_service import get_ci_target
from ..llm import compose_level_hint
from .word_selection import pick_words as _pick_words


def build_reading_prompt_spec(
    db: Session,
    *,
    account_id: int,
    lang: str,
    length: Optional[int] = None,
    include_words: Optional[List[str]] = None,
    ci_target_override: Optional[float] = None,
    topic: Optional[str] = None,
) -> Tuple[PromptSpec, List[str], Optional[str]]:
    """Assemble PromptSpec and supporting values (words, level_hint)."""
    script = None
    if lang.startswith("zh"):
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"

    unit = "chars" if lang.startswith("zh") else "words"
    prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    prof_len = None
    try:
        if prof and isinstance(prof.text_length, int) and prof.text_length and prof.text_length > 0:
            prof_len = int(prof.text_length)
    except Exception:
        prof_len = None
    approx_len = length if length is not None else (prof_len if prof_len is not None else (300 if unit == "chars" else 180))
    try:
        approx_len = max(50, min(2000, int(approx_len)))
    except Exception:
        approx_len = 300 if unit == "chars" else 180

    class _U: pass
    u = _U(); u.id = account_id
    # Use override ci_target if provided (for pool generation), otherwise use profile preference
    ci_target = ci_target_override if ci_target_override is not None else get_ci_target(db, account_id, lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))
    words = include_words or _pick_words(db, u, lang, count=12, new_ratio=base_new_ratio)
    level_hint = compose_level_hint(db, u, lang)

    # Preferences: free-form topics/styles from profile settings
    preferences: Optional[str] = None
    try:
        if prof and getattr(prof, "text_preferences", None):
            preferences = str(prof.text_preferences)
        elif prof and isinstance(getattr(prof, "settings", None), dict):
            p = prof.settings.get("text_preferences") or prof.settings.get("preferences")
            if p:
                preferences = str(p)
    except Exception:
        preferences = None

    spec = PromptSpec(
        lang=lang,
        unit=unit,
        approx_len=approx_len,
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
        preferences=preferences,
        recent_titles=_get_recent_read_titles(db, account_id, lang),
        topic=topic,
    )
    return spec, words, level_hint


def _get_recent_read_titles(db: Session, account_id: int, lang: str, limit: int = 5) -> List[str]:
    """Fetch titles of the last N read texts for this user/language."""
    from ..models import ReadingText, ReadingTextTranslation
    
    # Find last N read texts
    texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.account_id == account_id,
            ReadingText.lang == lang,
            ReadingText.read_at.is_not(None)
        )
        .order_by(ReadingText.read_at.desc())
        .limit(limit)
        .all()
    )
    
    if not texts:
        return []
    
    text_ids = [t.id for t in texts]
    
    # Fetch titles (unit='text', segment_index=0)
    # Note: This relies on titles being translated/stored in ReadingTextTranslation.
    # If titles are only in LLMRequestLog (legacy), this might miss them,
    # but going forward this is the standard.
    titles = []
    
    # Bulk fetch translations for these text IDs
    rows = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.account_id == account_id,
            ReadingTextTranslation.text_id.in_(text_ids),
            ReadingTextTranslation.unit == "text",
            ReadingTextTranslation.segment_index == 0
        )
        .all()
    )
    
    # Map text_id -> source_text (which is the title)
    title_map = {r.text_id: r.source_text for r in rows}
    
    # Reconstruct list in read_at order
    for t in texts:
        if t.id in title_map:
            titles.append(title_map[t.id])
            
    return titles
