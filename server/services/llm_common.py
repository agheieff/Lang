from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models import Profile
from ..llm.prompts import PromptSpec
from ..services.level_service import get_ci_target
from ..llm import compose_level_hint
from .word_selection import pick_words as _pick_words


def build_reading_prompt_spec(
    global_db: Session,
    *,
    account_id: int,
    lang: str,
    account_db: Optional[Session] = None,
    length: Optional[int] = None,
    include_words: Optional[List[str]] = None,
    ci_target_override: Optional[float] = None,
    topic: Optional[str] = None,
) -> Tuple[PromptSpec, List[str], Optional[str]]:
    """Assemble PromptSpec and supporting values (words, level_hint).
    
    Args:
        global_db: Session for global DB (Profile, ReadingText)
        account_db: Session for per-account DB (Lexeme). If None, word selection is skipped.
    """
    script = None
    if lang.startswith("zh"):
        prof = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"

    unit = "chars" if lang.startswith("zh") else "words"
    prof = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
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
    ci_target = ci_target_override if ci_target_override is not None else get_ci_target(global_db, account_id, lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))
    # Word selection requires per-account DB for Lexeme queries
    if account_db is not None:
        words = include_words or _pick_words(account_db, u, lang, count=12, new_ratio=base_new_ratio)
        level_hint = compose_level_hint(account_db, u, lang)
    else:
        words = include_words or []
        level_hint = None

    spec = PromptSpec(
        lang=lang,
        unit=unit,
        approx_len=approx_len,
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
        recent_titles=_get_recent_read_titles(global_db, account_id, lang),
        topic=topic,
    )
    return spec, words, level_hint


def _get_recent_read_titles(global_db: Session, account_id: int, lang: str, limit: int = 5) -> List[str]:
    """Fetch titles of the last N read texts for this user/language.
    
    Note: With the global pool architecture, reading history is tracked in 
    ProfileTextRead (per-account DB). This function returns titles from global DB
    but would need account_db to properly filter by read history.
    
    For now, returns empty list - TODO: implement properly with both DBs.
    """
    # TODO: Implement properly using ProfileTextRead from account_db
    # to get recently read text_ids, then fetch titles from global_db
    return []
