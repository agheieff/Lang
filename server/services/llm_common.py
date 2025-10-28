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
    ci_target = get_ci_target(db, account_id, lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))
    words = include_words or _pick_words(db, u, lang, count=12, new_ratio=base_new_ratio)
    level_hint = compose_level_hint(db, u, lang)

    spec = PromptSpec(
        lang=lang,
        unit=unit,
        approx_len=approx_len,
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
    )
    return spec, words, level_hint
