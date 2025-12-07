from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from server.auth import Account as User
from ..models import (
    Profile,
    Lexeme,
    LexemeVariant,
)
from ..level import update_level_if_stale


def _profile_for_lang(db: Session, user: User, lang: str) -> Optional[Profile]:
    return db.query(Profile).filter(Profile.account_id == user.id, Profile.lang == lang).first()


def _script_from_lang(lang: str) -> Optional[str]:
    if lang.startswith("zh"):
        # Default to simplified when no explicit preference present
        return "Hans"
    return None


def _variant_form_for_lang(db: Session, user: User, lx: Lexeme, lang: str) -> str:
    form = lx.lemma
    if lx.lang == "zh":
        script = None
        # Prefer user profile setting when available
        prof = _profile_for_lang(db, user, lang)
        if prof and getattr(prof, "preferred_script", None):
            script = prof.preferred_script
        if not script:
            script = _script_from_lang(lang)
        if script:
            v = db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lx.id, LexemeVariant.script == script).first()
            if v:
                form = v.form
    return form


def _hsk_numeric(level_code: Optional[str]) -> Optional[int]:
    if not level_code:
        return None
    # Expect codes like HSK1..HSK6
    try:
        if level_code.upper().startswith("HSK"):
            return int(level_code[3:])
    except Exception:
        return None
    return None


def urgent_words_detailed(db: Session, user: User, lang: str, total: int = 12, new_ratio: float = 0.3) -> List[Dict[str, Any]]:
    # ensure level estimate is fresh enough
    try:
        update_level_if_stale(db, user.id, lang)
    except Exception:
        pass
    prof = _profile_for_lang(db, user, lang)
    if not prof:
        return []
    pid = prof.id

    # Query lexemes with familiarity scores
    stmt = (
        select(
            Lexeme.id,
            Lexeme.lemma,
            Lexeme.lang,
            Lexeme.level_code,
            Lexeme.familiarity,
            Lexeme.last_seen,
        )
        .where(Lexeme.account_id == user.id, Lexeme.profile_id == pid, Lexeme.lang == lang)
        .order_by(
            Lexeme.familiarity.asc().nullsfirst(),
            Lexeme.level_code.asc().nullslast(),
        )
    )
    rows = db.execute(stmt).all()

    # Separate known and unknown words
    known = []
    unknown = []
    for r in rows:
        item = {
            "id": r.id,
            "lemma": r.lemma,
            "lang": r.lang,
            "hsk_level": r.level_code,
            "familiarity": r.familiarity,
            "last_seen": r.last_seen,
        }
        if r.familiarity is not None and r.familiarity > 0.3:
            known.append(item)
        else:
            unknown.append(item)

    # Calculate target counts
    target_new = max(1, int(total * new_ratio))
    target_known = total - target_new

    # Select words
    selected = []
    # Add unknown words (prioritize by HSK level)
    for w in unknown[:target_new]:
        selected.append(w)
    # Add known words (prioritize least familiar)
    for w in known[:target_known]:
        selected.append(w)

    # Fill remaining slots if needed
    if len(selected) < total:
        remaining = total - len(selected)
        # Try to add more unknown words
        for w in unknown[len(selected) - target_new : len(selected) - target_new + remaining]:
            selected.append(w)
        # If still not enough, add more known words
        if len(selected) < total:
            remaining = total - len(selected)
            for w in known[len(selected) - target_known : len(selected) - target_known + remaining]:
                selected.append(w)

    # Convert to display form
    result = []
    for w in selected:
        lx = db.get(Lexeme, w["id"])
        if lx:
            form = _variant_form_for_lang(db, user, lx, lang)
            result.append({
                "id": w["id"],
                "lemma": w["lemma"],
                "form": form,
                "hsk_level": w["hsk_level"],
                "familiarity": w["familiarity"],
                "last_seen": w["last_seen"],
            })
    return result


def pick_words(account_db: Session, global_db: Session, user: User, lang: str, count: int = 12, new_ratio: float = 0.1) -> List[str]:
    """Pick words for inclusion in generated text."""
    # Use the account_db for user-specific data
    items = urgent_words_detailed(account_db, user, lang, total=count, new_ratio=new_ratio)
    return [item["form"] for item in items]


def estimate_level(account_db: Session, global_db: Session, user: User, lang: str) -> Optional[str]:
    """Estimate user's language level."""
    prof = _profile_for_lang(account_db, user, lang)
    if not prof:
        return None
    level_val = getattr(prof, "level_value", None)
    if level_val is None:
        return None

    if lang.startswith("zh"):
        return _bucket_zh(float(level_val))[0]

    # Generic level estimation for other languages
    if level_val < 0.1:
        return "Beginner"
    elif level_val < 0.3:
        return "Elementary"
    elif level_val < 0.6:
        return "Intermediate"
    elif level_val < 0.8:
        return "Upper Intermediate"
    else:
        return "Advanced"


def _bucket_zh(level_value: float) -> Tuple[str, str]:
    """Convert numeric level to HSK-like codes for Chinese."""
    if level_value < 0.05:
        return "HSK1", "Beginner"
    elif level_value < 0.15:
        return "HSK2", "Elementary"
    elif level_value < 0.35:
        return "HSK3", "Intermediate"
    elif level_value < 0.6:
        return "HSK4", "Upper Intermediate"
    elif level_value < 0.8:
        return "HSK5", "Advanced"
    else:
        return "HSK6", "Proficient"


def compose_level_hint(account_db: Session, global_db: Session, user: User, lang: str) -> Optional[str]:
    """Compose a level hint for LLM prompts."""
    prof = _profile_for_lang(account_db, user, lang)
    if prof and lang.startswith("zh"):
        code, desc = _bucket_zh(getattr(prof, "level_value", 0.0) or 0.0)
        return f"{code}: {desc}"
    # Fallback to inferred level codes
    return estimate_level(account_db, global_db, user, lang)
