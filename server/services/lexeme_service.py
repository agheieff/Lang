from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from ..models import Lexeme, LexemeVariant, Profile


def _canon_lang(code: str) -> str:
    return "zh" if code.startswith("zh") else code


def resolve_lexeme(
    db: Session,
    lang: str,
    lemma: str,
    pos: Optional[str],
    account_id: Optional[int] = None,
    profile_id: Optional[int] = None,
) -> Lexeme:
    """Resolve or create a lexeme for a given word.
    
    If account_id and profile_id are provided, creates user-specific lexemes.
    Otherwise falls back to shared lexemes (legacy behavior).
    """
    canon = _canon_lang(lang)
    
    # Build query filters
    filters = [Lexeme.lang == canon, Lexeme.lemma == lemma, Lexeme.pos == pos]
    if account_id is not None:
        filters.append(Lexeme.account_id == account_id)
    if profile_id is not None:
        filters.append(Lexeme.profile_id == profile_id)
    
    lx = db.query(Lexeme).filter(*filters).first()
    if lx:
        return lx
    
    # Create new lexeme
    lx = Lexeme(
        lang=canon,
        lemma=lemma,
        pos=pos,
        account_id=account_id or 0,
        profile_id=profile_id or 0,
    )
    db.add(lx)
    db.flush()
    
    # For zh, ensure a variant for script form (best-effort; caller can add more)
    if canon == "zh":
        if not db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lx.id).first():
            db.add(LexemeVariant(lexeme_id=lx.id, script="Hans", form=lemma))
            db.flush()
    return lx


def get_or_create_userlexeme(db: Session, account_id: int, prof: Profile, lex: Lexeme) -> Lexeme:
    # Since lexemes are now user-specific, we just return the lexeme itself
    # This function now serves as a compatibility wrapper
    return lex

