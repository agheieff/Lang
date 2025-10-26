from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from ..models import Lexeme, LexemeVariant, UserLexeme, Profile


def _canon_lang(code: str) -> str:
    return "zh" if code.startswith("zh") else code


def resolve_lexeme(db: Session, lang: str, lemma: str, pos: Optional[str]) -> Lexeme:
    canon = _canon_lang(lang)
    lx = (
        db.query(Lexeme)
        .filter(Lexeme.lang == canon, Lexeme.lemma == lemma, Lexeme.pos == pos)
        .first()
    )
    if lx:
        return lx
    lx = Lexeme(lang=canon, lemma=lemma, pos=pos)
    db.add(lx)
    db.flush()
    # For zh, ensure a variant for script form (best-effort; caller can add more)
    if canon == "zh":
        if not db.query(LexemeVariant).filter(LexemeVariant.lexeme_id == lx.id).first():
            db.add(LexemeVariant(lexeme_id=lx.id, script="Hans", form=lemma))
            db.flush()
    return lx


def get_or_create_userlexeme(db: Session, account_id: int, prof: Profile, lex: Lexeme) -> UserLexeme:
    ul = (
        db.query(UserLexeme)
        .filter(
            UserLexeme.account_id == account_id,
            UserLexeme.profile_id == prof.id,
            UserLexeme.lexeme_id == lex.id,
        )
        .first()
    )
    if ul:
        return ul
    ul = UserLexeme(account_id=account_id, profile_id=prof.id, lexeme_id=lex.id)
    db.add(ul)
    db.flush()
    return ul

