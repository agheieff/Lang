from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    Lexeme,
    LexemeInfo,
    UserLexeme,
    UserLexemeContext,
    WordEvent,
)
from ..config import (
    _W_CLICK,
    _W_NONLOOK,
    _W_EXPOSURE,
    _HL_CLICK_D,
    _HL_NONLOOK_D,
    _HL_EXPOSURE_D,
    _FSRS_TARGET_R,
    _FSRS_FAIL_F,
    _G_PASS_WEAK,
    _G_PASS_NORM,
    _G_PASS_STRONG,
    _SESSION_MIN,
    _EXPOSURE_WEAK_W,
    _DISTINCT_PROMOTE,
    _FREQ_LOW_THRESH,
    _DIFF_HIGH,
    _SYN_NL_ENABLE,
    _SYN_NL_MIN_DISTINCT,
    _SYN_NL_MIN_DAYS,
    _SYN_NL_COOLDOWN_DAYS,
)
from ..level import update_level_if_stale
from .srs_logic import (
    decay_posterior as _decay_posterior,
    SRSParams,
    schedule_next,
)

# Optional resolver helpers (kept to preserve behavior)
from .lexeme_service import (
    resolve_lexeme as _resolve_lexeme,
    get_or_create_userlexeme as _get_or_create_userlexeme,
)


_SRS_PARAMS = SRSParams(
    target_retention=_FSRS_TARGET_R,
    fail_factor=_FSRS_FAIL_F,
    g_pass_weak=_G_PASS_WEAK,
    g_pass_norm=_G_PASS_NORM,
    g_pass_strong=_G_PASS_STRONG,
)


def _ensure_profile(db: Session, account_id: int, lang: str) -> Profile:
    prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    if prof:
        return prof
    prof = Profile(account_id=account_id, lang=lang)
    db.add(prof)
    db.flush()
    return prof


def _schedule_next(ul: UserLexeme, quality: int, now: datetime) -> None:
    schedule_next(ul, quality, now, _SRS_PARAMS)


def srs_click(
    db: Session,
    *,
    account_id: int,
    lang: str,
    lemma: str,
    pos: Optional[str],
    surface: Optional[str],
    context_hash: Optional[str],
    text_id: Optional[int] = None,
) -> None:
    prof = _ensure_profile(db, account_id, lang)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account_id, prof, lex)
    ul.a_click = (ul.a_click or 0) + 1
    ul.clicks = (ul.clicks or 0) + 1
    ul.last_clicked_at = datetime.utcnow()
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_CLICK_D)
    ul.alpha = float((ul.alpha or 1.0) + _W_CLICK)
    _schedule_next(ul, 0, now)
    db.add(WordEvent(
        ts=now,
        account_id=account_id,
        profile_id=prof.id,
        lexeme_id=lex.id,
        event_type="click",
        count=1,
        surface=surface,
        context_hash=context_hash,
        source="manual",
        meta={},
        text_id=text_id,
    ))


def srs_exposure(
    db: Session,
    *,
    account_id: int,
    lang: str,
    lemma: str,
    pos: Optional[str],
    surface: Optional[str],
    context_hash: Optional[str],
    text_id: Optional[int] = None,
) -> None:
    prof = _ensure_profile(db, account_id, lang)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account_id, prof, lex)
    now = datetime.utcnow()
    # 1) session collapse
    recent_ev = (
        db.query(WordEvent)
        .filter(
            WordEvent.account_id == account_id,
            WordEvent.profile_id == prof.id,
            WordEvent.lexeme_id == lex.id,
            WordEvent.event_type == "exposure",
        )
        .order_by(WordEvent.ts.desc())
        .first()
    )
    session_skip = False
    if recent_ev and recent_ev.ts:
        try:
            mins = (now - recent_ev.ts).total_seconds() / 60.0
            if mins < _SESSION_MIN:
                session_skip = True
        except Exception:
            session_skip = False

    # 2) distinct gating
    w_exp = _W_EXPOSURE
    quality = 2
    if int(ul.distinct_texts or 0) < _DISTINCT_PROMOTE:
        w_exp = min(w_exp, _EXPOSURE_WEAK_W)
        quality = 1

    # 3) difficulty/frequency adj
    li = db.query(LexemeInfo).filter(LexemeInfo.lexeme_id == lex.id).first()
    if (getattr(ul, "difficulty", 1.0) or 1.0) >= _DIFF_HIGH or (
        li and li.freq_rank and li.freq_rank > _FREQ_LOW_THRESH
    ):
        w_exp = min(w_exp, _EXPOSURE_WEAK_W)
        quality = min(quality, 1)

    # 4) apply updates
    ul.b_nonclick = (ul.b_nonclick or 0) + 1
    ul.exposures = (ul.exposures or 0) + 1
    if not ul.first_seen_at:
        ul.first_seen_at = now
    ul.last_seen_at = now
    _decay_posterior(ul, now, _HL_EXPOSURE_D)
    if not session_skip and w_exp > 0:
        ul.beta = float((ul.beta or 9.0) + w_exp)
    _schedule_next(ul, quality, now)
    if context_hash:
        exists = (
            db.query(UserLexemeContext)
            .filter(
                UserLexemeContext.user_lexeme_id == ul.id,
                UserLexemeContext.context_hash == context_hash,
            )
            .first()
        )
        if not exists:
            db.add(UserLexemeContext(user_lexeme_id=ul.id, context_hash=context_hash))
            ul.distinct_texts = (ul.distinct_texts or 0) + 1
    db.add(WordEvent(
        ts=now,
        account_id=account_id,
        profile_id=prof.id,
        lexeme_id=lex.id,
        event_type="exposure",
        count=1,
        surface=surface,
        context_hash=context_hash,
        source="manual",
        meta={},
        text_id=text_id,
    ))

    # Synthetic nonlookup
    if _SYN_NL_ENABLE and (ul.clicks or 0) == 0 and int(ul.distinct_texts or 0) >= _SYN_NL_MIN_DISTINCT and ul.first_seen_at:
        try:
            days_seen = (now - ul.first_seen_at).total_seconds() / 86400.0
            if days_seen >= _SYN_NL_MIN_DAYS:
                recent_nl = (
                    db.query(WordEvent)
                    .filter(
                        WordEvent.account_id == account_id,
                        WordEvent.profile_id == prof.id,
                        WordEvent.lexeme_id == lex.id,
                        WordEvent.event_type == "nonlookup",
                        WordEvent.ts >= (now - timedelta(days=_SYN_NL_COOLDOWN_DAYS)),
                    )
                    .first()
                )
                if not recent_nl:
                    srs_nonlookup(
                        db,
                        account_id=account_id,
                        lang=lang,
                        lemma=lemma,
                        pos=pos,
                        surface=surface,
                        context_hash=context_hash,
                        text_id=text_id,
                    )
        except Exception:
            pass


def srs_nonlookup(
    db: Session,
    *,
    account_id: int,
    lang: str,
    lemma: str,
    pos: Optional[str],
    surface: Optional[str],
    context_hash: Optional[str],
    text_id: Optional[int] = None,
) -> None:
    prof = _ensure_profile(db, account_id, lang)
    lex = _resolve_lexeme(db, lang, lemma, pos)
    ul = _get_or_create_userlexeme(db, account_id, prof, lex)
    now = datetime.utcnow()
    _decay_posterior(ul, now, _HL_NONLOOK_D)
    ul.b_nonclick = (ul.b_nonclick or 0) + 2
    ul.beta = float((ul.beta or 9.0) + _W_NONLOOK)
    ul.last_seen_at = now
    _schedule_next(ul, 3, now)
    db.add(WordEvent(
        ts=now,
        account_id=account_id,
        profile_id=prof.id,
        lexeme_id=lex.id,
        event_type="nonlookup",
        count=1,
        surface=surface,
        context_hash=context_hash,
        source="manual",
        meta={},
        text_id=text_id,
    ))
