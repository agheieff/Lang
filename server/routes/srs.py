from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth import Account  # type: ignore

from ..db import get_db
from ..db import get_global_db
from ..deps import get_current_account as _get_current_account, require_tier
from ..level import get_level_summary, update_level_if_stale
from ..services.word_selection import urgent_words_detailed
from ..models import Lexeme, Profile
from ..services.lexeme_service import (
    get_or_create_userlexeme as _get_or_create_userlexeme,
    resolve_lexeme as _resolve_lexeme,
)
from ..services.srs_service import (
    srs_click as _svc_srs_click,
    srs_exposure as _svc_srs_exposure,
    srs_nonlookup as _svc_srs_nonlookup,
)


router = APIRouter(tags=["srs"])


def _hash_context(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h[:32]
    except Exception:
        return None


class ExposureItem(BaseModel):
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None


class ExposuresRequest(BaseModel):
    lang: str
    items: List[ExposureItem]
    text_id: Optional[int] = None


class NonLookupRequest(BaseModel):
    lang: str
    items: List[ExposureItem]
    text_id: Optional[int] = None


class ClickRequest(BaseModel):
    lang: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    surface: Optional[str] = None
    context: Optional[str] = None
    text_id: Optional[int] = None


@router.get("/srs/urgent")
def srs_urgent(
    lang: str,
    total: int = 12,
    new_ratio: float = 0.3,
    db: Session = Depends(get_db),
    global_db: Session = Depends(get_global_db),
    account: Account = Depends(_get_current_account),
):
    items = urgent_words_detailed(db, global_db, account, lang, total=total, new_ratio=new_ratio)
    return {"words": [it["form"] for it in items], "items": items}


# LexemeInfo upsert endpoint removed - frequency and level data now stored directly in Lexeme model


# LexemeInfo get endpoint removed - frequency and level data now stored directly in Lexeme model


@router.post("/srs/event/exposures")
def srs_exposures(
    req: ExposuresRequest,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    count = 0
    for it in req.items:
        lemma = it.lemma
        pos = it.pos
        if not lemma and it.surface:
            # Fallback: use surface form as lemma when no lemma provided
            lemma = it.surface
        if not lemma:
            continue
        _svc_srs_exposure(
            db,
            global_db,
            account_id=account.id,
            lang=req.lang,
            lemma=lemma,
            pos=pos,
            surface=it.surface,
            context_hash=_hash_context(it.context),
            text_id=req.text_id,
        )
        count += 1
    try:
        update_level_if_stale(db, account.id, req.lang)
    except Exception:
        pass
    db.commit()
    return {"ok": True, "count": count}


@router.post("/srs/event/nonlookup")
def srs_nonlookup(
    req: NonLookupRequest,
    db: Session = Depends(get_db),
    global_db: Session = Depends(get_global_db),
    account: Account = Depends(_get_current_account),
):
    count = 0
    for it in req.items:
        lemma = it.lemma
        pos = it.pos
        if not lemma and it.surface:
            # Fallback: use surface form as lemma when no lemma provided
            lemma = it.surface
        if not lemma:
            continue
        _svc_srs_nonlookup(
            db,
            global_db,
            account_id=account.id,
            lang=req.lang,
            lemma=lemma,
            pos=pos,
            surface=it.surface,
            context_hash=_hash_context(it.context),
            text_id=req.text_id,
        )
        count += 1
    try:
        update_level_if_stale(global_db, account.id, req.lang, account_db=db)
    except Exception:
        pass
    db.commit()
    return {"ok": True, "count": count}


@router.post("/srs/event/click")
def srs_click(
    req: ClickRequest,
    db: Session = Depends(get_db),
    global_db: Session = Depends(get_global_db),
    account: Account = Depends(_get_current_account),
):
    lemma = req.lemma
    pos = req.pos
    if not lemma and req.surface:
        # Fallback: use surface form as lemma when no lemma provided
        lemma = req.surface
    if not lemma:
        raise HTTPException(400, "lemma or surface required")
    _svc_srs_click(
        db,
        global_db,
        account_id=account.id,
        lang=req.lang,
        lemma=lemma,
        pos=pos,
        surface=req.surface,
        context_hash=_hash_context(req.context),
        text_id=req.text_id,
    )
    try:
        update_level_if_stale(global_db, account.id, req.lang, account_db=db)
        db.commit()
    except Exception:
        pass
    
    # Get profile to query user-specific lexeme
    prof = db.query(Profile).filter(
        Profile.account_id == account.id,
        Profile.lang == req.lang
    ).first()
    if not prof:
        return {"ok": True}
    
    lex = _resolve_lexeme(db, req.lang, lemma, pos, account_id=account.id, profile_id=prof.id)
    a = lex.a_click or 0
    b = lex.b_nonclick or 0
    p_click = a / (a + b) if (a + b) > 0 else 0.0
    return {
        "ok": True,
        "lexeme_id": lex.id,
        "p_click": p_click,
        "n": a + b,
        "stability": lex.stability,
        "diversity": lex.distinct_texts,
    }


class SrsWordsOut(BaseModel):
    lexeme_id: int
    lemma: str
    pos: Optional[str]
    p_click: float
    n: int
    stability: float
    diversity: int
    freq_rank: Optional[int] = None
    level_code: Optional[str] = None
    next_due_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None


@router.get("/srs/words", response_model=List[SrsWordsOut])
def get_srs_words(
    lang: str = Query(...),
    min_p: Optional[float] = None,
    max_p: Optional[float] = None,
    min_S: Optional[float] = None,
    max_S: Optional[float] = None,
    min_D: Optional[int] = None,
    max_D: Optional[int] = None,
    limit: int = 500,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    q = (
        db.query(Lexeme)
        .filter(
            Lexeme.account_id == account.id,
            Lexeme.profile_id
            == db.query(Profile.id)
            .filter(Profile.account_id == account.id, Profile.lang == lang)
            .scalar_subquery(),
        )
    )
    # Sort by next due date (most urgent first)
    q = q.order_by(Lexeme.next_due_at.asc().nullsfirst())
    
    if min_S is not None:
        q = q.filter(Lexeme.stability >= float(min_S))
    if max_S is not None:
        q = q.filter(Lexeme.stability <= float(max_S))
    if min_D is not None:
        q = q.filter(Lexeme.distinct_texts >= int(min_D))
    if max_D is not None:
        q = q.filter(Lexeme.distinct_texts <= int(max_D))
    if min_p is not None or max_p is not None:
        denom = (Lexeme.a_click + Lexeme.b_nonclick)
        q = q.filter(denom > 0)
        if min_p is not None:
            q = q.filter((Lexeme.a_click * 1.0) / (denom * 1.0) >= float(min_p))
        if max_p is not None:
            q = q.filter((Lexeme.a_click * 1.0) / (denom * 1.0) <= float(max_p))
    rows = q.limit(2000).all()
    out: List[SrsWordsOut] = []
    for lex in rows:
        a = lex.a_click or 0
        b = lex.b_nonclick or 0
        p = (a / (a + b)) if (a + b) > 0 else 0.0
        S = float(lex.stability or 0.0)
        D = int(lex.distinct_texts or 0)
        if min_p is not None and p < min_p:
            continue
        if max_p is not None and p > max_p:
            continue
        if min_S is not None and S < min_S:
            continue
        if max_S is not None and S > max_S:
            continue
        if min_D is not None and D < min_D:
            continue
        if max_D is not None and D > max_D:
            continue
        out.append(
            SrsWordsOut(
                lexeme_id=lex.id,
                lemma=lex.lemma,
                pos=lex.pos,
                p_click=p,
                n=(a + b),
                stability=S,
                diversity=D,
                freq_rank=lex.frequency_rank,
                level_code=lex.level_code,
                next_due_at=lex.next_due_at,
                last_seen_at=lex.last_seen_at,
            )
        )
        if len(out) >= limit:
            break
    return out


class SrsStatsOut(BaseModel):
    total: int
    by_p: Dict[str, int]
    by_S: Dict[str, int]
    by_D: Dict[str, int]


@router.get("/srs/stats", response_model=SrsStatsOut)
def get_srs_stats(
    lang: str,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    pid = (
        db.query(Profile.id)
        .filter(Profile.account_id == account.id, Profile.lang == lang)
        .scalar()
    )
    if not pid:
        return SrsStatsOut(total=0, by_p={}, by_S={}, by_D={})
    rows = (
        db.query(Lexeme)
        .filter(Lexeme.account_id == account.id, Lexeme.profile_id == pid)
        .all()
    )

    def bucketize(val: float, bounds: List[float]):
        for i, b in enumerate(bounds):
            if val < b:
                return f"<{b}"
        return f">={bounds[-1]}"

    by_p: Dict[str, int] = {}
    by_S: Dict[str, int] = {}
    by_D: Dict[str, int] = {}
    for lex in rows:
        a = lex.a_click or 0
        b = lex.b_nonclick or 0
        p = (a / (a + b)) if (a + b) > 0 else 0.0
        S = float(lex.stability or 0.0)
        D = int(lex.distinct_texts or 0)
        pb = bucketize(p, [0.2, 0.4, 0.6, 0.8])
        Sb = bucketize(S, [0.33, 0.66, 0.85])
        Db = bucketize(min(1.0, 1 - pow(2.71828, -D / 8.0)), [0.25, 0.5, 0.75])
        by_p[pb] = by_p.get(pb, 0) + 1
        by_S[Sb] = by_S.get(Sb, 0) + 1
        by_D[Db] = by_D.get(Db, 0) + 1
    return SrsStatsOut(total=len(rows), by_p=by_p, by_S=by_S, by_D=by_D)


class LevelOut(BaseModel):
    level_value: float
    level_var: float
    last_update_at: Optional[str]
    last_activity_at: Optional[str]
    ess: float
    bins: Dict[str, List[float]]


@router.get("/srs/level", response_model=LevelOut)
def get_srs_level(
    lang: str,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    try:
        update_level_if_stale(db, account.id, lang)
        db.commit()
    except Exception:
        pass
    prof = (
        db.query(Profile)
        .filter(Profile.account_id == account.id, Profile.lang == lang)
        .first()
    )
    if not prof:
        raise HTTPException(404, "profile not found")
    summ = get_level_summary(db, prof)
    return LevelOut(**summ)

