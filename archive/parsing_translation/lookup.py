from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from server.auth import decode_token  # type: ignore

from nlp.parsing import ENGINES as _ENGINES, format_morph_label
from nlp.parsing.dicts import (
    CedictProvider,
    DictionaryProviderChain,
    StarDictProvider,
)

from ..account_db import get_db
from ..models import Profile, ReadingLookup, ReadingText
from ..services.srs_service import srs_click as _svc_srs_click


router = APIRouter(prefix="/api", tags=["lookup"])

_DICT_CHAIN: Optional[DictionaryProviderChain] = None
_JWT_SECRET = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")


def get_dict_chain() -> DictionaryProviderChain:
    global _DICT_CHAIN
    if _DICT_CHAIN is None:
        _DICT_CHAIN = DictionaryProviderChain(providers=[CedictProvider(), StarDictProvider()])
    return _DICT_CHAIN


def _hash_context(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    try:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return h[:32]
    except Exception:
        return None


class LookupRequest(BaseModel):
    source_lang: str = Field(..., description="BCP-47 or ISO code, e.g., 'es'")
    surface: str = Field(..., description="Surface form as clicked in text")
    context: Optional[str] = Field(None, description="Optional sentence context for disambiguation")
    text_id: Optional[int] = Field(None, description="Reading text id for caching")
    start: Optional[int] = Field(None, description="Start offset within text")
    end: Optional[int] = Field(None, description="End offset within text (exclusive)")


@router.post("/lookup")
def lookup(req: LookupRequest, request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    engine = _ENGINES.get(req.source_lang)
    if not engine:
        raise HTTPException(status_code=400, detail="language not supported yet")

    user = None
    target_lang = "en"
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        try:
            data = decode_token(auth.split(" ", 1)[1], _JWT_SECRET, ["HS256"])  # type: ignore
            uid = int((data or {}).get("sub")) if data and data.get("sub") is not None else None
            if uid is not None:
                class _U:
                    pass

                _tmp = _U()
                _tmp.id = uid
                user = _tmp
                profile = db.query(Profile).filter(
                    Profile.account_id == uid, Profile.lang == req.source_lang
                ).first()
                if profile and profile.target_lang:
                    target_lang = profile.target_lang
        except Exception:
            pass

    analysis = engine.analyze_word(req.surface, context=req.context)
    lemma = analysis.get("lemma") or req.surface
    morph = analysis.get("morph") or {}
    label = format_morph_label(analysis.get("pos"), morph)

    translations: Dict[str, Any] | list | None = None
    mode = "analysis"

    uid: Optional[int] = None
    rt: Optional[ReadingText] = None
    try:
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            data = decode_token(auth.split(" ", 1)[1], _JWT_SECRET, ["HS256"])  # type: ignore
            uid = int((data or {}).get("sub")) if data and data.get("sub") is not None else None
    except Exception:
        uid = None

    context_hash = _hash_context(req.context)
    if req.text_id is not None and req.start is not None and req.end is not None and user is not None:
        try:
            rt = db.get(ReadingText, int(req.text_id))
            if rt and rt.account_id == user.id:
                s = max(0, int(req.start))
                e = int(req.end)
                if e > s:
                    if rt.content:
                        e = min(e, len(rt.content))
                    row = (
                        db.query(ReadingLookup)
                        .filter(
                            ReadingLookup.account_id == user.id,
                            ReadingLookup.text_id == rt.id,
                            ReadingLookup.target_lang == target_lang,
                            ReadingLookup.span_start == s,
                            ReadingLookup.span_end == e,
                        )
                        .first()
                    )
                    if row and row.translations:
                        translations = row.translations
                        mode = "translation"
        except Exception:
            pass

    if translations is None:
        tr = get_dict_chain().translations(req.source_lang, target_lang, lemma)
        if not tr and req.surface != lemma:
            more = get_dict_chain().translations(req.source_lang, target_lang, req.surface)
            if more:
                tr = more
        translations = tr or []
        mode = "translation" if translations else "analysis"
        if user is not None and rt is not None and req.start is not None and req.end is not None:
            try:
                s = max(0, int(req.start))
                e = int(req.end)
                if rt.content:
                    e = min(e, len(rt.content))
                from sqlalchemy.dialects.sqlite import insert

                RL = ReadingLookup
                stmt = insert(RL).values(
                    account_id=user.id,
                    text_id=rt.id,
                    lang=req.source_lang,
                    target_lang=target_lang,
                    surface=req.surface,
                    lemma=lemma,
                    pos=analysis.get("pos"),
                    span_start=s,
                    span_end=e,
                    context_hash=context_hash,
                    translations=(translations if isinstance(translations, (dict, list)) else {}),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        RL.account_id,
                        RL.text_id,
                        RL.target_lang,
                        RL.span_start,
                        RL.span_end,
                    ],
                    set_={
                        "surface": stmt.excluded.surface,
                        "lemma": stmt.excluded.lemma,
                        "pos": stmt.excluded.pos,
                        "context_hash": stmt.excluded.context_hash,
                        "translations": stmt.excluded.translations,
                    },
                )
                db.execute(stmt)
            except Exception:
                pass

    resp = {
        "mode": mode,
        "surface": req.surface,
        "lemma": lemma,
        "pos": analysis.get("pos"),
        "morph": morph,
        "label": label,
        "translations": translations,
        "script": analysis.get("script"),
        "pronunciation": analysis.get("pronunciation"),
    }
    try:
        if user is not None:
            _svc_srs_click(
                db,
                account_id=user.id,
                lang=req.source_lang,
                lemma=lemma,
                pos=analysis.get("pos"),
                surface=req.surface,
                context_hash=context_hash,
                text_id=(rt.id if rt else None),
            )
            db.commit()
    except Exception:
        pass
    return resp

