from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from server.auth import Account  # type: ignore

from ..account_db import get_db
from ..deps import get_current_account as _get_current_account
from ..models import (
    Profile,
    ReadingLookup,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
)
from ..services.llm_service import generate_reading as _svc_generate_reading


router = APIRouter(tags=["reading"])


class GenRequest(BaseModel):
    lang: str
    length: Optional[int] = None
    include_words: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = "openrouter"
    base_url: str = "http://localhost:1234/v1"


@router.post("/gen/reading")
def gen_reading(
    req: GenRequest,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    try:
        # Check if we should generate a new text
        from ..services.llm_service import should_generate_new_text
        if not should_generate_new_text(db, account.id, req.lang):
            # Return the most recent unopened text
            unopened_text = (
                db.query(ReadingText)
                .filter(
                    ReadingText.account_id == account.id,
                    ReadingText.lang == req.lang,
                    ReadingText.opened_at.is_(None)
                )
                .order_by(ReadingText.created_at.desc())
                .first()
            )
            if unopened_text:
                return {
                    "text": unopened_text.content,
                    "text_id": unopened_text.id,
                    "level_hint": None,
                    "words": [],
                    "prompt": None,
                    "structured_translations": None,
                    "word_translations": None
                }
            else:
                # Fallback: generate anyway
                pass

        return _svc_generate_reading(
            db,
            account_id=account.id,
            lang=req.lang,
            length=req.length,
            include_words=req.include_words,
            model=req.model,
            provider=req.provider,
            base_url=req.base_url,
        )
    except Exception as e:
        error_msg = str(e)
        if "credits exhausted" in error_msg or "add credits" in error_msg:
            raise HTTPException(status_code=402, detail=error_msg)
        elif "rate limit" in error_msg:
            raise HTTPException(status_code=429, detail=error_msg)
        else:
            raise HTTPException(status_code=503, detail="No LLM backend available")


@router.get("/reading/{text_id}/translations")
def get_translations(
    text_id: int,
    unit: Literal["sentence", "paragraph", "text"],
    target_lang: Optional[str] = None,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    if target_lang is None:
        profile = (
            db.query(Profile)
            .filter(Profile.account_id == account.id, Profile.lang == rt.lang)
            .first()
        )
        target_lang = profile.target_lang if profile else "en"

    rows = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.account_id == account.id,
            ReadingTextTranslation.unit == unit,
            ReadingTextTranslation.target_lang == target_lang,
        )
        .order_by(
            ReadingTextTranslation.segment_index.asc().nullsfirst(),
            ReadingTextTranslation.span_start.asc().nullslast(),
        )
        .all()
    )
    items = [
        {
            "start": r.span_start,
            "end": r.span_end,
            "source": r.source_text,
            "translation": r.translated_text,
        }
        for r in rows
    ]
    return {"unit": unit, "target_lang": target_lang, "items": items}


@router.get("/reading/{text_id}/lookups")
def get_reading_lookups(
    text_id: int,
    target_lang: Optional[str] = None,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    if target_lang is None:
        profile = (
            db.query(Profile)
            .filter(Profile.account_id == account.id, Profile.lang == rt.lang)
            .first()
        )
        target_lang = profile.target_lang if profile else "en"

    rows = (
        db.query(ReadingLookup)
        .filter(
            ReadingLookup.account_id == account.id,
            ReadingLookup.text_id == text_id,
            ReadingLookup.target_lang == target_lang,
        )
        .order_by(ReadingLookup.span_start.asc())
        .all()
    )
    return [
        {
            "start": r.span_start,
            "end": r.span_end,
            "surface": r.surface,
            "lemma": r.lemma,
            "pos": r.pos,
            "translations": r.translations,
        }
        for r in rows
    ]


class MarkReadIn(BaseModel):
    read: bool


@router.post("/reading/{text_id}/mark_read")
def mark_reading(
    text_id: int,
    payload: MarkReadIn,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")
    now = datetime.utcnow()
    if payload.read:
        rt.is_read = True
        rt.read_at = now
    else:
        rt.is_read = False
        rt.read_at = None
    db.commit()
    return {"ok": True, "is_read": rt.is_read, "read_at": (rt.read_at.isoformat() if rt.read_at else None)}


@router.post("/reading/{text_id}/mark_opened")
def mark_opened(
    text_id: int,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Mark a text as opened (first time user views it)"""
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    # Only set opened_at if it's not already set
    if rt.opened_at is None:
        rt.opened_at = datetime.utcnow()
        db.commit()

        # Check if we should automatically generate a new text
        from ..services.llm_service import should_generate_new_text
        if should_generate_new_text(db, account.id, rt.lang):
            # Trigger automatic generation
            from ..services.llm_service import generate_reading
            try:
                # Generate a new text with default parameters
                result = generate_reading(
                    db,
                    account_id=account.id,
                    lang=rt.lang,
                    length=None,
                    include_words=None,
                    model=None,
                    provider="openrouter",
                    base_url="http://localhost:1234/v1"
                )
                # Return HTML for the new text that was generated
                from fastapi.responses import HTMLResponse
                new_text_html = f'''
                <div id="current-reading" class="border border-gray-200 rounded-lg p-4 bg-gray-50">
                  <div class="prose max-w-none">
                    {result.get("text", "")}
                  </div>
                  <div class="mt-4 flex gap-2">
                    <button
                      hx-post="/reading/{result.get('text_id')}/mark_opened"
                      hx-target="#current-reading"
                      hx-swap="outerHTML"
                      class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      Mark as Read
                    </button>
                    <button
                      hx-get="/reading/{result.get('text_id')}/words"
                      hx-target="#current-reading"
                      hx-swap="innerHTML"
                      class="border border-gray-300 hover:bg-gray-50 px-4 py-2 rounded-lg transition-colors"
                    >
                      View Words
                    </button>
                  </div>
                </div>
                '''
                return HTMLResponse(content=new_text_html)
            except Exception as e:
                # Log error but don't fail the mark_opened operation
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to auto-generate text after mark_opened: {e}")

    # Return a simple success message when no new text is generated
    from fastapi.responses import HTMLResponse
    success_html = '''
    <div class="text-center py-8">
      <p class="text-gray-500">Text marked as read.</p>
      <p class="text-sm text-gray-400 mt-2">A new text will be generated automatically when available.</p>
    </div>
    '''
    return HTMLResponse(content=success_html)


@router.get("/reading/{text_id}/words")
def get_reading_words(
    text_id: int,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    rows = (
        db.query(ReadingWordGloss)
        .filter(
            ReadingWordGloss.account_id == account.id,
            ReadingWordGloss.text_id == text_id,
        )
        .order_by(ReadingWordGloss.span_start, ReadingWordGloss.span_end)
        .all()
    )

    return {
        "text_id": text_id,
        "words": [
            {
                "surface": w.surface,
                "lemma": w.lemma,
                "pos": w.pos,
                "pinyin": w.pinyin,
                "translation": w.translation,
                "lemma_translation": w.lemma_translation,
                "grammar": w.grammar,
                "span_start": w.span_start,
                "span_end": w.span_end,
            }
            for w in rows
        ],
    }


@router.get("/reading/{text_id}/meta")
def get_reading_meta(
    text_id: int,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")
    return {
        "text_id": rt.id,
        "lang": rt.lang,
        "is_read": bool(getattr(rt, "is_read", False)),
        "read_at": (rt.read_at.isoformat() if getattr(rt, "read_at", None) else None),
        "created_at": (rt.created_at.isoformat() if getattr(rt, "created_at", None) else None),
    }

