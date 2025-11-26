from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
import json
import time
import asyncio
import logging

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
from ..models import LLMRequestLog
from ..utils.json_parser import extract_json_from_text, extract_word_translations
from ..schemas.reading import LookupEvent, NextPayload
from ..services.selection_service import SelectionService
from ..services.readiness_service import ReadinessService
from ..services.reconstruction_service import ReconstructionService
from ..services.progress_service import ProgressService
from ..services.generation_orchestrator import GenerationOrchestrator
from ..services.reading_view_service import ReadingViewService
from ..services.session_processing_service import SessionProcessingService
from ..services.state_manager import GenerationStateManager
from ..services.translation_backfill_service import TranslationBackfillService
from ..utils.text_segmentation import split_sentences
from ..services.notification_service import get_notification_service
from ..settings import get_settings
from ..services.title_extraction_service import TitleExtractionService
from ..views.reading_renderer import render_reading_block, render_loading_block
from ..services.translation_service import TranslationService


router = APIRouter(tags=["reading"])
logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
MAX_WAIT_SEC = float(_SETTINGS.NEXT_READY_MAX_WAIT_SEC)
# Deprecated manual override memory set removed; persisted overrides used instead

async def _tick(db: Session, interval: float = 0.5) -> None:
    """Rollback, expire, and sleep to advance long‑poll loops safely."""
    try:
        db.rollback()
    except Exception:
        logger.debug("rollback failed in _tick", exc_info=True)
    try:
        db.expire_all()
    except Exception:
        logger.debug("expire_all failed in _tick", exc_info=True)
    await asyncio.sleep(interval)


async def wait_until(pred, timeout: float, db: Session, interval: float = 0.5):
    """Poll pred() until it returns a truthy value or timeout elapses.

    Returns the last pred() value (truthy or falsy) at timeout.
    """
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        val = pred()
        if val:
            return val
        await _tick(db, interval)
    return pred()

## View helpers moved to server.views.reading_renderer


 


@router.get("/reading/current", response_class=HTMLResponse)
async def current_reading_block(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Return the Current Reading block HTML."""
    view_service = ReadingViewService()
    context = view_service.get_current_reading_context(db, account.id)
    
    if context.status == "error":
         return HTMLResponse(
            content='''
            <div id="current-reading" class="text-center py-8">
              <p class="text-red-500">Error loading text. Please refresh the page.</p>
            </div>
            ''', status=500
        )

    if context.status in ("loading", "generating"):
        return HTMLResponse(content=render_loading_block(context.status))

    # Render the reading view
    inner = render_reading_block(
        context.text_id,
        context.content,
        context.words,
        title=context.title,
        title_words=context.title_words,
        title_translation=context.title_translation,
    )
    
    # Wrap with SSE metadata for the client
    content_with_sse = f'''
        <div id="current-reading"
             data-sse-endpoint="{context.sse_endpoint}"
             data-text-id="{context.text_id}"
             data-is-ready="{str(context.is_fully_ready).lower()}">
            {inner}
            <script id="reading-seeds" type="application/json">
                {{
                    "sse_endpoint": "{context.sse_endpoint}",
                    "text_id": {context.text_id},
                    "ready": {str(context.is_fully_ready).lower()},
                    "account_id": {account.id}
                }}
            </script>
        </div>
    '''
    return HTMLResponse(content=content_with_sse)

## Models moved to server.schemas.reading


@router.post("/reading/next")
async def next_text(
    request: Request,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        raise HTTPException(400, "profile not found")

    # Get session data
    session_data = None
    try:
        content_type = request.headers.get("content-type", "").lower()
        
        # Try to get from form first
        if content_type.startswith("application/x-www-form-urlencoded"):
            form_data = await request.form()
            session_data_str = form_data.get("session_data")
            if session_data_str:
                session_data = json.loads(session_data_str)
        
        # If no form data, try JSON
        elif content_type.startswith("application/json"):
            try:
                data = await request.json()
                session_data = data
            except Exception as e:
                logger.warning(f"Failed to parse JSON data: {e}")
                
    except Exception as e:
        logger.warning(f"Failed to parse request data: {e}")
    
    # Process session data whether from form or JSON
    if session_data:
        session_service = SessionProcessingService()
        session_service.process_session_data(db, account.id, prof.current_text_id or 0, session_data)

    # Use SelectionService for proper text management
    selection_service = SelectionService()
    
    # Mark current text as read before moving to next
    if prof.current_text_id:
        state_manager = GenerationStateManager()
        try:
            state_manager.mark_read(db, account.id, prof.current_text_id)
        except Exception:
            logger.debug("Failed to mark text as read", exc_info=True)
    
    # Pick next text and set it as current (this updates current_text_id and marks opened)
    next_text = selection_service.pick_current_or_new(db, account.id, prof.lang)
    
    # Ensure next text is available after picking
    try:
        orchestrator = GenerationOrchestrator()
        orchestrator.ensure_text_available(db, account.id, prof.lang)
    except Exception:
        logger.debug("ensure_text_available failed in next_text", exc_info=True)

    # Respond JSON to programmatic clients; keep redirect for regular form posts
    # If HTMX request, send a redirect so htmx follows with GET and swaps into target
    if request.headers.get("hx-request", "").lower() == "true":
        return RedirectResponse(url="/reading/current", status_code=303)
    if request.headers.get("accept", "").lower().find("application/json") >= 0 or request.headers.get("content-type", "").lower().startswith("application/json"):
        return {"ok": True}
    # For form submissions, always redirect
    return RedirectResponse(url="/reading/current", status_code=303)


@router.get("/reading/events/sse")
async def reading_events_sse(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """
    SSE endpoint for real-time updates about reading events.
    Client connects to this endpoint to get notifications about:
    - generation_started
    - content_ready  
    - translations_ready
    - generation_failed
    """
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return Response(content="data: {\"error\": \"No profile found\"}\n\n", 
                       media_type="text/event-stream")
    
    # Get the global notification service
    notification_service = get_notification_service()
    
    # Create SSE stream for this account
    return notification_service.create_sse_stream(account.id, prof.lang)


@router.get("/reading/{text_id}/translations")
async def get_translations(
    text_id: int,
    unit: Literal["sentence", "paragraph", "text"],
    target_lang: Optional[str] = None,
    wait: Optional[float] = None,
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

    def _load_rows():
        return (
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

    rows = _load_rows()

    # If some sentences have no words (e.g., due to earlier JSON glitches), try best-effort reconstruction from logs once.
    try:
        if rows:
            sents = (
                db.query(ReadingTextTranslation.span_start, ReadingTextTranslation.span_end)
                .filter(
                    ReadingTextTranslation.account_id == account.id,
                    ReadingTextTranslation.text_id == text_id,
                    ReadingTextTranslation.unit == "sentence",
                    ReadingTextTranslation.span_start.is_not(None),
                    ReadingTextTranslation.span_end.is_not(None),
                ).all()
            )
            needs = False
            for (ss, ee) in sents:
                try:
                    has = (
                        db.query(ReadingWordGloss.id)
                        .filter(
                            ReadingWordGloss.account_id == account.id,
                            ReadingWordGloss.text_id == text_id,
                            ReadingWordGloss.span_start >= ss,
                            ReadingWordGloss.span_end <= ee,
                        )
                        .first()
                        is not None
                    )
                    if not has:
                        needs = True
                        break
                except Exception:
                    continue
            if needs:
                try:
                    ReconstructionService().ensure_words_from_logs(db, account.id, text_id, text=rt.content, lang=rt.lang)
                    rows = _load_rows()
                except Exception:
                    logger.debug("ensure_words_from_logs failed during translations path", exc_info=True)
    except Exception:
        logger.debug("sentence probe/reconstruction check failed", exc_info=True)
    if (not rows) and wait and wait > 0:
        timeout = min(float(wait), MAX_WAIT_SEC)
        def _pred():
            try:
                r = _load_rows()
                return r if r else None
            except Exception:
                logger.debug("load translations rows failed in poll", exc_info=True)
                return None
        rows = await wait_until(_pred, timeout, db) or []
    # Best-effort span reconstruction for legacy rows without spans (first try to persist spans)
    sent_spans: list[tuple[int, int, str]] = []
    if unit == "sentence" and any(getattr(r, "span_start", None) is None or getattr(r, "span_end", None) is None for r in rows):
        # Try to backfill spans once using the translation service
        try:
            TranslationService().backfill_sentence_spans(db, account.id, text_id)
            rows = _load_rows()
        except Exception:
            logger.debug("backfill_sentence_spans failed; falling back to in-memory spans", exc_info=True)
        # If still missing, compute in-memory spans for response only
        if any(getattr(r, "span_start", None) is None or getattr(r, "span_end", None) is None for r in rows):
            sent_spans = split_sentences(rt.content or "", rt.lang)

    def _row_item(r):
        s0 = r.span_start
        e0 = r.span_end
        if unit == "sentence" and (s0 is None or e0 is None) and r.segment_index is not None:
            try:
                if 0 <= int(r.segment_index) < len(sent_spans):
                    s0, e0, _ = sent_spans[int(r.segment_index)]
            except Exception:
                pass
        return {
            "start": s0,
            "end": e0,
            "source": r.source_text,
            "translation": r.translated_text,
        }

    items = [_row_item(r) for r in rows]
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




@router.get("/reading/{text_id}/words")
async def get_reading_words(
    text_id: int,
    wait: Optional[float] = None,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    rt = db.get(ReadingText, text_id)
    if not rt or rt.account_id != account.id:
        raise HTTPException(404, "reading text not found")

    def _load_rows():
        return (
            db.query(ReadingWordGloss)
            .filter(
                ReadingWordGloss.account_id == account.id,
                ReadingWordGloss.text_id == text_id,
            )
            .order_by(ReadingWordGloss.span_start, ReadingWordGloss.span_end)
            .all()
        )

    rows = _load_rows()

    # Optional long-poll until words are available (no streaming)
    if (not rows) and wait and wait > 0:
        timeout = min(float(wait), MAX_WAIT_SEC)
        reconstructed = False
        def _pred():
            nonlocal reconstructed
            r = _load_rows()
            if r:
                return r
            if (not reconstructed) and getattr(rt, "content", None):
                try:
                    ReconstructionService().ensure_words_from_logs(db, account.id, text_id, text=rt.content, lang=rt.lang)
                except Exception:
                    logger.debug("ensure_words_from_logs failed in words poll", exc_info=True)
                reconstructed = True
            return None
        rows = await wait_until(_pred, timeout, db) or []

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


@router.get("/reading/next/ready")
async def next_ready(
    wait: Optional[float] = None,
    force: Optional[bool] = None,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return {"ready": False, "text_id": None}

    view_service = ReadingViewService()
    deadline = time.time() + min(float(wait or 0), MAX_WAIT_SEC) if wait and wait > 0 else None

    while True:
        try:
            db.rollback()
        except Exception:
            logger.debug("rollback failed in next_ready", exc_info=True)
            
        status = view_service.check_next_text_readiness(db, account.id, prof.lang, force_check=bool(force))
        
        if status.ready:
             return {"ready": True, "text_id": status.text_id, "ready_reason": status.reason}
        
        if status.retry_info:
             return {
                "ready": False, 
                "text_id": status.text_id, 
                "ready_reason": "waiting",
                "retry_info": status.retry_info
            }

        if deadline is None or time.time() >= deadline:
            return {"ready": False, "text_id": status.text_id}
            
        await _tick(db, 0.5)


@router.post("/reading/backfill/{text_id}")
async def backfill_translations(
    text_id: int,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Backfill missing translations for a text."""
    try:
        backfill_service = TranslationBackfillService()
        results = backfill_service.backfill_missing_translations(account.id, text_id)
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Backfill failed for text_id={text_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

