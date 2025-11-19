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
    """Return the Current Reading block HTML.

    Uses the new generation orchestrator for cleaner state management.
    Will show content immediately available and rely on SSE for updates.
    """
    import time
    start_time = time.time()
    logger.info(f"[READING] current_reading_block called for account_id={account.id}")
    
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return HTMLResponse(
            content=render_loading_block("loading")
        )

    orchestrator = GenerationOrchestrator()
    state_manager = orchestrator.state_manager
    
    # Use SelectionService to get current text (current_text_id or pick new)
    from ..services.selection_service import SelectionService
    selection_service = SelectionService()
    
    # Get current text (either the one in profile.current_text_id or pick unopened)
    text_obj = None
    selection_start = time.time()
    try:
        text_obj = selection_service.pick_current_or_new(db, account.id, prof.lang)
        logger.info(f"[READING] SelectionService.pick_current_or_new took {time.time() - selection_start:.3f}s")
    except Exception as e:
        logger.error(f"SelectionService.pick_current_or_new failed: {e}", exc_info=True)
        # Fallback: try to get any available text
        try:
            text_obj = state_manager.get_unopened_text(db, account.id, prof.lang)
        except Exception as e2:
            logger.error(f"Fallback get_unopened_text also failed: {e2}", exc_info=True)
            # Return error page
            return HTMLResponse(
                content='''
                <div id="current-reading" class="text-center py-8">
                  <p class="text-red-500">Error loading text. Please refresh the page.</p>
                </div>
                ''', status=500
            )
    
    # Ensure something is queued (prefetch next text)
    ensure_start = time.time()
    try:
        orchestrator.ensure_text_available(db, account.id, prof.lang)
        logger.info(f"[READING] ensure_text_available took {time.time() - ensure_start:.3f}s")
    except Exception:
        logger.debug("ensure_text_available failed in current_reading_block", exc_info=True)

    if text_obj is None:
        # Check if something is generating
        generating = state_manager.get_generating_text(db, account.id, prof.lang)
        kind = "generating" if generating else "loading"
        return HTMLResponse(content=render_loading_block(kind))

    # SelectionService already set current_text_id if needed
    # Ensure next text is available 
    try:
        orchestrator.ensure_text_available(db, account.id, prof.lang)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    # Render the text content
    text_id = text_obj.id
    
    # Extract title and title words via service
    title_service = TitleExtractionService()
    raw_title, title_translation = title_service.get_title(db, account.id, text_id)
    title_words_list = title_service.get_title_words(db, account.id, text_id)

    # Include words immediately if available
    rows = (
        db.query(ReadingWordGloss)
        .filter(ReadingWordGloss.account_id == account.id, ReadingWordGloss.text_id == text_id)
        .order_by(ReadingWordGloss.span_start.asc().nullsfirst(), ReadingWordGloss.span_end.asc().nullsfirst())
        .all()
    )
    
    # Check if translations are ready via readiness service (single oracle)
    try:
        _ready = ReadinessService()
        is_fully_ready, _reason = _ready.evaluate(db, text_obj, account.id)
    except Exception:
        is_fully_ready = (len(rows) > 0)
    
    # Add SSE endpoint and state to help client
    sse_endpoint = f"/reading/events/sse?text_id={text_id}"
    
    # Render with SSE connection info
    inner = render_reading_block(
        text_id,
        text_obj.content or "",
        rows,
        title=raw_title,
        title_words=title_words_list,
        title_translation=title_translation if isinstance(title_translation, str) else None,
    )
    
    # Wrap with SSE metadata for the client
    content_with_sse = f'''
        <div id="current-reading"
             data-sse-endpoint="{sse_endpoint}"
             data-text-id="{text_id}"
             data-is-ready="false">
            {inner}
            <script id="reading-seeds" type="application/json">
                {{
                    "sse_endpoint": "{sse_endpoint}",
                    "text_id": {text_id},
                    "ready": {str(is_fully_ready).lower()},
                    "account_id": {account.id}
                }}
            </script>
        </div>
    '''
    
    total_time = time.time() - start_time if 'start_time' in locals() else 0
    logger.info(f"[READING] current_reading_block completed in {total_time:.3f}s")
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
        from ..services.session_processing_service import SessionProcessingService
        session_service = SessionProcessingService()
        session_service.process_session_data(db, account.id, prof.current_text_id or 0, session_data)

    # Use SelectionService for proper text management
    from ..services.selection_service import SelectionService
    selection_service = SelectionService()
    
    # Mark current text as read before moving to next
    if prof.current_text_id:
        from ..services.state_manager import GenerationStateManager
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

    _ready = ReadinessService()
    _recon = ReconstructionService()
    _gen = GenerationOrchestrator()

    def _next_unopened() -> Optional[ReadingText]:
        return _ready.next_unopened(db, account.id, prof.lang)

    # Ensure something is queued

    # Check and retry failed texts before queuing new ones
    try:
        retry_results = _gen.check_and_retry_failed_texts(db, account.id, prof.lang)
    except Exception:
        logger.debug("retry_failed_texts failed", exc_info=True)
    
    # Ensure something is queued
    try:
        _gen.ensure_text_available(db, account.id, prof.lang)
    except Exception:
        logger.debug("ensure_text_available failed", exc_info=True)

    rt = _next_unopened()
    deadline = time.time() + min(float(wait or 0), MAX_WAIT_SEC) if wait and wait > 0 else None

    while True:
        # Refresh transaction to see background thread commits
        try:
            db.rollback()
        except Exception:
            logger.debug("rollback failed at loop head in next_ready", exc_info=True)
        if rt:
            # Persisted manual override: consume if valid and content exists
            if getattr(rt, "content", None) and _ready.consume_if_valid(db, account.id, prof.lang):
                return {"ready": True, "text_id": rt.id, "ready_reason": "manual_override"}
            if force:
                # Persist a one-shot override; only return ready if content exists
                try:
                    _ready.force_once(db, account.id, prof.lang)
                except Exception:
                    logger.debug("force_once failed", exc_info=True)
                if getattr(rt, "content", None):
                    return {"ready": True, "text_id": rt.id, "ready_reason": "manual"}
            ready, reason = _ready.evaluate(db, rt, account.id)
            
            # Check if text needs retry
            needs_retry = False
            retry_status = None
            try:
                failed_components = _ready.get_failed_components(db, account.id, rt.id)
                if failed_components["words"] or failed_components["sentences"]:
                    retry_service = _gen.retry_service
                    can_retry, retry_reason = retry_service.can_retry(db, account.id, rt.id, failed_components)
                    needs_retry = can_retry
                    retry_status = {
                        "can_retry": can_retry,
                        "reason": retry_reason,
                        "failed_components": failed_components
                    }
            except Exception:
                logger.debug("retry status check failed", exc_info=True)
            
            # If both signals are present, return immediately
            if ready and reason == "both":
                return {"ready": True, "text_id": rt.id, "ready_reason": reason}
            # If we're in grace (content + one signal) but a wait deadline exists, keep waiting
            # to allow the missing signal to arrive; only return grace once the deadline elapses.
            if ready and reason == "grace":
                if deadline is None or time.time() >= deadline:
                    return {"ready": True, "text_id": rt.id, "ready_reason": reason}
                # else: fall through to wait/retry below
            
            # Return retry information if waiting and retry is available
            if needs_retry and not ready:
                return {
                    "ready": False, 
                    "text_id": rt.id, 
                    "ready_reason": "waiting",
                    "retry_info": retry_status
                }
            
            # Otherwise (not ready yet), continue waiting below until deadline
        if deadline is None or time.time() >= deadline:
            return {"ready": False, "text_id": (rt.id if rt else None)}
        # Try to reconstruct words/sentences from logs if content exists but data is missing
        try:
            if rt and getattr(rt, "content", None):
                if not _ready._has_words(db, account.id, rt.id):
                    _recon.ensure_words_from_logs(db, account.id, rt.id, text=rt.content, lang=rt.lang)
                if not _ready._has_sentences(db, account.id, rt.id):
                    _recon.ensure_sentence_translations_from_logs(db, account.id, rt.id)
        except Exception:
            logger.debug("reconstruction attempts failed in next_ready", exc_info=True)
        await _tick(db, 0.5)
        # Start a fresh view before selecting next candidate
        try:
            db.rollback()
        except Exception:
            logger.debug("rollback failed before selecting next in next_ready", exc_info=True)
        rt = _next_unopened()


@router.get("/reading/next/ready/sse")
async def next_ready_sse(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Server-Sent Events endpoint for next text readiness notifications."""
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return Response(content="data: {\"ready\": false, \"text_id\": null}\n\n", 
                       media_type="text/event-stream", 
                       headers={"Cache-Control": "no-cache", 
                               "Connection": "keep-alive",
                               "Access-Control-Allow-Origin": "*"})

    async def event_stream():
        try:
            # Send initial status
            yield "data: {\"ready\": false, \"text_id\": null, \"status\": \"connecting\"}\n\n"
            
            _ready = ReadinessService()
            _gen = GenerationOrchestrator()
            
            # Check and retry failed texts before queuing new ones
            try:
                _gen.check_and_retry_failed_texts(db, account.id, prof.lang)
            except Exception:
                logger.debug("retry_failed_texts failed", exc_info=True)
            
            # Ensure something is queued
            try:
                _gen.ensure_text_available(db, account.id, prof.lang)
            except Exception:
                logger.debug("ensure_text_available failed", exc_info=True)

            last_status = None
            while True:
                try:
                    # Refresh transaction to see background thread commits
                    try:
                        db.rollback()
                    except Exception:
                        logger.debug("rollback failed in SSE loop", exc_info=True)
                    
                    rt = _ready.next_unopened(db, account.id, prof.lang)
                    
                    if rt:
                        ready, reason = _ready.evaluate(db, rt, account.id)
                        
                        # Check if text needs retry
                        needs_retry = False
                        retry_status = None
                        try:
                            failed_components = _ready.get_failed_components(db, account.id, rt.id)
                            if failed_components["words"] or failed_components["sentences"]:
                                retry_service = _gen.retry_service
                                can_retry, retry_reason = retry_service.can_retry(db, account.id, rt.id, failed_components)
                                needs_retry = can_retry
                                retry_status = {
                                    "can_retry": can_retry,
                                    "reason": retry_reason,
                                    "failed_components": failed_components
                                }
                        except Exception:
                            logger.debug("retry status check failed in SSE", exc_info=True)
                        
                        current_status = {
                            "ready": ready and reason == "both",
                            "text_id": rt.id,
                            "ready_reason": reason if ready else "waiting",
                            "retry_info": retry_status if needs_retry and not ready else None
                        }
                        
                        # Only send update if status changed
                        if current_status != last_status:
                            data = json.dumps(current_status)
                            yield f"data: {data}\n\n"
                            last_status = current_status.copy()
                        
                        # If fully ready, close connection
                        if ready and reason == "both":
                            yield f"data: {{\"ready\": true, \"text_id\": {rt.id}, \"ready_reason\": \"both\", \"status\": \"complete\"}}\n\n"
                            break
                    else:
                        current_status = {"ready": False, "text_id": None, "status": "waiting"}
                        if current_status != last_status:
                            data = json.dumps(current_status)
                            yield f"data: {data}\n\n"
                            last_status = current_status.copy()
                    
                    # Wait before next check (heartbeat)
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {{\"error\": \"stream_error\", \"message\": \"{str(e)}\"}}\n\n"
                    break
        
        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {{\"error\": \"stream_error\", \"message\": \"{str(e)}\"}}\n\n"
        finally:
            logger.debug("SSE connection closed")

    return StreamingResponse(
        content=event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

