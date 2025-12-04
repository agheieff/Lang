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
from ..db import get_global_db
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
from ..schemas.session import TextSessionState
from ..services.user_content_service import UserContentService
from ..services.session_management_service import SessionManagementService
from ..utils.text_segmentation import split_sentences
from ..services.notification_service import get_notification_service
from ..settings import get_settings
from ..views.reading_renderer import render_reading_block, render_loading_block
from ..services.text_orchestration_service import get_text_orchestration_service


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
    global_db: Session = Depends(get_global_db),
    account_db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Return the Current Reading block HTML."""
    session_service = SessionManagementService()
    context = session_service.get_current_reading_context(account_db, global_db, account.id)
    
    if context.status == "error":
         return HTMLResponse(
            content='''
            <div class="text-center py-8">
              <p class="text-red-500">Error loading text. Please refresh the page.</p>
            </div>
            ''', status_code=500
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
        is_next_ready=context.is_next_ready,
        next_ready_reason=context.next_ready_reason,
    )
    
    # Return content for innerHTML swap into #current-reading
    content_with_sse = f'''
        {inner}
        <script id="reading-seeds" type="application/json">
            {{
                "sse_endpoint": "{context.sse_endpoint}",
                "text_id": {context.text_id},
                "ready": {str(context.is_fully_ready).lower()},
                "is_next_ready": {str(context.is_next_ready).lower()},
                "next_ready_reason": "{context.next_ready_reason}",
                "account_id": {account.id}
            }}
        </script>
    '''
    return HTMLResponse(content=content_with_sse)

## Models moved to server.schemas.reading


@router.post("/reading/next")
async def next_text(
    request: Request,
    db: Session = Depends(get_global_db),
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
    
    # Save full session JSON for debugging
    if session_data:
        try:
            import os
            from pathlib import Path
            debug_dir = Path(__file__).parent.parent.parent / "data" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"session_{account.id}_{int(time.time())}.json"
            with open(debug_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            logger.info(f"Saved session JSON to {debug_file}")
        except Exception as e:
            logger.warning(f"Failed to save debug session JSON: {e}")
    
    # Process session data in background (don't block the redirect)
    if session_data:
        import threading
        current_text_id = prof.current_text_id or 0
        account_id = account.id
        
        def _process_session():
            from ..services.session_processing_service import SessionProcessingService
            from ..account_db import open_account_session
            bg_db = None
            try:
                bg_db = open_account_session(account_id)
                session_service = SessionProcessingService()
                session_service.process_session_data(bg_db, account_id, current_text_id, session_data)
            except Exception as e:
                logger.warning(f"Background session processing failed: {e}")
            finally:
                if bg_db:
                    try:
                        bg_db.close()
                    except Exception:
                        pass
        
        thread = threading.Thread(target=_process_session, daemon=True)
        thread.start()
        
        # Handle length preference adjustment (sync, it's quick)
        length_pref = session_data.get("length_preference")
        if length_pref in ("longer", "shorter"):
            current_length = prof.text_length or 300  # Default to 300
            if length_pref == "longer":
                new_length = int(current_length * 1.15)  # +15%
            else:
                new_length = int(current_length * 0.85)  # -15%
            # Clamp to reasonable bounds
            prof.text_length = max(100, min(2000, new_length))
            db.commit()

    # Use SelectionService for proper text management
    from ..services.selection_service import SelectionService
    selection_service = SelectionService()
    
    # Mark current text as read and clear current_text_id before moving to next
    if prof.current_text_id:
        from ..services.state_manager import GenerationStateManager
        state_manager = GenerationStateManager()
        try:
            state_manager.mark_read(db, account.id, prof.current_text_id)
        except Exception:
            logger.debug("Failed to mark text as read", exc_info=True)
        # Clear current_text_id so pick_current_or_new selects a new text
        prof.current_text_id = None
        db.commit()
    
    # Pick next text and set it as current (this updates current_text_id and marks opened)
    next_text = selection_service.pick_current_or_new(db, account.id, prof.lang)

    # Respond JSON to programmatic clients; keep redirect for regular form posts
    # If HTMX request, send a redirect so htmx follows with GET and swaps into target
    if request.headers.get("hx-request", "").lower() == "true":
        return RedirectResponse(url="/reading/current", status_code=303)
    if request.headers.get("accept", "").lower().find("application/json") >= 0 or request.headers.get("content-type", "").lower().startswith("application/json"):
        return {"ok": True}
    # For form submissions, redirect to home page (full page with layout)
    return RedirectResponse(url="/", status_code=303)


@router.get("/reading/events/sse")
async def reading_events_sse(
    db: Session = Depends(get_global_db),
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


@router.post("/reading/sync")
async def sync_session_state(
    state: TextSessionState,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Sync current reading session state from client."""
    session_service = SessionManagementService()
    success = session_service.persist_session_state(db, account.id, state)
    return {"ok": success}

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
    if not rt:
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
                    ReadingTextTranslation.text_id == text_id,
                    ReadingTextTranslation.target_lang == target_lang,
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
                            ReadingWordGloss.text_id == text_id,
                            ReadingWordGloss.target_lang == target_lang,
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
                    session_service = SessionManagementService()
                    session_service.ensure_words_from_logs(db, account.id, text_id, text=rt.content, lang=rt.lang)
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
    if not rt:
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
    if not rt:
        raise HTTPException(404, "reading text not found")

    # Use text's target_lang for filtering
    target_lang = rt.target_lang or "en"
    
    def _load_rows():
        return (
            db.query(ReadingWordGloss)
            .filter(
                ReadingWordGloss.text_id == text_id,
                ReadingWordGloss.target_lang == target_lang,
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
                    session_service = SessionManagementService()
                    session_service.ensure_words_from_logs(db, account.id, text_id, text=rt.content, lang=rt.lang)
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
    if not rt:
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
    global_db: Session = Depends(get_global_db),
    account_db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    user_content_service = UserContentService()
    
    prof = account_db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return {"ready": False, "text_id": None}

    deadline = time.time() + min(float(wait or 0), MAX_WAIT_SEC) if wait and wait > 0 else None

    while True:
        try:
            account_db.rollback()
        except Exception:
            logger.debug("rollback failed in next_ready", exc_info=True)
            
        status = user_content_service.check_next_ready(global_db, account.id, prof.lang)
        
        if status.ready:
             return {"ready": True, "text_id": status.text_id, "ready_reason": status.reason}
        
        # Handle force mode differently in the consolidated approach
        if force:
            pass  # Force handling would need to be reinvented in UserContentService

        if deadline is None or time.time() >= deadline:
            return {"ready": False, "text_id": status.text_id}
            
        await _tick(global_db, 0.5)


@router.get("/reading/next/ready/sse")
async def next_ready_sse(
    global_db: Session = Depends(get_global_db),
    account_db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Server-Sent Events endpoint for next text readiness notifications."""
    prof = account_db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return Response(content="data: {\"ready\": false, \"text_id\": null}\n\n", 
                       media_type="text/event-stream", 
                       headers={"Cache-Control": "no-cache", 
                               "Connection": "keep-alive",
                               "Access-Control-Allow-Origin": "*"})

    user_content_service = UserContentService()
    
    async def event_stream():
        try:
            # Send initial status
            yield "data: {\"ready\": false, \"text_id\": null, \"status\": \"connecting\"}\n\n"
            
            last_status = None
            
            while True:
                try:
                    try:
                        account_db.rollback()
                    except Exception:
                        pass
                    
                    status = user_content_service.check_next_ready(global_db, account.id, prof.lang)
                    
                    current_status_dict = {
                        "ready": status.ready,
                        "text_id": status.text_id,
                        "ready_reason": status.reason,
                        "retry_info": status.retry_info,
                        "status": status.status
                    }

                    if current_status_dict != last_status:
                         data = json.dumps(current_status_dict)
                         yield f"data: {data}\n\n"
                         last_status = current_status_dict.copy()
                    
                    if status.ready:
                         yield f"data: {{\"ready\": true, \"text_id\": {status.text_id}, \"ready_reason\": \"{status.reason}\", \"status\": \"complete\"}}\n\n"
                         break
                    
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
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/reading/backfill/{text_id}")
async def backfill_translations(
    text_id: int,
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Backfill missing translations for a text."""
    try:
        text_orchestrator = get_text_orchestration_service()
        # Use the integrated backfill functionality from TextOrchestrationService
        from pathlib import Path
        # Look for logs directory for backfill
        log_dir = Path("logs") / "text_generation" / f"text_{text_id}"
        if log_dir.exists():
            # Try to find a log directory
            log_dirs = [d for d in log_dir.iterdir() if d.is_dir()]
            if log_dirs:
                success = text_orchestrator.validate_and_backfill(text_id)
                return {"success": success, "results": {"backfilled": success}}
        
        return {"success": False, "results": {"backfilled": False, "reason": "no_logs_found"}}
    except Exception as e:
        logger.error(f"Backfill failed for text_id={text_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

