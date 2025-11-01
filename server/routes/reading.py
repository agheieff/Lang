from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, RedirectResponse
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
from ..schemas.reading import LookupEvent, NextPayload
from ..services.selection_service import SelectionService
from ..services.readiness_service import ReadinessService
from ..services.reconstruction_service import ReconstructionService
from ..services.progress_service import ProgressService
from ..services.generation_orchestrator import GenerationOrchestrator
from ..settings import get_settings


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

def _safe_html(text: Optional[str]) -> str:
    """Escape untrusted text for safe HTML display. Preserve newlines as <br>.

    Also normalizes CRLF/CR to LF before processing to keep offsets consistent.
    """
    if not text:
        return ""
    try:
        norm = str(text).replace("\r\n", "\n").replace("\r", "\n")
        from markupsafe import escape  # type: ignore
        esc = str(escape(norm))
    except Exception:
        # Minimal escape fallback
        norm = str(text).replace("\r\n", "\n").replace("\r", "\n")
        esc = (
            norm
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    return esc.replace("\n", "<br>")


def _words_json(rows):
    return [
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
    ]


def _render_reading_block(text_id: int, html_content: str, words_rows) -> str:
    words_json = _words_json(words_rows)
    json_text = json.dumps(words_json, ensure_ascii=False).replace('</', '<\\/')
    # Avoid any leading whitespace before content to keep span offsets aligned with DOM
    return (
        '<div id="reading-block">'
        f'<div id="reading-text" class="prose max-w-none" data-text-id="{text_id}">{html_content}</div>'
        '<div class="mt-4 flex items-end w-full">'
        '  <div class="flex items-center gap-3 flex-1">'
        '    <button id="next-btn"'
        '      hx-post="/reading/next"'
        '      hx-target="#current-reading"'
        '      hx-select="#reading-block"'
        '      hx-swap="innerHTML"'
        '      hx-on--config-request="(function(){try{var p=(window.arcBuildNextParams&&window.arcBuildNextParams())||{};event.detail.headers=event.detail.headers||{};event.detail.headers[\'Content-Type\']=\'application/json\';event.detail.body=JSON.stringify(p);}catch(e){}})()"'
        '      class="px-4 py-2 rounded-lg transition-colors text-white bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"'
        '      disabled aria-disabled="true">Next text</button>'
        '    <span id="next-status" class="ml-3 text-sm text-gray-500" aria-live="polite">Loading next…</span>'
        '  </div>'
        '  <button id="see-translation-btn" type="button"'
        '    class="ml-4 shrink-0 px-3 py-1.5 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50"'
        '    onclick="window.arcToggleTranslation && window.arcToggleTranslation(event)"'
        '    aria-expanded="false">See translation</button>'
        '</div>'
        '<div id="translation-panel" class="hidden mt-4" hidden>'
        '  <div id="translation-content" class="prose max-w-none text-gray-800">'
        '    <hr class="my-3">'
        '    <div id="translation-text" class="whitespace-pre-wrap"></div>'
        '  </div>'
        '</div>'
        f'<script id="reading-words-json" type="application/json">{json_text}</script>'
        '<div id="word-tooltip" class="hidden absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm max-w-xs"></div>'
        '<script src="/static/reading.js" defer></script>'
        '</div>'
    )


 


@router.get("/reading/current", response_class=HTMLResponse)
async def current_reading_block(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Return the Current Reading block HTML.

    Long-polls up to MAX_WAIT_SEC for a ready text; when found, returns HTML with embedded words JSON.
    """
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        return HTMLResponse(
            content='''
            <div id="current-reading" class="text-center py-8">
              <p class="text-gray-500">No profile found yet.</p>
              <a href="/profile" class="text-blue-600 underline">Create a profile</a>
            </div>
            '''
        )

    _sel = SelectionService()
    # Ensure something is queued or in progress (non-blocking; respects unopened/pending requests)
    try:
        GenerationOrchestrator().ensure_text_available(db, account.id, prof.lang)
    except Exception:
        logger.debug("ensure_text_available failed in current_reading_block", exc_info=True)

    def _pick_or_start() -> Optional[ReadingText]:
        return _sel.pick_current_or_new(db, account.id, prof.lang)

    # Wait for a text to be available
    text_obj = _pick_or_start()
    if text_obj is None:
        text_obj = await wait_until(_pick_or_start, MAX_WAIT_SEC, db)

    if text_obj is None:
        return HTMLResponse(
            content='''
              <div class="text-center py-8"
                   hx-get="/reading/current"
                   hx-trigger="load, every:2s"
                   hx-swap="innerHTML"
                   hx-target="#current-reading"
                   hx-select="#reading-block">
                <div class="animate-pulse space-y-3">
                  <div class="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div class="h-4 bg-gray-200 rounded w-5/6"></div>
                  <div class="h-4 bg-gray-200 rounded w-2/3"></div>
                  <div class="h-4 bg-gray-200 rounded w-4/5"></div>
                </div>
                <div class="mt-2 text-sm text-gray-500">Loading text…</div>
              </div>
            '''
        )

    # If text is still generating, show placeholder that auto-reloads
    if not getattr(text_obj, "content", None):
        return HTMLResponse(
            content='''
              <div class="text-center py-8"
                   hx-get="/reading/current"
                   hx-trigger="load, every:2s"
                   hx-swap="innerHTML"
                   hx-target="#current-reading"
                   hx-select="#reading-block">
                <div class="animate-pulse space-y-3">
                  <div class="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div class="h-4 bg-gray-200 rounded w-5/6"></div>
                  <div class="h-4 bg-gray-200 rounded w-2/3"></div>
                  <div class="h-4 bg-gray-200 rounded w-4/5"></div>
                </div>
                <div class="mt-2 text-sm text-gray-500">Generating…</div>
              </div>
            '''
        )

    # Mark as opened on first render and pre-queue the next if none is in flight
    try:
        if getattr(text_obj, "opened_at", None) is None:
            text_obj.opened_at = datetime.utcnow()
            db.commit()
            try:
                GenerationOrchestrator().ensure_text_available(db, account.id, prof.lang)
            except Exception:
                logger.debug("ensure_text_available failed post-open in current_reading_block", exc_info=True)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    text_id = text_obj.id
    text_html = _safe_html(text_obj.content)

    # Include words immediately if available; otherwise the client will subscribe via SSE and update when ready
    rows = (
        db.query(ReadingWordGloss)
        .filter(ReadingWordGloss.account_id == account.id, ReadingWordGloss.text_id == text_id)
        .order_by(ReadingWordGloss.span_start.asc().nullsfirst(), ReadingWordGloss.span_end.asc().nullsfirst())
        .all()
    )
    inner = _render_reading_block(text_id, text_html, rows)
    return HTMLResponse(content=inner)

## Models moved to server.schemas.reading


@router.post("/reading/next")
async def next_text(
    request: Request,
    req: Optional[NextPayload] = Body(default=None),
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        raise HTTPException(400, "profile not found")

    # Parse JSON manually if needed
    try:
        if req is None and (request.headers.get("content-type", "").lower().startswith("application/json")):
            try:
                data = await request.json()
                req = NextPayload(**data)  # type: ignore[arg-type]
            except Exception:
                req = None
    except Exception:
        req = req

    # Record analytics (Phase 0: structured log only)
    ProgressService().record_session(db, account.id, req)

    # Mark current as read
    ProgressService().complete_and_mark_read(db, account.id, getattr(prof, "current_text_id", None))

    # Clear pointer and schedule next; delegate selection/rendering to GET /reading/current
    try:
        prof.current_text_id = None
        db.commit()
    except Exception:
        logger.debug("failed to clear current_text_id or commit in next_text", exc_info=True)
    try:
        GenerationOrchestrator().ensure_text_available(db, account.id, prof.lang)
    except Exception:
        logger.debug("ensure_text_available failed in next_text", exc_info=True)

    # Respond JSON to programmatic clients; keep redirect for regular form posts
    # If HTMX request, send a redirect so htmx follows with GET and swaps into target
    if request.headers.get("hx-request", "").lower() == "true":
        return RedirectResponse(url="/reading/current", status_code=303)
    if request.headers.get("accept", "").lower().find("application/json") >= 0 or request.headers.get("content-type", "").lower().startswith("application/json"):
        return {"ok": True}
    return RedirectResponse(url="/reading/current", status_code=303)


## SSE endpoint removed in favor of async long-polling


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
    # Best-effort span reconstruction for legacy rows without spans (do not write back; return only)
    sent_spans: list[tuple[int, int, str]] = []
    if unit == "sentence" and any(getattr(r, "span_start", None) is None or getattr(r, "span_end", None) is None for r in rows):
        def _split_sentences(text: str, lang: str):
            if not text:
                return []
            import re as _re
            if str(lang).startswith("zh"):
                pattern = r"[^。！？!?…]+(?:[。！？!?…]+|$)"
            else:
                pattern = r"[^\.!?]+(?:[\.!?]+|$)"
            out = []
            for m in _re.finditer(pattern, text):
                s, e = m.span()
                seg = (text or "")[s:e]
                if seg and seg.strip():
                    out.append((s, e, seg))
            return out
        sent_spans = _split_sentences(rt.content or "", rt.lang)

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
            # If both signals are present, return immediately
            if ready and reason == "both":
                return {"ready": True, "text_id": rt.id, "ready_reason": reason}
            # If we're in grace (content + one signal) but a wait deadline exists, keep waiting
            # to allow the missing signal to arrive; only return grace once the deadline elapses.
            if ready and reason == "grace":
                if deadline is None or time.time() >= deadline:
                    return {"ready": True, "text_id": rt.id, "ready_reason": reason}
                # else: fall through to wait/retry below
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

