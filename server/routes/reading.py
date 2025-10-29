from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Body, Request
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, RedirectResponse
import json
import time
import asyncio
import os
from pydantic import BaseModel

from server.auth import Account  # type: ignore

from ..account_db import get_db
from ..deps import get_current_account as _get_current_account
from ..models import (
    Profile,
    ReadingLookup,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    LLMRequestLog,
)
from ..services.gen_queue import ensure_text_available
from ..utils.gloss import reconstruct_glosses_from_logs


router = APIRouter(tags=["reading"])

MAX_WAIT_SEC = 25.0
# One-shot manual readiness override per (account_id, lang)
_NEXT_READY_MANUAL: set[tuple[int, str]] = set()
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
    # Avoid any leading whitespace before content to keep span offsets aligned with DOM
    return (
        f'<div id="reading-text" class="prose max-w-none" data-text-id="{text_id}">{html_content}</div>'
        '<div class="mt-4 flex items-center gap-3">'
        '  <button id="next-btn"'
        '    hx-post="/reading/next"'
        '    hx-target="#current-reading"'
        '    hx-select="*"'
        '    hx-swap="innerHTML"'
        '    hx-ext="json-enc"'
        '    hx-on--config-request="(function(){try{event.detail.headers=event.detail.headers||{};event.detail.headers[\'Content-Type\']=\'application/json\';event.detail.parameters=(window.arcBuildNextParams&&window.arcBuildNextParams())||{};}catch(e){}})()"'
        '    hx-on--after-request="(function(){try{var t=document.getElementById(\'reading-text\');var id=t&&t.dataset?t.dataset.textId:null;if(id){localStorage.removeItem(\'arc_rl_\'+String(id));}}catch(e){}})()"'
        '    class="px-4 py-2 rounded-lg transition-colors text-white bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"'
        '    disabled'
        '  >Next text</button>'
        '  <span id="next-status" class="ml-3 text-sm text-gray-500">Loading next…</span>'
        '</div>'
        f'<script id="reading-words-json" type="application/json">{json.dumps(words_json, ensure_ascii=False)}</script>'
        '<div id="word-tooltip" class="hidden absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm max-w-xs"></div>'
        '<script>(function(){'
        '  var container=document.getElementById("current-reading");'
        '  var textEl=document.getElementById("reading-text");'
        '  if(!container||!textEl){return;}'
        '  var curId=String(textEl.dataset.textId||"");'
        '  if(container.dataset.nextPollFor===curId){return;}'
        '  container.dataset.nextPollFor=curId;'
        '  try{ container.__lookups=[]; }catch(e){}'
        '  container.dataset.readStartMs=String(Date.now());'
        '  try{'
        '    var __key = "arc_rl_"+curId;'
        '    var __saved = null; try{ __saved = JSON.parse(localStorage.getItem(__key)||"null"); }catch(_e){}'
        '    if(!__saved || typeof __saved !== "object"){ __saved = { started_ms: Date.now(), lookups: [] }; localStorage.setItem(__key, JSON.stringify(__saved)); }'
        '    if(Array.isArray(__saved.lookups) && __saved.lookups.length){ container.__lookups = __saved.lookups.slice(); }'
        '  }catch(_e){}'
        '  var btn=document.getElementById("next-btn");'
        '  var st=document.getElementById("next-status");'
        '  if(container.__nextPollTimer){ try{ clearTimeout(container.__nextPollTimer); }catch(e){} }'
        '  function poll(){'
        '    try {'
        '      fetch("/reading/next/ready?wait=25",{headers:{"Accept":"application/json"}})'
        '        .then(function(res){ if(!res.ok){ return null; } return res.json(); })'
        '        .then(function(data){'
        '          if(!data){ container.__nextPollTimer=setTimeout(poll,1500); return; }'
        '          if(data && data.ready){'
        '            if(data.ready_reason==="both"){'
        '              if(btn){ btn.disabled=false; }'
        '              if(st){ try{ if(st.parentNode){ st.parentNode.removeChild(st); } }catch(e){ st.textContent=""; } }'
        '              container.__nextPollTimer=null; return;'
        '            } else {'
        '              if(btn){ btn.disabled=true; }'
        '              if(st){ st.textContent="Preparing next..."; }'
        '            }'
        '          }'
        '          container.__nextPollTimer=setTimeout(poll,1500);'
        '        })'
        '        .catch(function(){ container.__nextPollTimer=setTimeout(poll,1500); });'
        '    } catch(e) { container.__nextPollTimer=setTimeout(poll,1500); }'
        '  }'
        '  poll();'
        '  window.arcBuildNextParams = function(){'
        '    try{'
        '      var key = "arc_rl_"+curId;'
        '      var saved = null; try{ saved = JSON.parse(localStorage.getItem(key)||"null"); }catch(_e){}'
        '      var started = Number((saved && saved.started_ms) || container.dataset.readStartMs || Date.now());'
        '      var lookups = Array.isArray(saved && saved.lookups) ? saved.lookups.slice() : (Array.isArray(container.__lookups) ? container.__lookups.slice() : []);'
        '      var spent = Math.max(0, Date.now() - started);'
        '      return { text_id: Number(curId||0), time_spent_ms: spent, lookups: lookups };'
        '    }catch(e){ return { text_id: Number(curId||0), time_spent_ms: 0, lookups: [] }; }'
        '  };'
        '})();</script>'
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

    def _pick_or_start() -> Optional[ReadingText]:
        # Prefer current_text_id
        if getattr(prof, "current_text_id", None):
            obj = db.get(ReadingText, prof.current_text_id)
            if obj is not None:
                # If already read, clear pointer and proceed to find another
                if bool(getattr(obj, "is_read", False)):
                    prof.current_text_id = None
                    db.commit()
                    obj = None
                else:
                    if obj.opened_at is None:
                        obj.opened_at = datetime.utcnow()
                        db.commit()
                        try:
                            ensure_text_available(db, account.id, prof.lang)
                        except Exception:
                            pass
                    return obj
        # Else newest unopened
        obj = (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account.id,
                ReadingText.lang == prof.lang,
                ReadingText.opened_at.is_(None),
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )
        if obj is not None:
            prof.current_text_id = obj.id
            if obj.opened_at is None:
                obj.opened_at = datetime.utcnow()
            db.commit()
            try:
                ensure_text_available(db, account.id, prof.lang)
            except Exception:
                pass
            return obj
        # Trigger background generation
        try:
            ensure_text_available(db, account.id, prof.lang)
        except Exception:
            pass
        return None

    # Wait for a text to be available
    deadline = time.time() + MAX_WAIT_SEC
    text_obj = _pick_or_start()
    while text_obj is None and time.time() < deadline:
        await asyncio.sleep(0.5)
        try:
            db.expire_all()
        except Exception:
            pass
        text_obj = _pick_or_start()

    if text_obj is None:
        return HTMLResponse(
            content='''
              <div class="text-center py-8"
                   hx-get="/reading/current"
                   hx-trigger="load, every:2s"
                   hx-swap="innerHTML"
                   hx-target="#current-reading">
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
                   hx-target="#current-reading">
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

class LookupEvent(BaseModel):
    surface: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
    span_start: Optional[int] = None
    span_end: Optional[int] = None


class NextPayload(BaseModel):
    text_id: Optional[int] = None
    time_spent_ms: Optional[int] = 0
    lookups: list[LookupEvent] = []


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

    # Recalc stub: print the received lookups & time for now
    try:
        if req is None:
            # Attempt manual parse if JSON content-type but no body model
            if (request.headers.get("content-type", "").lower().startswith("application/json")):
                try:
                    data = await request.json()
                    req = NextPayload(**data)  # type: ignore[arg-type]
                except Exception:
                    req = None
        if req is not None:
            print("[reading.next] text_id=", req.text_id)
            print("[reading.next] time_spent_ms=", int(req.time_spent_ms or 0))
            try:
                print("[reading.next] lookups=", json.dumps([e.model_dump() for e in (req.lookups or [])], ensure_ascii=False))
            except Exception:
                print("[reading.next] lookups=", str(req.lookups))
    except Exception:
        pass

    # Mark current as read, if any
    if getattr(prof, "current_text_id", None):
        current = db.get(ReadingText, prof.current_text_id)
        if current and current.account_id == account.id:
            now = datetime.utcnow()
            current.is_read = True
            current.read_at = now
            # Do not clear opened_at
            db.commit()

    # Clear pointer and schedule next; delegate selection/rendering to GET /reading/current
    try:
        prof.current_text_id = None
        db.commit()
    except Exception:
        pass
    try:
        ensure_text_available(db, account.id, prof.lang)
    except Exception:
        pass

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
    if (not rows) and wait and wait > 0:
        deadline = time.time() + min(float(wait), MAX_WAIT_SEC)
        while time.time() < deadline and not rows:
            try:
                db.expire_all()
            except Exception:
                pass
            rows = _load_rows()
            if rows:
                break
            await asyncio.sleep(0.5)
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
        deadline = time.time() + min(float(wait), MAX_WAIT_SEC)

        def _try_reconstruct_from_logs() -> None:
            try:
                reconstruct_glosses_from_logs(
                    db,
                    account_id=account.id,
                    text_id=text_id,
                    text=rt.content or "",
                    lang=rt.lang,
                    prefer_db=True,
                )
            except Exception:
                return

        while time.time() < deadline and not rows:
            try:
                db.expire_all()
            except Exception:
                pass
            # try reconstruct once per loop
            _try_reconstruct_from_logs()
            rows = _load_rows()
            if rows:
                break
            await asyncio.sleep(0.5)

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

    def _next_unopened() -> Optional[ReadingText]:
        return (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account.id,
                ReadingText.lang == prof.lang,
                ReadingText.opened_at.is_(None),
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )

    def _has_words(rt: ReadingText) -> bool:
        try:
            return (
                db.query(ReadingWordGloss.id)
                .filter(ReadingWordGloss.account_id == account.id, ReadingWordGloss.text_id == rt.id)
                .first()
                is not None
            )
        except Exception:
            return False

    def _has_sentences(rt: ReadingText) -> bool:
        try:
            return (
                db.query(ReadingTextTranslation.id)
                .filter(
                    ReadingTextTranslation.account_id == account.id,
                    ReadingTextTranslation.text_id == rt.id,
                    ReadingTextTranslation.unit == "sentence",
                )
                .first()
                is not None
            )
        except Exception:
            return False

    def _is_ready(rt: ReadingText) -> tuple[bool, str]:
        if not getattr(rt, "content", None):
            return (False, "no_content")
        hw = _has_words(rt)
        hs = _has_sentences(rt)
        if hw and hs:
            return (True, "both")
        # Grace fallback
        try:
            from datetime import datetime as _dt
            import os as _os
            grace = float(_os.getenv("ARC_NEXT_READY_GRACE_SEC", "20"))
            if getattr(rt, "generated_at", None):
                age = (_dt.utcnow() - rt.generated_at).total_seconds()
                if age >= grace and (hw or hs):
                    return (True, "grace")
        except Exception:
            pass
        return (False, "waiting")

    def _reconstruct_sentences(rt: ReadingText) -> None:
        row = (
            db.query(LLMRequestLog)
            .filter(
                LLMRequestLog.account_id == account.id,
                LLMRequestLog.text_id == rt.id,
                LLMRequestLog.kind == "structured_translation",
                LLMRequestLog.status == "ok",
            )
            .order_by(LLMRequestLog.created_at.desc())
            .first()
        )
        if not row or not row.response:
            return
        payload = row.response
        try:
            blob = json.loads(payload)
        except Exception:
            blob = payload
        content = None
        if isinstance(blob, dict):
            try:
                ch = blob.get("choices")
                if isinstance(ch, list) and ch:
                    msg = ch[0].get("message") if isinstance(ch[0], dict) else None
                    if isinstance(msg, dict):
                        content = msg.get("content")
            except Exception:
                content = None
            if content is None and isinstance(blob.get("paragraphs"), list):
                parsed = blob
            else:
                from ..utils.json_parser import extract_structured_translation as _extract
                parsed = _extract(str(content or "")) if content is not None else None
        else:
            from ..utils.json_parser import extract_structured_translation as _extract
            parsed = _extract(str(blob))
        if not parsed:
            return
        target_lang = None
        try:
            target_lang = parsed.get("target_lang")
        except Exception:
            target_lang = None
        if not target_lang:
            prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == rt.lang).first()
            target_lang = prof.target_lang if prof else "en"
        # Insert missing sentence rows with running segment_index
        try:
            existing_idx = set(
                i for (i,) in db.query(ReadingTextTranslation.segment_index)
                .filter(
                    ReadingTextTranslation.account_id == account.id,
                    ReadingTextTranslation.text_id == rt.id,
                    ReadingTextTranslation.unit == "sentence",
                    ReadingTextTranslation.target_lang == target_lang,
                ).all()
            )
        except Exception:
            existing_idx = set()
        idx = 0
        try:
            for p in parsed.get("paragraphs", []):
                for s in p.get("sentences", []):
                    if not ("text" in s and "translation" in s):
                        continue
                    if idx in existing_idx:
                        idx += 1
                        continue
                    db.add(
                        ReadingTextTranslation(
                            account_id=account.id,
                            text_id=rt.id,
                            unit="sentence",
                            target_lang=target_lang,
                            segment_index=idx,
                            span_start=None,
                            span_end=None,
                            source_text=s["text"],
                            translated_text=s["translation"],
                            provider=None,
                            model=None,
                        )
                    )
                    idx += 1
            try:
                db.commit()
            except Exception:
                db.rollback()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

    # Ensure something is queued
    try:
        ensure_text_available(db, account.id, prof.lang)
    except Exception:
        pass

    rt = _next_unopened()
    deadline = time.time() + min(float(wait or 0), MAX_WAIT_SEC) if wait and wait > 0 else None

    while True:
        if rt:
            key = (int(account.id), str(prof.lang))
            # If a manual override was set earlier (via ?force=1), consume it and allow readiness
            if key in _NEXT_READY_MANUAL and getattr(rt, "content", None):
                try:
                    _NEXT_READY_MANUAL.discard(key)
                except Exception:
                    pass
                return {"ready": True, "text_id": rt.id, "ready_reason": "manual_override"}
            if force:
                # Set a one-shot override so subsequent normal polls will be ready
                if getattr(rt, "content", None):
                    try:
                        _NEXT_READY_MANUAL.add(key)
                    except Exception:
                        pass
                return {"ready": True, "text_id": rt.id, "ready_reason": "manual"}
            ready, reason = _is_ready(rt)
            if ready:
                return {"ready": True, "text_id": rt.id, "ready_reason": reason}
        if deadline is None or time.time() >= deadline:
            return {"ready": False, "text_id": (rt.id if rt else None)}
        # Try to reconstruct words if content exists but words are missing
        try:
            if rt and getattr(rt, "content", None):
                if not _has_words(rt):
                    reconstruct_glosses_from_logs(db, account_id=account.id, text_id=rt.id, text=rt.content or "", lang=rt.lang, prefer_db=True)
                if not _has_sentences(rt):
                    _reconstruct_sentences(rt)
        except Exception:
            pass
        try:
            db.expire_all()
        except Exception:
            pass
        await asyncio.sleep(0.5)
        rt = _next_unopened()

