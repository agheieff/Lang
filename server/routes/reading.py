from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, RedirectResponse
import json
import time
import asyncio
import os

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
from ..services.gen_queue import ensure_text_available
from ..utils.gloss import reconstruct_glosses_from_logs


router = APIRouter(tags=["reading"])

MAX_WAIT_SEC = 25.0
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
        '<div class="mt-4 flex gap-2">'
        '  <button'
        '    hx-post="/reading/next"'
        '    hx-target="#current-reading"'
        '    hx-swap="innerHTML"'
        '    class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"'
        '  >'
        '    Next text'
        '  </button>'
        '</div>'
        f'<script id="reading-words-json" type="application/json">{json.dumps(words_json, ensure_ascii=False)}</script>'
        '<div id="word-tooltip" class="hidden absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm max-w-xs"></div>'
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

@router.post("/reading/next")
async def next_text(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        raise HTTPException(400, "profile not found")

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

