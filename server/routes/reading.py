from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi.responses import HTMLResponse, StreamingResponse

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


@router.get("/reading/current", response_class=HTMLResponse)
def current_reading_block(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Return the Current Reading block HTML, generating content if needed.

    This endpoint is designed to be called via HTMX after the main page loads,
    so the UI isn't blocked by LLM generation.
    """
    # Determine user profile and any unopened text
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

    # Prefer profile's current_text_id if set
    text_obj = None
    if getattr(prof, "current_text_id", None):
        text_obj = db.get(ReadingText, prof.current_text_id)

    if text_obj is None:
        unopened_text = (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account.id,
                ReadingText.lang == prof.lang,
                ReadingText.opened_at.is_(None),
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )
        if unopened_text is not None:
            prof.current_text_id = unopened_text.id
            db.commit()
            text_obj = unopened_text
        else:
            # Generate one if policy allows
            from ..services.llm_service import should_generate_new_text, generate_reading
            try:
                if should_generate_new_text(db, account.id, prof.lang):
                    result = generate_reading(
                        db,
                        account_id=account.id,
                        lang=prof.lang,
                        length=None,
                        include_words=None,
                        model=None,
                        provider="openrouter",
                        base_url="http://localhost:1234/v1",
                    )
                    text_id = result.get("text_id")
                    if text_id:
                        prof.current_text_id = int(text_id)
                        db.commit()
                        text_obj = db.get(ReadingText, text_id)
                if text_obj is None:
                    # Fallback: latest text
                    text_obj = (
                        db.query(ReadingText)
                        .filter(ReadingText.account_id == account.id, ReadingText.lang == prof.lang)
                        .order_by(ReadingText.created_at.desc())
                        .first()
                    )
                    if text_obj:
                        prof.current_text_id = text_obj.id
                        db.commit()
            except Exception as e:
                return HTMLResponse(
                    content=f'''
                    <div id="current-reading" class="bg-red-50 border border-red-200 rounded-lg p-4">
                      <p class="text-red-700">Failed to generate text: {str(e)}</p>
                      <p class="text-sm text-red-600 mt-2">Please check your LLM configuration.</p>
                    </div>
                    '''
                )

    if text_obj is None:
        # Return bootstrap container that streams generation
        return HTMLResponse(
            content='''
              <div id="cr-content" class="prose max-w-none">
                <div class="animate-pulse space-y-3">
                  <div class="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div class="h-4 bg-gray-200 rounded w-5/6"></div>
                  <div class="h-4 bg-gray-200 rounded w-2/3"></div>
                  <div class="h-4 bg-gray-200 rounded w-4/5"></div>
                </div>
              </div>
              <div id="cr-actions" class="mt-4 hidden"></div>
              <div id="cr-status" class="mt-2 text-sm text-gray-500">Generating…</div>
            '''
        )

    text_id = text_obj.id
    text_html = text_obj.content

    # Return the same block structure used on the page
    inner = f'''
      <div class="prose max-w-none">
        {text_html}
      </div>
      <div class="mt-4 flex gap-2">
        <button
          hx-post="/reading/{text_id}/mark_opened"
          hx-target="#current-reading"
          hx-swap="outerHTML"
          class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
        >
          Mark as Read
        </button>
        <button
          hx-get="/reading/{text_id}/words"
          hx-target="#current-reading"
          hx-swap="innerHTML"
          class="border border-gray-300 hover:bg-gray-50 px-4 py-2 rounded-lg transition-colors"
        >
          View Words
        </button>
      </div>
    '''
    return HTMLResponse(content=inner)


@router.get("/reading/current/stream")
def stream_current_reading(
    db: Session = Depends(get_db),
    account: Account = Depends(_get_current_account),
):
    """Stream generation of current reading text as plain text chunks.

    Client is responsible for inserting text into the page and finalizing UI when stream completes.
    """
    prof = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not prof:
        def _errgen():
            yield "No profile configured."
        return StreamingResponse(_errgen(), media_type="text/plain; charset=utf-8")

    # If there's already a current text, just stream its content quickly
    if getattr(prof, "current_text_id", None):
        rt0 = db.get(ReadingText, prof.current_text_id)
        if rt0 and isinstance(rt0.content, str):
            try:
                print(f"[LLM] Streaming shortcut: existing text id={rt0.id} lang={prof.lang}")
            except Exception:
                pass
            def _existing():
                yield rt0.content
            return StreamingResponse(_existing(), media_type="text/plain; charset=utf-8")

    # Build prompt spec similar to generate_reading
    from ..services.level_service import get_ci_target
    from ..llm import PromptSpec, build_reading_prompt
    from ..llm import pick_words as _pick_words, compose_level_hint as _compose_level_hint
    from ..llm.client import _strip_thinking_blocks
    from ..utils.json_parser import extract_partial_json_string, extract_text_from_llm_response

    # script and other hints
    script = None
    if prof.lang.startswith("zh"):
        ps = getattr(prof, "preferred_script", None)
        script = ps if ps in ("Hans", "Hant") else "Hans"
    ci_target = get_ci_target(db, account.id, prof.lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))

    class _U:
        pass
    u = _U()
    u.id = account.id
    words = _pick_words(db, u, prof.lang, count=12, new_ratio=base_new_ratio)
    level_hint = _compose_level_hint(db, u, prof.lang)
    unit = "chars" if prof.lang.startswith("zh") else "words"
    approx_len = (prof.text_length if isinstance(prof.text_length, int) and prof.text_length else (300 if unit == "chars" else 180))

    spec = PromptSpec(
        lang=prof.lang,
        unit=unit,
        approx_len=int(max(50, min(2000, approx_len))),
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
    )
    messages = build_reading_prompt(spec)

    # OpenRouter streaming
    try:
        from server.llm.openrouter import astream  # type: ignore
    except Exception:
        astream = None  # type: ignore

    def _gen():
        buffer = ""
        last_emitted = 0
        final_text: Optional[str] = None
        # If streaming not available, fall back to sync call
        if astream is None:
            from ..llm import chat_complete
            try:
                try:
                    print(f"[LLM] Streaming unavailable; falling back to sync generation lang={prof.lang}")
                except Exception:
                    pass
                txt = chat_complete(messages, provider="openrouter", model=None, base_url="http://localhost:1234/v1")
            except Exception:
                txt = ""
            final = extract_text_from_llm_response(_strip_thinking_blocks(txt))
            yield final
            # Persist
            rt = ReadingText(account_id=account.id, lang=prof.lang, content=final)
            db.add(rt)
            db.flush()
            prof.current_text_id = rt.id
            db.commit()
            return
        # Use async stream via blocking iteration
        import asyncio
        async def _run():
            nonlocal buffer, last_emitted, final_text
            try:
                print(f"[LLM] Starting to receive the stream lang={prof.lang}")
            except Exception:
                pass
            ctrl = astream(messages, model=None, max_tokens=4096, thinking=False)
            try:
                async for ch in ctrl:
                    kind = ch.get("kind")
                    if kind == "usage":
                        continue
                    if kind == "reasoning":
                        continue
                    if kind == "content":
                        delta = ch.get("text") or ""
                        if not delta:
                            continue
                        buffer += _strip_thinking_blocks(delta)
                        if last_emitted == 0:
                            try:
                                print("[LLM] First stream chunk received")
                            except Exception:
                                pass
                        # Try to extract partial text from JSON, else raw buffer
                        partial = extract_partial_json_string(buffer, "text")
                        if partial is None:
                            partial = buffer
                        if len(partial) > last_emitted:
                            chunk = partial[last_emitted:]
                            last_emitted = len(partial)
                            yield chunk
                # finalize
                final_text = extract_text_from_llm_response(buffer) or buffer
            except Exception:
                final_text = extract_text_from_llm_response(buffer) or buffer

        # Bridge async generator into sync StreamingResponse
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = _run()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(async_gen.__anext__())
                except StopAsyncIteration:
                    break
                if chunk:
                    yield chunk
        finally:
            try:
                loop.run_until_complete(async_gen.aclose())
            except Exception:
                pass
            loop.close()

        # Persist final text
        final = final_text or extract_text_from_llm_response(buffer) or buffer
        try:
            rt = ReadingText(account_id=account.id, lang=prof.lang, content=final)
            db.add(rt)
            db.flush()
            prof.current_text_id = rt.id
            db.commit()
            try:
                print(f"[LLM] Stream complete; saved text id={rt.id} len={len(final)}")
            except Exception:
                pass
        except Exception:
            pass

    return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8")


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

    # If marked as read and it was current, clear profile's pointer
    if payload.read:
        prof = (
            db.query(Profile)
            .filter(Profile.account_id == account.id, Profile.lang == rt.lang)
            .first()
        )
        if prof and getattr(prof, "current_text_id", None) == rt.id:
            prof.current_text_id = None
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

