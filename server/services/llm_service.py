from __future__ import annotations

from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    ReadingText,
    GenerationLog,
)
from ..llm import PromptSpec, build_reading_prompt, chat_complete
from .llm_logging import log_llm_request as _db_log_llm_request
from ..utils.gloss import compute_spans
# MSP_ENABLE currently unused in this module
try:
    from logic.mstream.saver import save_word_gloss  # type: ignore
except Exception:
    def save_word_gloss(*args, **kwargs):  # type: ignore
        return None
try:
    from logic.logs import log_llm_request as _log_llm_request_safe  # type: ignore
except Exception:
    def _log_llm_request_safe(*args, **kwargs):  # type: ignore
        return None


def should_generate_new_text(db: Session, account_id: int, lang: str) -> bool:
    """Check if a new text should be generated.

    Returns True if all existing texts for this account/language have been opened.
    """
    # Count total texts and texts that have been opened
    total_texts = db.query(ReadingText).filter(
        ReadingText.account_id == account_id,
        ReadingText.lang == lang
    ).count()

    if total_texts == 0:
        return True  # No texts exist, generate one

    opened_texts = db.query(ReadingText).filter(
        ReadingText.account_id == account_id,
        ReadingText.lang == lang,
        ReadingText.opened_at.isnot(None)
    ).count()

    # Generate new text only when all existing texts have been opened
    return opened_texts >= total_texts


def generate_reading(
    db: Session,
    *,
    account_id: int,
    lang: str,
    length: Optional[int],
    include_words: Optional[List[str]],
    model: Optional[str],
    provider: Optional[str],
    base_url: str,
) -> Dict[str, Any]:
    from .llm_common import build_reading_prompt_spec
    spec, words, level_hint = build_reading_prompt_spec(
        db,
        account_id=account_id,
        lang=lang,
        length=length,
        include_words=include_words,
    )
    messages = build_reading_prompt(spec)

    try:
        text = chat_complete(messages, provider=provider, model=model, base_url=base_url)
    except Exception as e:
        # DB log
        try:
            _db_log_llm_request(
                db,
                account_id=account_id,
                text_id=None,
                kind="reading",
                provider=provider,
                model=model,
                base_url=base_url,
                status="error",
                request={"messages": messages},
                response=None,
                error=str(e),
            )
        except Exception:
            pass
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": None,
            "kind": "reading",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": "error",
            "request": {"messages": messages},
            "response": None,
            "error": str(e),
        })
        raise

    if not text or text.strip() == "":
        raise RuntimeError("empty llm output")

    # Extract text from JSON response if present
    from ..utils.json_parser import extract_text_from_llm_response
    text = extract_text_from_llm_response(text)

    # Persist reading + generation log
    rt = ReadingText(account_id=account_id, lang=lang, content=text)
    db.add(rt)
    db.flush()

    # Update profile's current_text_id to point to this new text
    try:
        prof_for_flag = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if prof_for_flag:
            prof_for_flag.current_text_id = rt.id
    except Exception:
        pass

    # Generate structured translations and word translations IN PARALLEL, using the reading context (4 messages)
    structured_translations = None
    word_translations = None
    try:
        # Get target language from profile
        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        target_lang = prof.target_lang if prof and prof.target_lang else "en"

        # Prepare canonical 4-message contexts
        from ..llm.prompts import build_translation_contexts
        ctx = build_translation_contexts(messages, source_lang=lang, target_lang=target_lang, text=text)
        translation_messages = ctx["structured"]
        word_messages = ctx["words"]

        # Execute both LLM calls in parallel threads (no DB ops in threads)
        results: Dict[str, Optional[str]] = {"structured": None, "words": None}
        def _call(kind: str, msgs: List[Dict[str, str]]) -> None:
            try:
                out = chat_complete(msgs, provider=provider, model=model, base_url=base_url)
                results[kind] = out
            except Exception:  # pragma: no cover
                results[kind] = None

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_tr = ex.submit(_call, "structured", translation_messages)
            fut_wd = ex.submit(_call, "words", word_messages)
            # wait for both to complete
            for _ in as_completed([fut_tr, fut_wd]):
                pass

        # Parse and persist structured translation
        from ..utils.json_parser import extract_structured_translation
        tr_resp = results["structured"]
        if tr_resp:
            structured_translations = extract_structured_translation(tr_resp)
            if structured_translations:
                from ..models import ReadingTextTranslation
                paragraph_index = 0
                for paragraph in structured_translations.get("paragraphs", []):
                    sentence_index = 0
                    for sentence in paragraph.get("sentences", []):
                        if "text" in sentence and "translation" in sentence:
                            rtt = ReadingTextTranslation(
                                account_id=account_id,
                                text_id=rt.id,
                                unit="sentence",
                                target_lang=target_lang,
                                segment_index=sentence_index,
                                span_start=None,
                                span_end=None,
                                source_text=sentence["text"],
                                translated_text=sentence["translation"],
                                provider=provider,
                                model=model,
                            )
                            db.add(rtt)
                            sentence_index += 1
                    paragraph_index += 1

        # Parse and persist word translations
        from ..utils.json_parser import extract_word_translations
        wd_resp = results["words"]
        if wd_resp:
            word_translations = extract_word_translations(wd_resp)
            if word_translations:
                from ..models import ReadingWordGloss
                items = [w for w in word_translations.get("words", []) if "word" in w and w.get("word") and "translation" in w]
                spans = compute_spans(text, items, key="word")
                for it, sp in zip(items, spans):
                    if sp is None:
                        continue
                    db.add(ReadingWordGloss(
                        account_id=account_id,
                        text_id=rt.id,
                        lang=lang,
                        surface=it["word"],
                        lemma=it.get("lemma", it["word"]),
                        pos=it.get("part_of_speech"),
                        pinyin=it.get("pinyin"),
                        translation=it["translation"],
                        lemma_translation=it.get("lemma_translation"),
                        grammar=it.get("grammar", {}),
                        span_start=sp[0],
                        span_end=sp[1],
                    ))

        # Log requests
        # DB logs
        try:
            _db_log_llm_request(
                db,
                account_id=account_id,
                text_id=rt.id,
                kind="structured_translation",
                provider=provider,
                model=model,
                base_url=base_url,
                status=("ok" if tr_resp else "error"),
                request={"messages": translation_messages},
                response=tr_resp,
                error=None if tr_resp else "none",
            )
            _db_log_llm_request(
                db,
                account_id=account_id,
                text_id=rt.id,
                kind="word_translation",
                provider=provider,
                model=model,
                base_url=base_url,
                status=("ok" if wd_resp else "error"),
                request={"messages": word_messages},
                response=wd_resp,
                error=None if wd_resp else "none",
            )
        except Exception:
            pass
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": rt.id,
            "kind": "structured_translation",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": ("ok" if tr_resp else "error"),
            "request": {"messages": translation_messages},
            "response": tr_resp,
            "error": None if tr_resp else "none",
        })
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": rt.id,
            "kind": "word_translation",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": ("ok" if wd_resp else "error"),
            "request": {"messages": word_messages},
            "response": wd_resp,
            "error": None if wd_resp else "none",
        })

    except Exception as e:
        # Log failures but don't fail main reading generation
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": rt.id,
            "kind": "post_reading",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": "error",
            "request": None,
            "response": None,
            "error": str(e),
        })
    gl = GenerationLog(
        account_id=account_id,
        profile_id=(prof.id if prof else None),
        text_id=rt.id,
        model=model,
        base_url=base_url,
        prompt={"messages": messages},
        words={"include": words},
        level_hint=level_hint or None,
        approx_len=spec.approx_len,
        unit=spec.unit,
        created_at=datetime.utcnow(),
    )
    db.add(gl)
    try:
        _db_log_llm_request(
            db,
            account_id=account_id,
            text_id=rt.id,
            kind="reading",
            provider=provider,
            model=model,
            base_url=base_url,
            status="ok",
            request={"messages": messages},
            response=text,
            error=None,
        )
        _log_llm_request_safe({
            "account_id": account_id,
            "text_id": rt.id,
            "kind": "reading",
            "provider": (provider or None),
            "model": (model or None),
            "base_url": (base_url or None),
            "status": "ok",
            "request": {"messages": messages},
            "response": text,
            "error": None,
        })
    except Exception:
        pass

    # Save word gloss placeholders for local caches (best-effort)
    try:
        if words and rt.id:
            for i, w in enumerate(words):
                try:
                    save_word_gloss(
                        db,
                        account_id=account_id,
                        text_id=rt.id,
                        lang=lang,
                        surface=w,
                        span_start=-1 * (i + 1),
                        span_end=-1 * (i + 1),
                        lemma=w,
                        pos=None,
                        pinyin=None,
                        translation=None,
                        lemma_translation=None,
                        grammar=dict(),
                    )
                except Exception:
                    continue
    except Exception:
        pass

    db.commit()
    return {
        "prompt": messages,
        "text": text,
        "level_hint": level_hint,
        "words": words,
        "text_id": rt.id,
        "structured_translations": structured_translations,
        "word_translations": word_translations
    }
