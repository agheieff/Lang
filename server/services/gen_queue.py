from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session, sessionmaker

from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from ..llm import build_reading_prompt
from ..utils.json_parser import (
    extract_text_from_llm_response,
    extract_structured_translation,
    extract_word_translations,
)
from ..llm.client import _strip_thinking_blocks, _pick_openrouter_model, chat_complete_with_raw
import threading
import random
from .llm_logging import log_llm_request
from ..utils.gloss import compute_spans


# In-memory registry of running jobs per (account_id, lang)
_running: set[Tuple[int, str]] = set()
_running_lock = threading.Lock()


def _log_dir_root() -> Path:
    base = os.getenv("ARC_OR_LOG_DIR", str(Path.cwd() / "data" / "llm_stream_logs"))
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _job_dir(account_id: int, lang: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d = _log_dir_root() / str(int(account_id)) / lang / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _enforce_retention(account_id: int, lang: str) -> None:
    try:
        keep = int(os.getenv("ARC_OR_LOG_KEEP", "5"))
    except Exception:
        keep = 5
    if keep <= 0:
        return
    base = _log_dir_root() / str(int(account_id)) / lang
    if not base.exists():
        return
    dirs = [p for p in base.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in dirs[keep:]:
        try:
            # best-effort recursive deletion
            for sub in sorted(old.rglob("*"), reverse=True):
                try:
                    sub.unlink()
                except Exception:
                    pass
            try:
                old.rmdir()
            except Exception:
                pass
        except Exception:
            continue


def _account_session(account_id: int) -> Session:
    from ..account_db import open_account_session
    return open_account_session(int(account_id))


def ensure_text_available(db: Session, account_id: int, lang: str) -> None:
    """If there are no unopened texts for (account, lang) and no running job,
    schedule a background generation job. Non-blocking.
    """
    # Count unopened texts
    unopened = (
        db.query(ReadingText)
        .filter(ReadingText.account_id == account_id, ReadingText.lang == lang, ReadingText.opened_at.is_(None))
        .count()
    )
    key = (int(account_id), str(lang))
    if unopened > 0:
        return
    # Quick non-locking check to avoid thread start; final guard happens in _run_generation_job
    if key in _running:
        return
    # Fire-and-forget via a dedicated thread + event loop
    import threading

    def _worker():
        try:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_run_generation_job(int(account_id), str(lang)))
        finally:
            try:
                loop = asyncio.get_event_loop()
                loop.stop()
                loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, name=f"gen-job-{account_id}-{lang}", daemon=True)
    t.start()


def _complete_and_log(messages: List[Dict], model: str, out_path: Path) -> tuple[str, Dict]:
    """Call OpenRouter sync completion and write a combined log with full prompt history.

    The written file contains both the exact request payload (including all messages)
    and the full JSON response from the provider, for easy reading and auditing:

    {
      "request": { "provider": "openrouter", "model": "...", "max_tokens": 4096, "messages": [...] },
      "response": { ...full OpenRouter response... }
    }

    Returns: (cleaned_text_content, full_response_dict)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_tokens = 4096
    text, resp_dict_or_none = chat_complete_with_raw(
        messages,
        provider="openrouter",
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    resp: Dict = resp_dict_or_none or {}
    try:
        log_obj = {
            "request": {
                "provider": "openrouter",
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            "response": resp,
        }
        out_path.write_text(json.dumps(log_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    # We already have cleaned text from chat_complete_with_raw
    return text or "", resp


def _log_llm_request(db: Session, *, account_id: int, text_id: Optional[int], kind: str, model: str, messages: List[Dict], resp: Optional[Dict], status: str, error: Optional[str] = None) -> None:
    # Backwards-compatible local helper delegating to centralized logger
    log_llm_request(
        db,
        account_id=account_id,
        text_id=text_id,
        kind=kind,
        provider="openrouter",
        model=model,
        base_url=None,
        status=status,
        request={"messages": messages},
        response=resp,
        error=error,
    )


async def _run_generation_job(account_id: int, lang: str) -> None:
    key = (account_id, lang)
    # Atomic check+add under lock
    with _running_lock:
        if key in _running:
            return
        _running.add(key)
    try:
        # Prepare per-account session
        db = _account_session(account_id)
        try:
            # Compose prompt spec (shared helper)
            from .llm_common import build_reading_prompt_spec
            prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
            if prof is None:
                return
            spec, words, level_hint = build_reading_prompt_spec(db, account_id=account_id, lang=lang)
            messages = build_reading_prompt(spec)

            # Working directory for raw job outputs
            job_dir = _job_dir(account_id, lang)

            # Determine OpenRouter model id (non-reasoning preferred)
            model_id = _pick_openrouter_model(None)

            # 1) Reading completion -> persist ReadingText (retry on 429/5xx)
            max_attempts = int(os.getenv("ARC_OR_READING_ATTEMPTS", "3"))
            reading_buf = ""
            reading_resp: Optional[Dict] = None
            last_err: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    reading_buf, reading_resp = _complete_and_log(messages, model=model_id, out_path=job_dir / "reading.json")
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    # Detect HTTP status and Retry-After if available
                    status = None
                    retry_after = None
                    try:
                        status = getattr(e, "response", None).status_code  # type: ignore[attr-defined]
                        retry_after = getattr(e, "response", None).headers.get("Retry-After")  # type: ignore[attr-defined]
                    except Exception:
                        status = None
                    # Log error attempt
                    try:
                        _log_llm_request(db, account_id=account_id, text_id=None, kind="reading", model=model_id, messages=messages, resp=None, status="error", error=str(e))
                    except Exception:
                        pass
                    # Backoff only on retriable statuses; otherwise abort
                    retriable = status in {429, 500, 502, 503, 504}
                    if not retriable or attempt >= max_attempts - 1:
                        break
                    # Compute backoff with jitter; honor Retry-After seconds if numeric
                    delay_base: float
                    try:
                        delay_base = float(retry_after) if retry_after and str(retry_after).isdigit() else float(2 ** (attempt + 1))
                    except Exception:
                        delay_base = float(2 ** (attempt + 1))
                    jitter = random.uniform(0, delay_base * 0.5)
                    await asyncio.sleep(delay_base + jitter)
            if last_err is not None and (not reading_buf or not reading_buf.strip()):
                return
            final_text = extract_text_from_llm_response(reading_buf) or reading_buf

            if not final_text or not final_text.strip():
                return

            rt = ReadingText(account_id=account_id, lang=lang, content=final_text)
            db.add(rt)
            db.flush()

            # Set profile current_text_id to this new text (first-time view will mark opened)
            try:
                prof.current_text_id = rt.id
            except Exception:
                pass
            db.commit()

            # Log reading request/response with full conversation
            _log_llm_request(db, account_id=account_id, text_id=rt.id, kind="reading", model=model_id, messages=messages, resp=reading_resp, status="ok")

            # Release running lock early so next reading can start while translations run
            # Release lock: allow next reading job to start while translations are running
            with _running_lock:
                _running.discard(key)

            # Finish translations in a separate background thread (new DB session per thread)
            def _finish_translations(account_id_: int, lang_: str, text_id_: int, text_html_: str, model_id_: str, job_dir_path: Path, reading_messages: List[Dict]):
                db2 = _account_session(account_id_)
                try:
                    # Rebuild translation contexts
                    prof2 = db2.query(Profile).filter(Profile.account_id == account_id_, Profile.lang == lang_).first()
                    target_lang2 = prof2.target_lang if prof2 and getattr(prof2, "target_lang", None) else "en"
                    reading_user_content2 = reading_messages[1]["content"] if reading_messages and len(reading_messages) > 1 else ""

                    from ..llm.prompts import build_translation_contexts
                    ctx2 = build_translation_contexts(
                        reading_messages,
                        source_lang=lang_,
                        target_lang=target_lang2,
                        text=text_html_,
                    )
                    tr_messages = ctx2["structured"]
                    w_messages = ctx2["words"]

                    # Run both completions concurrently (threads)
                    from concurrent.futures import ThreadPoolExecutor
                    def _call_structured() -> Optional[tuple[str, Dict]]:
                        try:
                            return _complete_and_log(tr_messages, model=model_id_, out_path=job_dir_path / "structured.json")
                        except Exception:
                            return None
                    def _call_words() -> Optional[tuple[str, Dict]]:
                        try:
                            return _complete_and_log(w_messages, model=model_id_, out_path=job_dir_path / "words.json")
                        except Exception:
                            return None
                    with ThreadPoolExecutor(max_workers=2) as ex:
                        fut_tr = ex.submit(_call_structured)
                        fut_wd = ex.submit(_call_words)
                        tr_res = fut_tr.result()
                        wd_res = fut_wd.result()

                    # Persist structured translations
                    try:
                        if tr_res:
                            tr_buf, tr_resp = tr_res
                            tr_parsed = extract_structured_translation(tr_buf)
                            if tr_parsed:
                                target = tr_parsed.get("target_lang") or target_lang2
                                for p in tr_parsed.get("paragraphs", []):
                                    idx = 0
                                    for s in p.get("sentences", []):
                                        if "text" in s and "translation" in s:
                                            db2.add(ReadingTextTranslation(
                                                account_id=account_id_,
                                                text_id=text_id_,
                                                unit="sentence",
                                                target_lang=target,
                                                segment_index=idx,
                                                span_start=None,
                                                span_end=None,
                                                source_text=s["text"],
                                                translated_text=s["translation"],
                                                provider="openrouter",
                                                model=None,
                                            ))
                                            idx += 1
                    except Exception:
                        pass
                    # Log structured translation request/response
                    try:
                        if tr_res:
                            _log_llm_request(db2, account_id=account_id_, text_id=text_id_, kind="structured_translation", model=model_id_, messages=tr_messages, resp=tr_res[1], status="ok")
                    except Exception:
                        pass

                    # Persist word translations with span alignment to the original text
                    try:
                        if wd_res:
                            wd_buf, wd_resp = wd_res
                            wd_parsed = extract_word_translations(wd_buf)
                            if wd_parsed:
                                words_list = [w for w in wd_parsed.get("words", []) if "word" in w and w.get("word")]
                                spans = compute_spans(text_html_, words_list, key="word")
                                for it, sp in zip(words_list, spans):
                                    if sp is None:
                                        continue
                                    db2.add(ReadingWordGloss(
                                        account_id=account_id_,
                                        text_id=text_id_,
                                        lang=lang_,
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
                    except Exception:
                        pass
                    # Log word translation request/response
                    try:
                        if wd_res:
                            _log_llm_request(db2, account_id=account_id_, text_id=text_id_, kind="word_translation", model=model_id_, messages=w_messages, resp=wd_res[1], status="ok")
                    except Exception:
                        pass

                    try:
                        db2.commit()
                    except Exception:
                        db2.rollback()

                    # Meta and retention
                    try:
                        meta = {
                            "account_id": account_id_,
                            "lang": lang_,
                            "text_id": text_id_,
                            "created_at": datetime.utcnow().isoformat(),
                        }
                        (job_dir_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                    _enforce_retention(account_id_, lang_)
                finally:
                    try:
                        db2.close()
                    except Exception:
                        pass

            threading.Thread(
                target=_finish_translations,
                args=(account_id, lang, rt.id, final_text, model_id, job_dir, messages),
                daemon=True,
            ).start()

        finally:
            try:
                db.close()
            except Exception:
                pass
    finally:
        # If we already discarded earlier, this is a no-op
        with _running_lock:
            _running.discard(key)
