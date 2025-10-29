from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError

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
_cooldown_next: Dict[Tuple[int, str], float] = {}


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


def _lock_dir_root() -> Path:
    base = Path.cwd() / "data" / "gen_locks"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _lock_path(account_id: int, lang: str) -> Path:
    return _lock_dir_root() / f"{int(account_id)}-{str(lang)}.lock"


def _acquire_file_lock(account_id: int, lang: str) -> Optional[Path]:
    """Cross-process lock using O_EXCL file creation. Stale locks are cleared after TTL.

    Returns lock Path if acquired, else None.
    """
    import errno
    import time as _t
    ttl = float(os.getenv("ARC_GEN_LOCK_TTL_SEC", "300"))
    p = _lock_path(account_id, lang)
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, str(datetime.utcnow()).encode("utf-8"))
        finally:
            os.close(fd)
        return p
    except OSError as e:
        if e.errno != errno.EEXIST:
            return None
        # Lock exists: if stale, remove and retry once
        try:
            st = p.stat()
            age = (datetime.utcnow().timestamp() - st.st_mtime)
            if age > ttl:
                try:
                    p.unlink()
                except Exception:
                    return None
                # retry
                try:
                    fd2 = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    try:
                        os.write(fd2, str(datetime.utcnow()).encode("utf-8"))
                    finally:
                        os.close(fd2)
                    return p
                except Exception:
                    return None
        except Exception:
            return None
    return None


def _release_file_lock(lock_path: Optional[Path]) -> None:
    if not lock_path:
        return
    try:
        lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


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


def ensure_text_available(db: Session, account_id: int, lang: str, *, prefetch: bool = False) -> None:
    """If there are no unopened texts for (account, lang) and no running job,
    schedule a background generation job. Non-blocking.

    When prefetch=True, we allow scheduling the next job even if there is already
    an unopened text available (but never if one is running). Cooldown is also
    bypassed to start immediately after the previous reading is generated.
    """
    # Count unopened texts
    unopened = (
        db.query(ReadingText)
        .filter(ReadingText.account_id == account_id, ReadingText.lang == lang, ReadingText.opened_at.is_(None))
        .count()
    )
    key = (int(account_id), str(lang))
    if (not prefetch) and unopened > 0:
        return
    # Quick non-locking check to avoid thread start; final guard happens in _run_generation_job
    if key in _running:
        return
    # Simple cooldown to avoid hammering providers on repeated failures (skip if prefetch)
    if not prefetch:
        try:
            cd = float(os.getenv("ARC_GEN_COOLDOWN_SEC", "8"))
        except Exception:
            cd = 8.0
        now = time.time()
        next_ok = _cooldown_next.get(key, 0.0)
        if now < next_ok:
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
    if not prefetch:
        _cooldown_next[key] = now + cd


def _complete_and_log(
    messages: List[Dict], *, provider: str, model: Optional[str], base_url: Optional[str], out_path: Path
) -> tuple[str, Dict, str, Optional[str]]:
    """Call a completion provider and write request+response to a single JSON file.

    Returns: (cleaned_text_content, full_response_dict, provider_used, model_used)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_tokens = 16384
    text, resp_dict_or_none = chat_complete_with_raw(
        messages,
        provider=provider,
        model=model,
        base_url=(base_url or "http://localhost:1234/v1"),
        max_tokens=max_tokens,
    )
    resp: Dict = resp_dict_or_none or {}
    try:
        log_obj = {
            "request": {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "max_tokens": max_tokens,
                "messages": messages,
            },
            "response": resp,
        }
        out_path.write_text(json.dumps(log_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return (text or ""), resp, provider, model


def _log_llm_request(
    db: Session,
    *,
    account_id: int,
    text_id: Optional[int],
    kind: str,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    messages: List[Dict],
    resp: Optional[Dict],
    status: str,
    error: Optional[str] = None,
) -> None:
    log_llm_request(
        db,
        account_id=account_id,
        text_id=text_id,
        kind=kind,
        provider=provider,
        model=model,
        base_url=base_url,
        status=status,
        request={"messages": messages},
        response=resp,
        error=error,
    )


async def _run_generation_job(account_id: int, lang: str) -> None:
    key = (account_id, lang)
    lock_path: Optional[Path] = None
    # Cross-process lock first
    lock_path = _acquire_file_lock(account_id, lang)
    if lock_path is None:
        return
    # Atomic in-process guard
    with _running_lock:
        if key in _running:
            _release_file_lock(lock_path)
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

            # Create a skeleton reading row immediately so UI can show a placeholder
            rt = ReadingText(
                account_id=account_id,
                lang=lang,
                content=None,
                request_sent_at=datetime.utcnow(),
            )
            db.add(rt)
            try:
                db.flush()
            except IntegrityError:
                try:
                    db.rollback()
                except Exception:
                    pass
                # Fallback: store empty string if content cannot be NULL (pre-reset DB)
                try:
                    rt = ReadingText(
                        account_id=account_id,
                        lang=lang,
                        content="",
                        request_sent_at=datetime.utcnow(),
                    )
                    db.add(rt)
                    db.flush()
                except Exception:
                    raise

            # Provider order with fallback (default: openrouter,local)
            provider_order = [p.strip() for p in os.getenv("ARC_LLM_PROVIDERS", "openrouter,local").split(",") if p.strip()]
            local_base = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")

            chosen_provider: Optional[str] = None
            chosen_model: Optional[str] = None
            chosen_base: Optional[str] = None

            # 1) Reading completion -> persist ReadingText (retry/backoff for OpenRouter; fallback to local)
            max_attempts = int(os.getenv("ARC_OR_READING_ATTEMPTS", "3"))
            reading_buf = ""
            reading_resp: Optional[Dict] = None
            last_err: Optional[Exception] = None
            for provider in provider_order:
                attempts = max_attempts if provider == "openrouter" else 1
                model_id = _pick_openrouter_model(None) if provider == "openrouter" else None
                base_url = None if provider == "openrouter" else local_base
                for attempt in range(attempts):
                    try:
                        reading_buf, reading_resp, used_provider, used_model = _complete_and_log(
                            messages,
                            provider=provider,
                            model=model_id,
                            base_url=base_url,
                            out_path=job_dir / "text.json",
                        )
                        chosen_provider, chosen_model, chosen_base = used_provider, used_model, base_url
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
                            _log_llm_request(
                                db,
                                account_id=account_id,
                                text_id=rt.id,
                                kind="reading",
                                provider=provider,
                                model=model_id,
                                base_url=base_url,
                                messages=messages,
                                resp=None,
                                status="error",
                                error=str(e),
                            )
                        except Exception:
                            pass
                        # Backoff only on retriable statuses for OpenRouter; otherwise break
                        retriable = provider == "openrouter" and (status in {429, 500, 502, 503, 504})
                        if not retriable or attempt >= attempts - 1:
                            break
                        # Compute backoff with jitter; honor Retry-After seconds if numeric
                        try:
                            delay_base = float(retry_after) if retry_after and str(retry_after).isdigit() else float(2 ** (attempt + 1))
                        except Exception:
                            delay_base = float(2 ** (attempt + 1))
                        jitter = random.uniform(0, delay_base * 0.5)
                        await asyncio.sleep(delay_base + jitter)
                if last_err is None and reading_buf and reading_buf.strip():
                    break
            if last_err is not None and (not reading_buf or not reading_buf.strip()):
                # Drop skeleton on failure
                try:
                    db.delete(rt)
                    db.commit()
                except Exception:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                return
            final_text = extract_text_from_llm_response(reading_buf) or reading_buf

            if not final_text or not final_text.strip():
                # Drop skeleton on empty output
                try:
                    db.delete(rt)
                    db.commit()
                except Exception:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                return

            # Update skeleton with final text and generated timestamp
            try:
                rt.content = final_text
                rt.generated_at = datetime.utcnow()
            except Exception:
                pass
            db.flush()

            # Persist the updated reading but do not update Profile.current_text_id here.
            # The current text pointer is advanced only via POST /reading/next or first open in GET /reading/current.
            db.commit()

            # Log reading request/response with full conversation
            _log_llm_request(
                db,
                account_id=account_id,
                text_id=rt.id,
                kind="reading",
                provider=(chosen_provider or "openrouter"),
                model=chosen_model,
                base_url=chosen_base,
                messages=messages,
                resp=reading_resp,
                status="ok",
            )

            # Release running/file locks early so next reading can start while translations run.
            # We do NOT auto-prefetch here; routes trigger ensure_text_available when a text is opened.
            with _running_lock:
                _running.discard(key)
            try:
                _release_file_lock(lock_path)
            except Exception:
                pass
            lock_path = None

            # Finish translations in a separate background thread (new DB session per thread)
            def _finish_translations(
                account_id_: int,
                lang_: str,
                text_id_: int,
                text_html_: str,
                provider_: str,
                model_id_: Optional[str],
                base_url_: Optional[str],
                job_dir_path: Path,
                reading_messages: List[Dict],
            ):
                db2 = _account_session(account_id_)
                try:
                    # Rebuild translation contexts
                    prof2 = db2.query(Profile).filter(Profile.account_id == account_id_, Profile.lang == lang_).first()
                    target_lang2 = prof2.target_lang if prof2 and getattr(prof2, "target_lang", None) else "en"
                    reading_user_content2 = reading_messages[1]["content"] if reading_messages and len(reading_messages) > 1 else ""

                    from ..llm.prompts import build_translation_contexts, build_word_translation_prompt
                    ctx2 = build_translation_contexts(
                        reading_messages,
                        source_lang=lang_,
                        target_lang=target_lang2,
                        text=text_html_,
                    )
                    tr_messages = ctx2["structured"]
                    w_messages = ctx2["words"]

                    # Run structured + words (optionally per-sentence) concurrently
                    from concurrent.futures import ThreadPoolExecutor
                    def _call_structured() -> Optional[tuple[str, Dict, str, Optional[str]]]:
                        try:
                            return _complete_and_log(
                                tr_messages,
                                provider=provider_,
                                model=model_id_,
                                base_url=base_url_,
                                out_path=job_dir_path / "structured.json",
                            )
                        except Exception:
                            return None

                    def _split_sentences(text: str, lang: str) -> List[Tuple[int, int, str]]:
                        if not text:
                            return []
                        if str(lang).startswith("zh"):
                            pattern = r"[^。！？!?…]+(?:[。！？!?…]+|$)"
                        else:
                            pattern = r"[^\.\!\?]+(?:[\.\!\?]+|$)"
                        out: List[Tuple[int, int, str]] = []
                        for m in re.finditer(pattern, text):
                            s, e = m.span()
                            seg = text[s:e]
                            if seg and seg.strip():
                                out.append((s, e, seg))
                        return out

                    def _attempt_words(messages: List[Dict], out_path: Path) -> Optional[tuple[str, Dict, str, Optional[str]]]:
                        attempts = 1
                        if provider_ == "openrouter":
                            try:
                                attempts = int(os.getenv("ARC_OR_WORDS_ATTEMPTS", "2"))
                            except Exception:
                                attempts = 2
                        last_err: Optional[Exception] = None
                        for attempt in range(attempts):
                            try:
                                return _complete_and_log(
                                    messages,
                                    provider=provider_,
                                    model=model_id_,
                                    base_url=base_url_,
                                    out_path=out_path,
                                )
                            except Exception as e:
                                last_err = e
                                if provider_ != "openrouter" or attempt >= attempts - 1:
                                    break
                                try:
                                    delay_base = float(2 ** (attempt + 1))
                                except Exception:
                                    delay_base = 2.0
                                jitter = random.uniform(0, delay_base * 0.5)
                                try:
                                    import time as _t
                                    _t.sleep(delay_base + jitter)
                                except Exception:
                                    pass
                        return None

                    try:
                        words_parallel = int(os.getenv("ARC_WORDS_PARALLEL", "10"))
                    except Exception:
                        words_parallel = 10
                    # Always split by sentence; use parallelism when >1, otherwise sequential per sentence
                    words_parallel = max(1, words_parallel)

                    tr_res = None
                    wd_seg_results: List[Tuple[int, str, Optional[tuple[str, Dict, str, Optional[str]]], List[Dict]]] = []

                    sent_spans = _split_sentences(text_html_, lang_)
                    # Build per-sentence messages preserving reading context
                    per_msgs: List[Tuple[int, str, List[Dict]]] = []
                    for (s, e, seg) in sent_spans:
                        msgs = build_word_translation_prompt(lang_, target_lang2, seg)
                        words_ctx = [
                            {"role": "system", "content": msgs[0]["content"]},
                            {"role": "user", "content": reading_user_content2},
                            {"role": "assistant", "content": text_html_},
                            {"role": "user", "content": msgs[1]["content"]},
                        ]
                        per_msgs.append((s, seg, words_ctx))

                    if words_parallel == 1:
                        # Run structured in parallel with sequential per-sentence words
                        with ThreadPoolExecutor(max_workers=2) as ex:
                            fut_tr = ex.submit(_call_structured)
                            for i, (s, seg, m) in enumerate(per_msgs):
                                try:
                                    tup = _attempt_words(m, job_dir_path / f"words_{i}.json")
                                except Exception:
                                    tup = None
                                wd_seg_results.append((s, seg, tup, m))
                            tr_res = fut_tr.result()
                    else:
                        max_workers = max(2, min(words_parallel, len(per_msgs)) + 1)
                        with ThreadPoolExecutor(max_workers=max_workers) as ex:
                            fut_tr = ex.submit(_call_structured)
                            futs = []
                            for i, (s, seg, m) in enumerate(per_msgs):
                                futs.append((s, seg, m, ex.submit(_attempt_words, m, job_dir_path / f"words_{i}.json")))
                            tr_res = fut_tr.result()
                            for s, seg, m, f in futs:
                                try:
                                    wd_seg_results.append((s, seg, f.result(), m))
                                except Exception:
                                    wd_seg_results.append((s, seg, None, m))

                    # Persist structured translations
                    try:
                        if tr_res:
                            tr_buf, tr_resp = tr_res[0], tr_res[1]
                            tr_parsed = extract_structured_translation(tr_buf)
                            if tr_parsed:
                                target = tr_parsed.get("target_lang") or target_lang2
                                idx = 0
                                for p in tr_parsed.get("paragraphs", []):
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
                                                provider=provider_,
                                                model=model_id_,
                                            ))
                                            idx += 1
                    except Exception:
                        pass
                    # Log structured translation request/response
                    try:
                        if tr_res:
                            _log_llm_request(
                                db2,
                                account_id=account_id_,
                                text_id=text_id_,
                                kind="structured_translation",
                                provider=provider_,
                                model=model_id_,
                                base_url=base_url_,
                                messages=tr_messages,
                                resp=tr_res[1],
                                status="ok",
                            )
                    except Exception:
                        pass

                    # Persist word translations with span alignment to the original text
                    try:
                        # Avoid duplicates by preloading existing spans and tracking newly added
                        try:
                            existing = set(
                                (rw.span_start, rw.span_end)
                                for rw in db2.query(ReadingWordGloss.span_start, ReadingWordGloss.span_end)
                                .filter(ReadingWordGloss.account_id == account_id_, ReadingWordGloss.text_id == text_id_)
                                .all()
                            )
                        except Exception:
                            existing = set()
                        seen = set()

                        def _persist(seg_start: int, seg_text: str, tup: Optional[tuple[str, Dict, str, Optional[str]]], msgs_used: List[Dict]) -> None:
                            if not tup:
                                return
                            wd_buf, wd_resp = tup[0], tup[1]
                            wd_parsed = extract_word_translations(wd_buf)
                            if not wd_parsed:
                                return
                            words_list = [w for w in wd_parsed.get("words", []) if isinstance(w, dict) and w.get("word")]
                            if not words_list:
                                return
                            spans = compute_spans(seg_text, words_list, key="word")
                            for it, sp in zip(words_list, spans):
                                if sp is None:
                                    continue
                                gs = (seg_start + sp[0], seg_start + sp[1])
                                if gs in existing or gs in seen:
                                    continue
                                _pos_local = it.get("pos") if isinstance(it, dict) else None
                                if not _pos_local and isinstance(it, dict):
                                    _pos_local = it.get("part_of_speech")
                                db2.add(ReadingWordGloss(
                                    account_id=account_id_,
                                    text_id=text_id_,
                                    lang=lang_,
                                    surface=it["word"],
                                    lemma=(None if str(lang_).startswith("zh") else it.get("lemma")),
                                    pos=_pos_local,
                                    pinyin=it.get("pinyin"),
                                    translation=it["translation"],
                                    lemma_translation=it.get("lemma_translation"),
                                    grammar=it.get("grammar", {}),
                                    span_start=gs[0],
                                    span_end=gs[1],
                                ))
                                seen.add(gs)
                            # Log each call
                            try:
                                _log_llm_request(
                                    db2,
                                    account_id=account_id_,
                                    text_id=text_id_,
                                    kind="word_translation",
                                    provider=provider_,
                                    model=model_id_,
                                    base_url=base_url_,
                                    messages=msgs_used,
                                    resp=tup[1],
                                    status="ok",
                                )
                            except Exception:
                                pass

                        try:
                            words_parallel_val = max(1, int(os.getenv("ARC_WORDS_PARALLEL", "1")))
                        except Exception:
                            words_parallel_val = 1
                        if words_parallel_val <= 1:
                            _persist(0, text_html_, wd_res, w_messages)
                        else:
                            # Use the same sentence splitting as above order
                            sent_spans2 = _split_sentences(text_html_, lang_)
                            # Ensure mapping by start offset
                            seg_map = {s: (s, e, seg) for (s, e, seg) in sent_spans2}
                            for s, seg, tup, msgs_used in wd_seg_results:
                                _persist(s, seg, tup, msgs_used)
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
                args=(account_id, lang, rt.id, final_text, (chosen_provider or "openrouter"), chosen_model, chosen_base, job_dir, messages),
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
        _release_file_lock(lock_path)
