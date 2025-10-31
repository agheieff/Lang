from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..models import LLMRequestLog, ReadingWordGloss
from .json_parser import extract_word_translations


def compute_spans(text: str, items: Iterable[Dict[str, Any]], *, key: str = "word") -> List[Optional[Tuple[int, int]]]:
    """Compute left-to-right non-overlapping spans strictly forward-only.

    For each item, finds the first occurrence of item[key] after the previous match end.
    Does NOT fall back to searching from the beginning, to avoid duplicate spans.
    Returns None when not found.
    """
    spans: List[Optional[Tuple[int, int]]] = []
    i = 0
    for it in items:
        s = str((it or {}).get(key) or "")
        if not s:
            spans.append(None)
            continue
        idx = text.find(s, i)
        if idx == -1:
            spans.append(None)
            continue
        spans.append((idx, idx + len(s)))
        i = idx + len(s)
    return spans


def _parse_word_items_from_response_blob(blob: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    try:
        if isinstance(blob, (dict, list)):
            # Try provider JSON structure first
            raw = blob
            if isinstance(raw, dict):
                ch = raw.get("choices")
                if isinstance(ch, list) and ch:
                    msg = ch[0].get("message") if isinstance(ch[0], dict) else None
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            parsed = extract_word_translations(content)
                            if parsed and isinstance(parsed.get("words"), list):
                                return [w for w in parsed["words"] if isinstance(w, dict)]
            # Fallthrough: check if this is already a structured dict
            parsed2 = blob
            if isinstance(parsed2, dict) and isinstance(parsed2.get("words"), list):
                return [w for w in parsed2["words"] if isinstance(w, dict)]
        # String content: may contain JSON or plain text
        s = str(blob)
        parsed3 = extract_word_translations(s)
        if parsed3 and isinstance(parsed3.get("words"), list):
            return [w for w in parsed3["words"] if isinstance(w, dict)]
    except Exception:
        pass
    return items


def _reconstruct_from_db_logs(
    db: Session,
    *,
    account_id: int,
    text_id: int,
    text: str,
    lang: str,
) -> int:
    row = (
        db.query(LLMRequestLog)
        .filter(
            LLMRequestLog.account_id == account_id,
            LLMRequestLog.text_id == text_id,
            LLMRequestLog.kind == "word_translation",
            LLMRequestLog.status == "ok",
        )
        .order_by(LLMRequestLog.created_at.desc())
        .first()
    )
    if not row or not row.response:
        return 0
    blob: Any
    try:
        blob = json.loads(row.response)
    except Exception:
        blob = row.response
    items = _parse_word_items_from_response_blob(blob)
    if not items:
        return 0
    spans = compute_spans(text, items, key="word")
    count = 0
    # Load existing rows to allow in-place updates of missing fields
    try:
        existing_rows = (
            db.query(ReadingWordGloss)
            .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == text_id)
            .all()
        )
        existing_map = {(rw.span_start, rw.span_end): rw for rw in existing_rows}
    except Exception:
        existing_map = {}
    seen: set[Tuple[int, int]] = set()
    for it, sp in zip(items, spans):
        if sp is None:
            continue
        if sp in seen:
            continue
        _pos = None
        try:
            _pos = it.get("pos") or it.get("part_of_speech")
        except Exception:
            _pos = None
        try:
            if sp in existing_map:
                rw = existing_map[sp]
                updated = False
                if (rw.pos is None or rw.pos == "") and _pos:
                    rw.pos = _pos
                    updated = True
                if updated:
                    seen.add(sp)
                    continue
            # Insert if not existing
            if sp not in existing_map:
                db.add(
                    ReadingWordGloss(
                        account_id=account_id,
                        text_id=text_id,
                        lang=lang,
                        surface=it.get("word"),
                        lemma=(None if str(lang).startswith("zh") else it.get("lemma")),
                        pos=_pos,
                        pinyin=it.get("pinyin"),
                        translation=it.get("translation"),
                        lemma_translation=it.get("lemma_translation"),
                        grammar=it.get("grammar", {}),
                        span_start=sp[0],
                        span_end=sp[1],
                    )
                )
                count += 1
                seen.add(sp)
        except Exception:
            continue
    try:
        db.commit()
    except Exception:
        db.rollback()
        return 0
    return count


def _reconstruct_from_file_logs(
    db: Session,
    *,
    account_id: int,
    text_id: int,
    text: str,
    lang: str,
    base_dir: Optional[str] = None,
) -> int:
    try:
        base = Path(base_dir or os.getenv("ARC_OR_LOG_DIR", str(Path.cwd() / "data" / "llm_stream_logs")))
        acc_dir = base / str(int(account_id))
        if not acc_dir.exists():
            return 0
        for lang_dir in sorted([p for p in acc_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            for job in sorted([p for p in lang_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
                meta_path = job / "meta.json"
                try:
                    if not meta_path.exists():
                        continue
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if int(meta.get("text_id")) != int(text_id):
                        continue
                    # Collect items from single-file words.json or per-sentence words_*.json
                    items: List[Dict[str, Any]] = []
                    # Prefer any words_*.json files; fall back to words.json
                    word_files = list(sorted(job.glob("words_*.json")))
                    if not word_files:
                        wp = job / "words.json"
                        if wp.exists():
                            word_files = [wp]
                    for wf in word_files:
                        try:
                            wlog = json.loads(wf.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        its = _parse_word_items_from_response_blob((wlog or {}).get("response"))
                        if its:
                            items.extend([w for w in its if isinstance(w, dict)])
                    if not items:
                        continue
                    spans = compute_spans(text, items, key="word")
                    count = 0
                    try:
                        existing_rows = (
                            db.query(ReadingWordGloss)
                            .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == text_id)
                            .all()
                        )
                        existing_map = {(rw.span_start, rw.span_end): rw for rw in existing_rows}
                    except Exception:
                        existing_map = {}
                    seen: set[Tuple[int, int]] = set()
                    for it, sp in zip(items, spans):
                        if sp is None:
                            continue
                        if sp in seen:
                            continue
                        _pos = None
                        try:
                            _pos = it.get("pos") or it.get("part_of_speech")
                        except Exception:
                            _pos = None
                        try:
                            if sp in existing_map:
                                rw = existing_map[sp]
                                updated = False
                                if (rw.pos is None or rw.pos == "") and _pos:
                                    rw.pos = _pos
                                    updated = True
                                if updated:
                                    seen.add(sp)
                                    continue
                            if sp not in existing_map:
                                db.add(
                                    ReadingWordGloss(
                                        account_id=account_id,
                                        text_id=text_id,
                                        lang=lang,
                                        surface=it.get("word"),
                                        lemma=(None if str(lang).startswith("zh") else it.get("lemma")),
                                        pos=_pos,
                                        pinyin=it.get("pinyin"),
                                        translation=it.get("translation"),
                                        lemma_translation=it.get("lemma_translation"),
                                        grammar=it.get("grammar", {}),
                                        span_start=sp[0],
                                        span_end=sp[1],
                                    )
                                )
                                count += 1
                                seen.add(sp)
                        except Exception:
                            continue
                    try:
                        db.commit()
                    except Exception:
                        db.rollback()
                        continue
                    return count
                except Exception:
                    continue
    except Exception:
        return 0
    return 0


def reconstruct_glosses_from_logs(
    db: Session,
    *,
    account_id: int,
    text_id: int,
    text: str,
    lang: str,
    prefer_db: bool = True,
) -> int:
    """Try to reconstruct ReadingWordGloss rows from stored logs.

    Returns number of rows inserted (best-effort).
    """
    if prefer_db:
        n = _reconstruct_from_db_logs(db, account_id=account_id, text_id=text_id, text=text, lang=lang)
        if n:
            return n
        return _reconstruct_from_file_logs(db, account_id=account_id, text_id=text_id, text=text, lang=lang)
    else:
        n = _reconstruct_from_file_logs(db, account_id=account_id, text_id=text_id, text=text, lang=lang)
        if n:
            return n
        return _reconstruct_from_db_logs(db, account_id=account_id, text_id=text_id, text=text, lang=lang)
