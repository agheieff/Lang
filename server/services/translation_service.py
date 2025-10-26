from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Literal

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    ReadingText,
    ReadingTextTranslation,
    TranslationLog,
)
from ..llm import TranslationSpec, build_translation_prompt, chat_complete


def sentence_spans(text: str, lang: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    enders = [".", "!", "?"]
    zh_enders = ["。", "！", "？"]
    if lang.startswith("zh"):
        start = 0
        while i < n:
            ch = text[i]
            if ch in zh_enders:
                spans.append((start, i + 1))
                start = i + 1
            i += 1
        if start < n:
            spans.append((start, n))
        return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]
    start = 0
    while i < n:
        ch = text[i]
        if ch in enders:
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            spans.append((start, j))
            start = j
            i = j
            continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]


def paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    start = 0
    while i < n:
        if text[i] == "\n":
            j = i
            while j < n and text[j] == "\n":
                j += 1
            if j - i >= 2:
                if start < i:
                    spans.append((start, i))
                start = j
                i = j
                continue
        i += 1
    if start < n:
        spans.append((start, n))
    return [(s, e) for (s, e) in spans if e > s and text[s:e].strip()]


def assemble_prev_messages(db: Session, account_id: int, text_id: Optional[int]) -> Optional[List[Dict[str, str]]]:
    if not text_id:
        return None
    from ..models import GenerationLog  # local import to avoid cycles

    msgs: List[Dict[str, str]] = []
    gl = (
        db.query(GenerationLog)
        .filter(GenerationLog.text_id == text_id)
        .order_by(GenerationLog.id.asc())
        .all()
    )
    if gl:
        first = gl[0]
        if isinstance(first.prompt, dict):
            base = first.prompt.get("messages")
            if isinstance(base, list):
                for m in base[-4:]:
                    if isinstance(m, dict) and m.get("content"):
                        msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        rt = db.get(ReadingText, text_id)
        if rt and rt.content:
            msgs.append({"role": "assistant", "content": rt.content})
    tlogs = (
        db.query(TranslationLog)
        .filter(TranslationLog.text_id == text_id)
        .order_by(TranslationLog.id.desc())
        .limit(3)
        .all()
    )
    for tl in reversed(tlogs):
        try:
            pm = tl.prompt.get("messages") if isinstance(tl.prompt, dict) else None
            if isinstance(pm, list) and pm:
                last_user = None
                for m in reversed(pm):
                    if isinstance(m, dict) and m.get("role") == "user":
                        last_user = m.get("content", "")
                        break
                if last_user:
                    msgs.append({"role": "user", "content": last_user})
            if getattr(tl, "response", None):
                msgs.append({"role": "assistant", "content": tl.response})
        except Exception:
            continue
    return msgs or None


def translate_text(
    db: Session,
    *,
    account_id: int,
    lang: str,
    target_lang: str,
    unit: Literal["sentence", "paragraph", "text"],
    text: Optional[str],
    text_id: Optional[int],
    start: Optional[int],
    end: Optional[int],
    continue_with_reading: bool,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
) -> Dict[str, Any]:
    raw: Optional[str] = text
    base_offset = 0
    if raw is None and text_id is not None:
        rt = db.get(ReadingText, text_id)
        if not rt or rt.account_id != account_id:
            raise ValueError("reading text not found")
        raw = rt.content
        if start is not None or end is not None:
            s = max(0, int(start or 0))
            e = min(len(raw), int(end or len(raw)))
            if e <= s:
                raise ValueError("invalid span")
            base_offset = s
            raw = raw[s:e]
    if raw is None:
        raise ValueError("text or text_id required")

    if unit == "text":
        spans = [(0, len(raw))]
    elif unit == "sentence":
        spans = sentence_spans(raw, lang)
    elif unit == "paragraph":
        spans = paragraph_spans(raw)
    else:
        raise ValueError("invalid unit")
    segments = [raw[s:e] for (s, e) in spans]

    prev_msgs: Optional[List[Dict[str, str]]] = None
    if continue_with_reading and text_id:
        prev_msgs = assemble_prev_messages(db, account_id, text_id)

    spec = TranslationSpec(
        lang=lang,
        target_lang=target_lang,
        unit=unit,
        content=(segments if len(segments) > 1 else segments[0]),
        continue_with_reading=bool(continue_with_reading),
        script=None,
    )
    messages = build_translation_prompt(spec, prev_messages=prev_msgs)

    def _call_llm(msgs: List[Dict[str, str]]) -> str:
        return chat_complete(
            msgs,
            provider=provider,
            model=model,
            base_url=(base_url or "http://localhost:1234/v1"),
            temperature=0.3,
        )

    raw_response = _call_llm(messages)

    # Parse response (JSON list or simple lines)
    items: List[Dict[str, Any]] = []
    try:
        tr_lines = _parse_json_translations(raw_response)
    except Exception:
        # fallback: split by newlines
        tr_lines = [ln for ln in (raw_response or "").splitlines() if ln.strip()]

    if len(spans) == len(tr_lines):
        for i, (s, e) in enumerate(spans):
            items.append({
                "start": base_offset + s,
                "end": base_offset + e,
                "source": segments[i],
                "translation": tr_lines[i],
            })
    else:
        # best-effort: single translation
        joined = "\n".join(tr_lines)
        items.append({"start": base_offset, "end": base_offset + len(raw), "source": raw, "translation": joined})

    # Persist translations
    if text_id is not None:
        for i, (s, e) in enumerate(spans):
            r = ReadingTextTranslation(
                account_id=account_id,
                text_id=text_id,
                unit=unit,
                target_lang=target_lang,
                segment_index=(i if unit != "text" else None),
                span_start=base_offset + s,
                span_end=base_offset + e,
                source_text=segments[i],
                translated_text=items[i]["translation"],
                provider=provider,
                model=model,
            )
            db.add(r)

        try:
            db.add(TranslationLog(
                account_id=account_id,
                text_id=text_id,
                unit=unit,
                target_lang=target_lang,
                provider=(provider or "openrouter"),
                model=model,
                prompt={"messages": messages},
                segments={"count": len(segments)},
                response=raw_response,
            ))
        except Exception:
            pass
        db.commit()

    return {
        "unit": unit,
        "target_lang": target_lang,
        "items": items,
        "provider": (provider or "openrouter"),
        "model": model,
    }


def _strip_fences_json(s: str) -> str:
    import re
    t = s.strip()
    m = re.match(r"^```[ \t]*([a-zA-Z0-9_-]+)?\s*\n(?P<body>[\s\S]*?)\n?```\s*$", t)
    return (m.group("body") if m else t).strip()


def _parse_json_translations(text: str) -> List[str]:
    import json
    cleaned = _strip_fences_json(text)
    try:
        data = json.loads(cleaned)
    except Exception:
        try:
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                data = json.loads(cleaned[start : end + 1])
            else:
                return [cleaned.strip()]
        except Exception:
            return [cleaned.strip()]
    if isinstance(data, list):
        return [str(x).strip() for x in data]
    if isinstance(data, dict):
        vals = data.get("translations")
        if isinstance(vals, list):
            return [str(x).strip() for x in vals]
        if isinstance(data.get("translation"), str):
            return [data.get("translation").strip()]
    return [str(data).strip()]

