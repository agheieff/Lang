from __future__ import annotations

"""
Consolidated Natural Language Processing utilities.
Merged from text_segmentation.py, json_parser.py, and gloss.py
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Iterable


def strip_thinking_blocks(text: str) -> str:
    """Remove thinking blocks from LLM responses."""
    original = text or ""
    cleaned = re.sub(
        r"<\s*(think|thinking|analysis)[^>]*>.*?<\s*/\s*\1\s*>",
        "",
        original,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(
        r"^(?:\s*(?:Thoughts?|Thinking|Reasoning)\s*:?\s*\n)+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    fenced_full = re.compile(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$", flags=re.DOTALL)
    m = fenced_full.match(cleaned.strip())
    if m:
        cleaned = m.group(1)

    cleaned = cleaned.strip()
    if not cleaned and original.strip():
        return original.strip()
    return cleaned


# Text Segmentation Functions
def split_sentences(text: str, lang: str) -> List[Tuple[int, int, str]]:
    """Split text into sentences, returning (start, end, segment) tuples."""
    if not text:
        return []
    if str(lang).startswith("zh"):
        pattern = r"[^。！？!?…]+(?:[。！？!?…]+|$)"
    else:
        pattern = r"[^\.!?]+(?:[\.!?]+|$)"
    out: List[Tuple[int, int, str]] = []
    for m in re.finditer(pattern, text):
        s, e = m.span()
        seg = text[s:e]
        if seg and seg.strip():
            out.append((s, e, seg))
    return out


def split_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """Split text into paragraphs separated by blank lines (\n\n+)."""
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        # skip leading newlines
        while i < n and text[i] == "\n":
            i += 1
        if i >= n:
            break
        s = i
        # paragraph ends at next blank line boundary
        while i < n - 1 and not (text[i] == "\n" and text[i + 1] == "\n"):
            i += 1
        # include until end of line
        while i < n and text[i] != "\n":
            i += 1
        e = min(n, i)
        seg = text[s:e]
        if seg and seg.strip():
            out.append((s, e, seg))
    return out


# JSON Parsing Functions
def extract_json_from_text(text: str, expected_key: str = "text") -> Optional[Any]:
    """Extract JSON object from text response."""
    if not text or text.strip() == "":
        return None

    s = text.strip()
    # 1) Try full-document JSON first
    try:
        data = json.loads(s)
        if isinstance(data, dict) and expected_key in data:
            return data[expected_key]
    except Exception:
        pass

    # 2) Try code-fenced JSON (```json ... ```)
    def _extract_balanced(buf: str, start: int) -> Optional[str]:
        depth = 0
        i = start
        n = len(buf)
        in_str = False
        esc = False
        while i < n:
            ch = buf[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return buf[start : i + 1]
            i += 1
        return None

    m = re.search(r"```json\s*", s)
    if m:
        start_brace = s.find("{", m.end())
        if start_brace != -1:
            blob = _extract_balanced(s, start_brace)
            if blob:
                try:
                    data2 = json.loads(blob)
                    if isinstance(data2, dict) and expected_key in data2:
                        return data2[expected_key]
                except Exception:
                    pass

    # 3) Try to find a simple JSON object with the key in the response
    json_match = re.search(
        r'\{[^{}]*"' + re.escape(expected_key) + r'"[^{}]*\}', text, re.DOTALL
    )
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if isinstance(json_data, dict) and expected_key in json_data:
                return json_data[expected_key]
        except json.JSONDecodeError:
            pass

    return None


def extract_word_translations(text: str) -> List[Dict[str, Any]]:
    """Extract word translation array from LLM response."""
    if not text or text.strip() == "":
        return []

    try:
        # Try parsing as JSON array
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Try extracting JSON array from text
    json_match = re.search(r"\[[^\]]*\]", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def compute_spans(
    text: str, items: Iterable[Dict[str, Any]], *, key: str = "word"
) -> List[Optional[Tuple[int, int]]]:
    """Compute left-to-right non-overlapping spans strictly forward-only."""
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


def extract_text_from_llm_response(text: str) -> str:
    """Extract plain text from LLM response when JSON parsing fails."""
    if not text or text.strip() == "":
        return ""

    # Remove code fences and extract text
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            text = (
                "\n".join(lines[1:-1])
                if lines[-1].startswith("```")
                else "\n".join(lines[1:])
            )

    # Try to extract JSON first, then fall back to plain text
    try:
        possible_json = extract_json_from_text(text, "text")
        if possible_json and isinstance(possible_json, str):
            return possible_json
    except Exception:
        pass

    return text


def extract_structured_translation(text: str) -> Optional[Dict[str, Any]]:
    """Extract structured translation from LLM response."""
    try:
        # Try full JSON first
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try to extract JSON object from text
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def parse_word_items_from_response_blob(blob: Any) -> List[Dict[str, Any]]:
    """Parse word items from LLM response blob."""
    items: List[Dict[str, Any]] = []
    try:
        if isinstance(blob, (dict, list)):
            raw = blob
            if isinstance(raw, dict):
                ch = raw.get("choices")
                if isinstance(ch, list) and ch:
                    msg = ch[0].get("message") if isinstance(ch[0], dict) else None
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            parsed = extract_word_translations(content)
                            if isinstance(parsed, list):
                                items.extend(parsed)
            elif isinstance(raw, list):
                items = raw
    except Exception:
        pass
    return items


def parse_csv_word_translations(text: str, lang: str = "en") -> List[Dict[str, str]]:
    """Parse CSV word translations from LLM response.

    Language-specific formats:
    - Chinese (zh-CN, zh-TW): word|translation|pos|pinyin (no lemma, same as surface)
    - Other languages: word|translation|pos|lemma|pronunciation

    For non-continuous words (e.g., German prefixes), use ... notation:
    ruf...an|call|VERB|anrufen|
    """
    if not text or text.strip() == "":
        return []

    items: List[Dict[str, str]] = []

    # Chinese languages don't need lemma (same as surface)
    is_chinese = lang.startswith("zh-")

    # Remove code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 1:
            cleaned = (
                "\n".join(lines[1:-1])
                if lines[-1].startswith("```")
                else "\n".join(lines[1:])
            )

    # Split by lines and parse each
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        # Skip header row
        first_col = parts[0].strip().lower()
        if first_col in ("word", "words", "surface"):
            continue

        if is_chinese:
            # Chinese format: word|translation|pos|pinyin
            surface = parts[0].strip()
            item = {
                "surface": surface,
                "lemma": surface,  # For Chinese, lemma = surface
                "translation": parts[1].strip() if len(parts) > 1 else "",
                "pos": parts[2].strip() if len(parts) > 2 else "",
                "pinyin": parts[3].strip() if len(parts) > 3 else "",
            }
        else:
            # Other languages format: word|translation|pos|lemma|pronunciation
            item = {
                "surface": parts[0].strip(),
                "translation": parts[1].strip() if len(parts) > 1 else "",
                "pos": parts[2].strip() if len(parts) > 2 else "",
                "lemma": parts[3].strip() if len(parts) > 3 else "",
                "pinyin": parts[4].strip() if len(parts) > 4 else "",
            }
        items.append(item)

    return items


def compute_word_spans(
    text: str, word_data: List[Dict[str, str]]
) -> List[List[Tuple[int, int]]]:
    """Compute spans for words, handling non-continuous words marked with ...

    For non-continuous words like "ruf...an", finds the position of "ruf"
    and then "an" after it, returning [(start1, end1), (start2, end2)].

    For continuous words, returns [(start, end)].
    """
    all_spans: List[List[Tuple[int, int]]] = []
    current_pos = 0

    for word in word_data:
        surface = word.get("surface", "")
        if not surface:
            all_spans.append([])
            continue

        if "..." in surface:
            # Non-continuous word
            segments = surface.split("...")
            word_spans: List[Tuple[int, int]] = []
            segment_pos = current_pos

            for segment in segments:
                if not segment:
                    continue
                idx = text.find(segment, segment_pos)
                if idx == -1:
                    # Try finding from beginning if not found
                    idx = text.find(segment)
                    if idx == -1:
                        word_spans.append((0, 0))
                        continue

                word_spans.append((idx, idx + len(segment)))
                segment_pos = idx + len(segment)

            # Update current position after last segment
            if word_spans:
                current_pos = word_spans[-1][1]
            all_spans.append(word_spans)
        else:
            # Continuous word
            idx = text.find(surface, current_pos)
            if idx == -1:
                # Try finding from beginning
                idx = text.find(surface)

            if idx == -1:
                all_spans.append([])
                continue

            all_spans.append([(idx, idx + len(surface))])
            current_pos = idx + len(surface)

    return all_spans


def parse_csv_translation(text: str) -> List[Dict[str, str]]:
    """Parse CSV sentence translations from LLM response.

    Expected format:
    source|translation

    One line per sentence/paragraph.
    """
    if not text or text.strip() == "":
        return []

    items: List[Dict[str, str]] = []

    # Remove code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 1:
            cleaned = (
                "\n".join(lines[1:-1])
                if lines[-1].startswith("```")
                else "\n".join(lines[1:])
            )

    # Split by lines and parse each
    for line in cleaned.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|")
        if len(parts) < 2:
            continue

        # Skip header row
        first_col = parts[0].strip().lower()
        if first_col in ("source", "original", "text"):
            continue

        item = {
            "source": parts[0].strip(),
            "translation": parts[1].strip() if len(parts) > 1 else "",
        }
        items.append(item)

    return items


def parse_csv_title_translation(text: str) -> Optional[Dict[str, str]]:
    """Parse CSV title translation from LLM response.

    Expected format:
    title|translation
    """
    if not text or text.strip() == "":
        return None

    # Remove code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if len(lines) > 1:
            cleaned = (
                "\n".join(lines[1:-1])
                if lines[-1].startswith("```")
                else "\n".join(lines[1:])
            )

    parts = cleaned.split("|")
    if len(parts) < 2:
        return None

    return {
        "title": parts[0].strip(),
        "title_translation": parts[1].strip() if len(parts) > 1 else "",
    }
