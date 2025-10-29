"""
JSON parsing utilities for LLM responses.
"""

import json
import re
from typing import Any, Dict, Optional, Tuple


def extract_json_from_text(text: str, expected_key: str = "text") -> Optional[Any]:
    """
    Extract JSON object from text response.

    Args:
        text: The text response from LLM
        expected_key: The key to look for in the JSON object

    Returns:
        The extracted value if found, otherwise None
    """
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
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return buf[start:i+1]
            i += 1
        return None

    m = re.search(r"```json\s*", s)
    if m:
        start_brace = s.find('{', m.end())
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
    json_match = re.search(r'\{[^{}]*"' + re.escape(expected_key) + r'"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if isinstance(json_data, dict) and expected_key in json_data:
                return json_data[expected_key]
        except json.JSONDecodeError:
            # If JSON parsing fails, try to find the first valid JSON object
            pass

    # 4) Alternative: try to find any JSON object containing the key
    try:
        # Look for JSON objects with the expected key
        json_objects = re.findall(r'\{[^{}]*"' + re.escape(expected_key) + r'"[^{}]*\}', text, re.DOTALL)
        for json_str in json_objects:
            try:
                json_data = json.loads(json_str)
                if isinstance(json_data, dict) and expected_key in json_data:
                    return json_data[expected_key]
            except json.JSONDecodeError:
                continue
    except Exception:
        pass

    return None




def extract_text_from_llm_response(response: str) -> str:
    """
    Extract text from LLM response, handling JSON format if present.

    Args:
        response: The raw response from LLM

    Returns:
        The extracted text
    """
    if not response or response.strip() == "":
        return ""

    # Try to extract text from JSON
    extracted_text = extract_json_from_text(response, "text")
    if extracted_text is not None:
        return str(extracted_text)

    # Fallback to original response
    return response


def extract_partial_json_string(buffer: str, key: str = "text") -> Optional[str]:
    """Best-effort extraction of a JSON string value for a given key from a partial/incomplete buffer.

    Handles unfinished JSON while streaming by locating the opening quote after "key": and
    scanning until the next unescaped matching quote. Returns the partial contents collected so far.
    """
    if not buffer:
        return None
    try:
        start_key = f'"{key}"'
        i = buffer.find(start_key)
        if i < 0:
            return None
        # find colon after key
        i = buffer.find(":", i)
        if i < 0:
            return None
        # skip spaces
        n = len(buffer)
        i += 1
        while i < n and buffer[i].isspace():
            i += 1
        if i >= n:
            return None
        q = buffer[i]
        if q not in ('"', "'"):
            return None
        i += 1
        out: list[str] = []
        esc = False
        while i < n:
            ch = buffer[i]
            i += 1
            if esc:
                # rudimentary unescape for common sequences
                if ch in ('"', "'", "\\", "/"):
                    out.append(ch)
                elif ch == "n":
                    out.append("\n")
                elif ch == "t":
                    out.append("\t")
                elif ch == "r":
                    out.append("\r")
                else:
                    out.append(ch)
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == q:
                # closing quote reached
                break
            out.append(ch)
        return "".join(out)
    except Exception:
        return None


def extract_structured_translation(response: str) -> Optional[Dict[str, Any]]:
    """Extract structured sentence translations and normalize to
    {"paragraphs": [{"sentences": [{"text": str, "translation": str}, ...]}, ...]}.

    Accepts the provider schema found in recent logs:
    {"text": [ {"paragraph": [ {"sentence": {"text": str, "translation": str}}, ... ]}, ... ]}

    Falls back to already-normalized {"paragraphs": [{"sentences": [...]}]} when present.
    """
    if not response or response.strip() == "":
        return None

    def _json_from_any(s: str) -> Optional[Dict[str, Any]]:
        # 1) Try direct JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        # 2) Try code-fenced
        m = re.search(r"```json\s*([\s\S]*?)```", s)
        if m:
            body = m.group(1).strip()
            try:
                obj = json.loads(body)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        # 3) Try to extract first object braces
        first = s.find('{')
        last = s.rfind('}')
        if first != -1 and last != -1 and last > first:
            frag = s[first:last+1]
            try:
                obj = json.loads(frag)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return None

    data = _json_from_any(response)
    if not isinstance(data, dict):
        return None

    # Case A: already normalized shape
    if isinstance(data.get("paragraphs"), list):
        return data

    # Case B: provider schema: {"text":[{"paragraph":[{"sentence":{...}}, ...]}, ...]}
    txt = data.get("text")
    if isinstance(txt, list):
        paragraphs_out: list[dict] = []
        for para in txt:
            if not isinstance(para, dict):
                continue
            sent_items = para.get("paragraph")
            if not isinstance(sent_items, list):
                continue
            sentences: list[dict] = []
            for item in sent_items:
                if not isinstance(item, dict):
                    continue
                sent = item.get("sentence")
                if not isinstance(sent, dict):
                    continue
                t = sent.get("text")
                tr = sent.get("translation")
                if t is None or tr is None:
                    continue
                sentences.append({"text": str(t), "translation": str(tr)})
            if sentences:
                paragraphs_out.append({"sentences": sentences})
        if paragraphs_out:
            out: Dict[str, Any] = {"paragraphs": paragraphs_out}
            # carry target_lang if present
            if isinstance(data.get("target_lang"), str):
                out["target_lang"] = data["target_lang"]
            return out

    return None


def extract_word_translations(response: str) -> Optional[Dict[str, Any]]:
    """
    Extract word translations from LLM response.

    Args:
        response: The raw response from LLM

    Returns:
        The word translation data if found, otherwise None
    """
    if not response or response.strip() == "":
        return None

    # Helper: find balanced JSON object starting at given '{'
    def _extract_balanced(text: str, start: int) -> Optional[str]:
        depth = 0
        i = start
        n = len(text)
        in_str = False
        esc = False
        while i < n:
            ch = text[i]
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
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
            i += 1
        return None

    # Prefer code-fenced JSON if present (support dict or array)
    fence = re.search(r"```json\s*", response)
    if fence:
        fence_start = fence.end()
        fence_end = response.find("```", fence_start)
        if fence_end != -1:
            body = response[fence_start:fence_end].strip()
            try:
                data_any = json.loads(body)
                if isinstance(data_any, dict) and isinstance(data_any.get("words"), list):
                    return data_any
                if isinstance(data_any, list):
                    return {"words": data_any}
            except Exception:
                pass
        # fallback to object-only balanced extraction inside fenced block
        start_brace = response.find('{', fence_start)
        if start_brace != -1:
            blob = _extract_balanced(response, start_brace)
            if blob:
                try:
                    data = json.loads(blob)
                    if isinstance(data, dict) and isinstance(data.get("words"), list):
                        return data
                except Exception:
                    pass

    # Otherwise, try from the first '{' or closest before "\"words\""
    candidates: list[int] = []
    idx_words = response.find('"words"')
    if idx_words != -1:
        brace_before = response.rfind('{', 0, idx_words)
        if brace_before != -1:
            candidates.append(brace_before)
    first_brace = response.find('{')
    if first_brace != -1:
        candidates.append(first_brace)

    # Try full-document JSON array
    try:
        any_data = json.loads(response)
        if isinstance(any_data, list):
            return {"words": any_data}
    except Exception:
        pass

    seen: set[int] = set()
    for pos in candidates:
        if pos in seen:
            continue
        seen.add(pos)
        blob = _extract_balanced(response, pos)
        if not blob:
            continue
        try:
            data = json.loads(blob)
            if isinstance(data, dict) and isinstance(data.get("words"), list):
                return data
        except Exception:
            continue

    return None