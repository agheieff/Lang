from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import os

try:
    # Local OpenRouter client from server/llm/openrouter/
    from server.llm.openrouter import complete as _or_complete  # type: ignore
except Exception:  # pragma: no cover - optional during dev
    _or_complete = None  # type: ignore


def _http_json(url: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Any:
    body = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if data is not None:
        body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers=hdrs, method=method)
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_model(base_url: str, prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    try:
        data = _http_json(base_url.rstrip("/") + "/models")
        arr = data.get("data") or []
        if arr:
            return arr[0].get("id") or "local"
    except Exception:
        pass
    return "local"


def _pick_openrouter_model(requested: Optional[str]) -> str:
    """Prefer non-thinking model variants by default.

    Order:
    1) explicit requested model
    2) env OPENROUTER_MODEL_NONREASONING (preferred override)
    3) env OPENROUTER_MODEL
    4) fallback 'moonshotai/kimi-k2:free'
    """
    if requested:
        return requested
    m = os.getenv("OPENROUTER_MODEL_NONREASONING")
    if m:
        return m
    m2 = os.getenv("OPENROUTER_MODEL")
    return m2 or "moonshotai/kimi-k2:free"


def _strip_thinking_blocks(text: str) -> str:
    import re
    original = text or ""
    # Remove <think>...</think> (and common variants), case-insensitive, multiline
    cleaned = re.sub(r"<\s*(think|thinking|analysis)[^>]*>.*?<\s*/\s*\1\s*>", "", original, flags=re.IGNORECASE | re.DOTALL)
    # Drop leading lines that look like reasoning headers
    cleaned = re.sub(r"^(?:\s*(?:Thoughts?|Thinking|Reasoning)\s*:?\s*\n)+", "", cleaned, flags=re.IGNORECASE)
    # If the model wrapped the whole passage in a fenced block, unwrap and keep the inner content
    # Support optional fence label like ```xml
    fenced_full = re.compile(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$", flags=re.DOTALL)
    m = fenced_full.match(cleaned.strip())
    if m:
        cleaned = m.group(1)
    # Final trim
    cleaned = cleaned.strip()
    # Last-resort: if stripping produced empty but original wasn't empty, preserve original
    if not cleaned and original.strip():
        return original.strip()
    return cleaned


def chat_complete(
    messages: List[Dict[str, str]],
    *,
    provider: Optional[str] = "openrouter",
    model: Optional[str] = None,
    base_url: str = "http://localhost:1234/v1",
    temperature: float = 0.7,
) -> str:
    """Call LLM API and return text response."""
    if provider == "openrouter":
        if _or_complete is None:
            raise RuntimeError("openrouter client not available")
        model_id = _pick_openrouter_model(model)
        try:
            print(f"[LLM] Calling provider=openrouter model={model_id} messages={len(messages)}")
        except Exception:
            pass
        resp = _or_complete(
            messages,
            model=model_id,
            temperature=temperature,
            max_tokens=2048,
        )
        if isinstance(resp, dict):
            choices = resp.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        try:
                            print(f"[LLM] Received response len={len(content)}")
                        except Exception:
                            pass
                        return _strip_thinking_blocks(content)
        raise RuntimeError("invalid openrouter response")

    # Local OpenAI-compatible API
    model_id = resolve_model(base_url, model)
    try:
        print(f"[LLM] Calling provider=local base_url={base_url} model={model_id} messages={len(messages)}")
    except Exception:
        pass
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    try:
        resp = _http_json(base_url.rstrip("/") + "/chat/completions", "POST", data)
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"LLM backend error: {e}")
    choices = resp.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        raise RuntimeError("no choices in response")
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        raise RuntimeError("no message in choice")
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError("no content in message")
    try:
        print(f"[LLM] Received response len={len(content)}")
    except Exception:
        pass
    return _strip_thinking_blocks(content)