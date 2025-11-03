from __future__ import annotations

from typing import Any, Optional, List, Dict
from pathlib import Path
import json

from sqlalchemy.orm import Session

from ..models import LLMRequestLog
from ..llm.client import chat_complete_with_raw


def _to_response_string(response: Any) -> Optional[str]:
    if response is None:
        return None
    try:
        # Prefer JSON string for dict-like responses
        if isinstance(response, (dict, list)):
            import json
            return json.dumps(response, ensure_ascii=False)
        # Otherwise store as-is (e.g., assistant content string)
        return str(response)
    except Exception:
        try:
            return str(response)
        except Exception:
            return None


def log_llm_request(
    db: Session,
    *,
    account_id: Optional[int],
    text_id: Optional[int],
    kind: str,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    status: str,
    request: Optional[dict],
    response: Optional[Any],
    error: Optional[str] = None,
) -> None:
    """Persist an LLM request/response log.

    Stores request payload (messages and params), and response as JSON string when possible.
    Safe to call in error paths; rolls back on failure.
    """
    try:
        db.add(
            LLMRequestLog(
                account_id=account_id,
                text_id=text_id,
                kind=kind,
                provider=provider,
                model=model,
                base_url=base_url,
                status=status,
                request=request or {},
                response=_to_response_string(response),
                error=error,
            )
        )
        db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


def llm_call_and_log_to_file(
    messages: List[Dict],
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    out_path: Path,
    *,
    max_tokens: int = 16384,
    model_config: Optional[object] = None,
) -> tuple[str, Dict, str, Optional[str]]:
    """Call an LLM and write request/response JSON to out_path. Returns (text, resp, provider, model).

    This helper performs only filesystem logging to remain thread-safe; DB logging should be done by callers.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine effective params
    if model_config is not None:
        try:
            base_url_eff = getattr(model_config, "base_url", None) or base_url or "http://localhost:1234/v1"
            provider_eff = ("openrouter" if isinstance(base_url_eff, str) and ("openrouter" in base_url_eff) else (provider or "local"))
            model_eff = getattr(model_config, "model", None) or model
            max_toks = min(max_tokens, getattr(model_config, "max_tokens", max_tokens) or max_tokens)
        except Exception:
            base_url_eff = base_url or "http://localhost:1234/v1"
            provider_eff = provider or "local"
            model_eff = model
            max_toks = max_tokens
        text, resp_dict = chat_complete_with_raw(
            messages,
            model_config=model_config,
            provider=provider,
            model=model,
            base_url=base_url_eff,
            max_tokens=max_toks,
        )
    else:
        base_url_eff = base_url or "http://localhost:1234/v1"
        provider_eff = provider or "local"
        model_eff = model
        text, resp_dict = chat_complete_with_raw(
            messages,
            provider=provider_eff,
            model=model_eff,
            base_url=base_url_eff,
            max_tokens=max_tokens,
        )
    resp: Dict = resp_dict or {}
    try:
        log_obj = {
            "request": {
                "provider": provider_eff,
                "model": model_eff,
                "base_url": base_url_eff,
                "max_tokens": (max_toks if model_config is not None else max_tokens),
                "messages": messages,
            },
            "response": resp,
        }
        out_path.write_text(json.dumps(log_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return (text or ""), resp, provider_eff, model_eff
