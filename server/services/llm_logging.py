from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from ..models import LLMRequestLog


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
