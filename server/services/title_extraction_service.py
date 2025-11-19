from __future__ import annotations

"""
Title extraction helpers used by reading routes.

Encapsulates parsing of LLMRequestLog rows to derive:
- title (HTML-escaped by caller)
- title words (array of word objects)
- structured title translation (string)
"""

from typing import Optional, Tuple, List, Dict
import json

from sqlalchemy.orm import Session

from ..models import LLMRequestLog
from ..utils.json_parser import extract_json_from_text, extract_word_translations


class TitleExtractionService:
    def __init__(self) -> None:
        pass

    def _extract_content_str(self, raw_response: object) -> Optional[str]:
        """Best-effort extraction of the message content from provider response."""
        try:
            if isinstance(raw_response, str):
                # Might already be a JSON string or plain content
                try:
                    obj = json.loads(raw_response)
                except Exception:
                    return raw_response
            else:
                obj = raw_response
        except Exception:
            return None

        if isinstance(obj, dict):
            try:
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg.get("content")
            except Exception:
                pass
            # Fallback to top-level content
            if isinstance(obj.get("content"), str):
                return obj.get("content")

        return None

    def get_title(self, db: Session, account_id: int, text_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Return (title, title_translation) from the latest successful reading log, if present."""
        row = (
            db.query(LLMRequestLog)
            .filter(
                LLMRequestLog.account_id == account_id,
                LLMRequestLog.text_id == text_id,
                LLMRequestLog.kind == "reading",
                LLMRequestLog.status == "ok",
            )
            .order_by(LLMRequestLog.created_at.desc())
            .first()
        )
        if not row or not getattr(row, "response", None):
            return None, None

        content_str: Optional[str] = None
        try:
            # response may be dict-like JSON or string; store both raw and parsed
            raw = row.response
            try:
                obj = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                obj = raw
            content_str = self._extract_content_str(obj if obj is not None else raw)
        except Exception:
            content_str = None

        if not content_str:
            return None, None

        title = None
        title_tr = None
        try:
            t = extract_json_from_text(content_str, "title")
            if t is not None:
                title = str(t)
        except Exception:
            title = None
        try:
            title_tr = extract_json_from_text(content_str, "title_translation")
            if title_tr is not None:
                title_tr = str(title_tr)
        except Exception:
            title_tr = None

        return title, title_tr

    def get_title_words(self, db: Session, account_id: int, text_id: int) -> List[Dict]:
        """Return list of title words with translations from title_word_translation log, if any."""
        row = (
            db.query(LLMRequestLog)
            .filter(
                LLMRequestLog.account_id == account_id,
                LLMRequestLog.text_id == text_id,
                LLMRequestLog.kind == "title_word_translation",
                LLMRequestLog.status == "ok",
            )
            .order_by(LLMRequestLog.created_at.desc())
            .first()
        )
        if not row or not getattr(row, "response", None):
            return []

        try:
            obj = json.loads(row.response)
        except Exception:
            obj = row.response

        content = None
        try:
            if isinstance(obj, dict):
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        content = msg.get("content")
            if content is None and isinstance(obj, str):
                content = obj
        except Exception:
            content = None

        if not content:
            return []

        parsed = extract_word_translations(content)
        ws = []
        if parsed and isinstance(parsed.get("words"), list):
            ws = [w for w in parsed.get("words", []) if isinstance(w, dict)]
            for w in ws:
                # Ensure we do not leak span positions for title words
                w.pop("span_start", None)
                w.pop("span_end", None)

        return ws
