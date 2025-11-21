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

from ..models import LLMRequestLog, ReadingTextTranslation, Profile
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
        """Return (title, title_translation) from the DB."""
        # 1. Get raw title from generation log
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
        
        title = None
        
        # Extract title from log
        if row and getattr(row, "response", None):
            content_str = None
            try:
                raw = row.response
                try:
                    obj = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    obj = raw
                content_str = self._extract_content_str(obj if obj is not None else raw)
            except Exception:
                content_str = None

            if content_str:
                try:
                    t = extract_json_from_text(content_str, "title")
                    if t is not None:
                        title = str(t)
                except Exception:
                    title = None
        
        # 2. Get title translation from ReadingTextTranslation (PRIMARY source)
        title_tr = None
        
        # Find target language for this user/text
        # We need the text language first, but we can try to infer or just look for any translation
        # Best to query specifically for unit='text'
        
        tr_row = (
            db.query(ReadingTextTranslation)
            .filter(
                ReadingTextTranslation.account_id == account_id,
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.unit == "text",
                # We assume segment_index=0 for title
                ReadingTextTranslation.segment_index == 0
            )
            .first()
        )
        
        if tr_row and tr_row.translated_text:
            title_tr = tr_row.translated_text
            
        # 3. Fallback: try to extract translation from generation log if DB lookup failed
        if not title_tr and title and content_str:
            try:
                tt = extract_json_from_text(content_str, "title_translation")
                if tt is not None:
                    title_tr = str(tt)
            except Exception:
                pass

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

    def persist_title_translation(self, db: Session, account_id: int, text_id: int, title: str, title_translation: str, target_lang: str) -> bool:
        """Persist title translation to reading_text_translations table."""
        try:
            db.add(ReadingTextTranslation(
                account_id=account_id,
                text_id=text_id,
                unit="text",
                target_lang=target_lang,
                segment_index=0,
                span_start=0,
                span_end=len(title),
                source_text=title,
                translated_text=title_translation,
                provider=None,  # We don't have provider info here
                model=None,
            ))
            
            # Flush to ensure data is written
            db.flush()
            return True
        except Exception as e:
            print(f"[TITLE] Failed to persist title translation: {e}")
            return False
