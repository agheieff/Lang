from __future__ import annotations

"""
Title extraction helpers used by reading routes.

Encapsulates parsing of LLMRequestLog rows to derive:
- title (HTML-escaped by caller)
- title words (array of word objects)
- structured title translation (string)

Now supports global/per-account DB split:
- global_db: ReadingText, ReadingTextTranslation
- account_db: LLMRequestLog
"""

from typing import Optional, Tuple, List, Dict
import json

from sqlalchemy.orm import Session

from ..models import LLMRequestLog, ReadingTextTranslation, ReadingText
from ..utils.json_parser import extract_json_from_text, extract_word_translations


class TitleExtractionService:
    def __init__(self) -> None:
        pass

    def _extract_content_str(self, raw_response: object) -> Optional[str]:
        """Best-effort extraction of the message content from provider response."""
        try:
            if isinstance(raw_response, str):
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
            if isinstance(obj.get("content"), str):
                return obj.get("content")

        return None

    def get_title(
        self,
        global_db: Session,
        text_id: int,
        target_lang: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (title, title_translation) from the DB."""
        # 1. Get raw title from ReadingText
        text = global_db.get(ReadingText, text_id)
        title = text.title if text else None
        
        # 2. Get title translation from ReadingTextTranslation
        title_tr = None
        
        tr_row = (
            global_db.query(ReadingTextTranslation)
            .filter(
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.target_lang == target_lang,
                ReadingTextTranslation.unit == "text",
                ReadingTextTranslation.segment_index == 0  # Title has segment_index=0
            )
            .first()
        )
        
        if tr_row and tr_row.translated_text:
            title_tr = tr_row.translated_text

        return title, title_tr

    def get_title_words(
        self,
        global_db: Session,
        text_id: int,
        target_lang: str,
    ) -> List[Dict]:
        """Return list of title words with translations.
        
        For now, we don't have separate title word translations.
        Returns empty list.
        """
        # Title words could be stored separately or extracted from the main word glosses
        # For now, return empty list
        return []

    def persist_title_translation(
        self,
        global_db: Session,
        text_id: int,
        title: str,
        title_translation: str,
        target_lang: str,
    ) -> bool:
        """Persist title translation to reading_text_translations table."""
        try:
            global_db.add(ReadingTextTranslation(
                text_id=text_id,
                target_lang=target_lang,
                unit="text",
                segment_index=0,
                span_start=0,
                span_end=len(title),
                source_text=title,
                translated_text=title_translation,
                provider=None,
                model=None,
            ))
            
            global_db.flush()
            return True
        except Exception as e:
            print(f"[TITLE] Failed to persist title translation: {e}")
            return False
    
    # Legacy single-session method for backwards compatibility
    def get_title_legacy(
        self,
        db: Session,
        account_id: int,
        text_id: int,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Legacy method using single DB session."""
        # Try to get title from ReadingText
        text = db.get(ReadingText, text_id)
        title = text.title if text else None
        
        # Try to get title translation
        tr_row = (
            db.query(ReadingTextTranslation)
            .filter(
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.unit == "text",
                ReadingTextTranslation.segment_index == 0
            )
            .first()
        )
        
        title_tr = tr_row.translated_text if tr_row else None
        return title, title_tr
