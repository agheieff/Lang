from __future__ import annotations

from typing import Optional
from sqlalchemy.orm import Session

from ..models import ReadingText


class ReconstructionService:
    def ensure_words_from_logs(self, db: Session, account_id: int, text_id: int, *, text: Optional[str] = None, lang: Optional[str] = None) -> None:
        if not text or not lang:
            rt = db.get(ReadingText, int(text_id))
            if not rt or rt.account_id != int(account_id):
                return
            text = rt.content or ""
            lang = rt.lang
        try:
            from ..utils.gloss import reconstruct_glosses_from_logs
            reconstruct_glosses_from_logs(db, account_id=account_id, text_id=text_id, text=text or "", lang=str(lang), prefer_db=True)
        except Exception:
            pass

    def ensure_sentence_translations_from_logs(self, db: Session, account_id: int, text_id: int) -> None:
        rt = db.get(ReadingText, int(text_id))
        if not rt or rt.account_id != int(account_id):
            return
        try:
            from .reading_service import reconstruct_sentences
            reconstruct_sentences(db, int(account_id), rt)
        except Exception:
            pass
