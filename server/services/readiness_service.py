from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss, NextReadyOverride
from ..enums import TextUnit
from ..settings import get_settings
from ..utils.migrations import ensure_reading_text_lifecycle_columns


class ReadinessService:
    def __init__(self):
        self.settings = get_settings()

    def next_unopened(self, db: Session, account_id: int, lang: str) -> Optional[ReadingText]:
        return (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.opened_at.is_(None),
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )

    def _has_words(self, db: Session, account_id: int, text_id: int) -> bool:
        try:
            return (
                db.query(ReadingWordGloss.id)
                .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == text_id)
                .first()
                is not None
            )
        except Exception:
            return False

    def _has_sentences(self, db: Session, account_id: int, text_id: int) -> bool:
        try:
            return (
                db.query(ReadingTextTranslation.id)
                .filter(
                    ReadingTextTranslation.account_id == account_id,
                    ReadingTextTranslation.text_id == text_id,
                    ReadingTextTranslation.unit == TextUnit.SENTENCE,
                )
                .first()
                is not None
            )
        except Exception:
            return False

    def evaluate(self, db: Session, rt: ReadingText, account_id: int) -> Tuple[bool, str]:
        ensure_reading_text_lifecycle_columns(db)
        if not getattr(rt, "content", None):
            return (False, "no_content")
        
        has_w = self._has_words(db, account_id, rt.id)
        has_s = self._has_sentences(db, account_id, rt.id)
        
        # Full readiness: both words and sentences present
        if has_w and has_s:
            return (True, "both")
        
        # Check age-based grace periods
        gen_at = getattr(rt, "generated_at", None)
        if not gen_at:
            return (False, "waiting")
        
        try:
            age = (datetime.utcnow() - gen_at).total_seconds()
            
            # Grace period: partial translations after 60s
            if age >= float(self.settings.NEXT_READY_GRACE_SEC) and (has_w or has_s):
                return (True, "grace")
            
            # Content-only fallback: no translations but text exists after 120s
            if age >= float(self.settings.CONTENT_ONLY_GRACE_SEC):
                return (True, "content_only")
        except Exception:
            pass
        
        return (False, "waiting")

    def get_failed_components(self, db: Session, account_id: int, text_id: int) -> dict:
        """Return dict with missing components: {'words': bool, 'sentences': bool}"""
        if not db:
            return {"words": False, "sentences": False}
        
        has_words = self._has_words(db, account_id, text_id)
        has_sentences = self._has_sentences(db, account_id, text_id)
        
        return {
            "words": not has_words,
            "sentences": not has_sentences
        }

    def needs_retry(self, db: Session, account_id: int, text_id: int) -> bool:
        """Check if text has any missing components that need retry"""
        failed = self.get_failed_components(db, account_id, text_id)
        return failed["words"] or failed["sentences"]

    def force_once(self, db: Session, account_id: int, lang: str, ttl_s: int = 60) -> None:
        exp = datetime.utcnow() + timedelta(seconds=max(1, int(ttl_s)))
        row = (
            db.query(NextReadyOverride)
            .filter(NextReadyOverride.account_id == account_id, NextReadyOverride.lang == lang)
            .first()
        )
        if row:
            row.expires_at = exp
        else:
            db.add(NextReadyOverride(account_id=account_id, lang=lang, expires_at=exp))
        try:
            db.commit()
        except Exception:
            db.rollback()

    def consume_if_valid(self, db: Session, account_id: int, lang: str) -> bool:
        now = datetime.utcnow()
        row = (
            db.query(NextReadyOverride)
            .filter(NextReadyOverride.account_id == account_id, NextReadyOverride.lang == lang)
            .first()
        )
        if not row:
            return False
        valid = bool(row.expires_at is None or row.expires_at > now)
        try:
            db.delete(row)
            db.commit()
        except Exception:
            db.rollback()
        return valid
