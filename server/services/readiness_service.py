from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Tuple, Set

from sqlalchemy.orm import Session

from ..models import (
    Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss, 
    NextReadyOverride, ProfileTextRead
)
from ..enums import TextUnit
from ..settings import get_settings
from ..utils.migrations import ensure_reading_text_lifecycle_columns


class ReadinessService:
    """
    Service to evaluate text readiness.
    Now supports global/per-account DB split:
    - global_db: ReadingText, ReadingTextTranslation, ReadingWordGloss
    - account_db: ProfileTextRead, NextReadyOverride
    """
    
    def __init__(self):
        self.settings = get_settings()

    def next_unopened(
        self,
        global_db: Session,
        account_db: Session,
        lang: str,
        target_lang: str,
        profile_id: int,
    ) -> Optional[ReadingText]:
        """Get newest unread text for a profile."""
        # Get read text IDs
        read_ids = self._get_read_text_ids(account_db, profile_id)
        
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
            )
        )
        
        if read_ids:
            query = query.filter(~ReadingText.id.in_(read_ids))
        
        return query.order_by(ReadingText.created_at.desc()).first()

    def first_ready_backup(
        self,
        global_db: Session,
        account_db: Session,
        lang: str,
        target_lang: str,
        profile_id: int,
        exclude_text_id: Optional[int] = None,
    ) -> Tuple[Optional[ReadingText], str]:
        """
        Find any ready backup text from the global pool.
        Returns (text, reason) where reason is 'both', 'grace', etc.
        """
        read_ids = self._get_read_text_ids(account_db, profile_id)
        
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
                ReadingText.content.isnot(None),
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        if read_ids:
            query = query.filter(~ReadingText.id.in_(read_ids))
        
        if exclude_text_id:
            query = query.filter(ReadingText.id != exclude_text_id)
        
        text = query.first()
        if text:
            return text, "both"
        
        return None, "waiting"
    
    def _get_read_text_ids(self, account_db: Session, profile_id: int) -> Set[int]:
        """Get IDs of texts the profile has read."""
        return set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == profile_id)
            .all()
        )

    def _has_words(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        try:
            return (
                global_db.query(ReadingWordGloss.id)
                .filter(
                    ReadingWordGloss.text_id == text_id,
                    ReadingWordGloss.target_lang == target_lang
                )
                .first()
                is not None
            )
        except Exception:
            return False

    def _has_sentences(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        try:
            return (
                global_db.query(ReadingTextTranslation.id)
                .filter(
                    ReadingTextTranslation.text_id == text_id,
                    ReadingTextTranslation.target_lang == target_lang,
                    ReadingTextTranslation.unit == TextUnit.SENTENCE,
                )
                .first()
                is not None
            )
        except Exception:
            return False

    def evaluate(
        self,
        global_db: Session,
        rt: ReadingText,
        target_lang: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Evaluate if a text is ready to be shown."""
        if not getattr(rt, "content", None):
            return (False, "no_content")
        
        # Use text's target_lang if not provided
        tgt = target_lang or rt.target_lang
        
        has_w = self._has_words(global_db, rt.id, tgt)
        has_s = self._has_sentences(global_db, rt.id, tgt)
        
        # Full readiness: both words and sentences present
        if has_w and has_s:
            return (True, "both")
        
        # Also check the completion flags
        if getattr(rt, "words_complete", False) and getattr(rt, "sentences_complete", False):
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

    def get_failed_components(
        self,
        global_db: Session,
        text_id: int,
        target_lang: str,
    ) -> dict:
        """Return dict with missing components: {'words': bool, 'sentences': bool}"""
        if not global_db:
            return {"words": False, "sentences": False}
        
        has_words = self._has_words(global_db, text_id, target_lang)
        has_sentences = self._has_sentences(global_db, text_id, target_lang)
        
        return {
            "words": not has_words,
            "sentences": not has_sentences
        }

    def needs_retry(
        self,
        global_db: Session,
        text_id: int,
        target_lang: str,
    ) -> bool:
        """Check if text has any missing components that need retry"""
        failed = self.get_failed_components(global_db, text_id, target_lang)
        return failed["words"] or failed["sentences"]

    def force_once(
        self,
        account_db: Session,
        account_id: int,
        lang: str,
        ttl_s: int = 60,
    ) -> None:
        exp = datetime.utcnow() + timedelta(seconds=max(1, int(ttl_s)))
        row = (
            account_db.query(NextReadyOverride)
            .filter(NextReadyOverride.account_id == account_id, NextReadyOverride.lang == lang)
            .first()
        )
        if row:
            row.expires_at = exp
        else:
            account_db.add(NextReadyOverride(account_id=account_id, lang=lang, expires_at=exp))
        try:
            account_db.commit()
        except Exception:
            account_db.rollback()

    def consume_if_valid(
        self,
        account_db: Session,
        account_id: int,
        lang: str,
    ) -> bool:
        now = datetime.utcnow()
        row = (
            account_db.query(NextReadyOverride)
            .filter(NextReadyOverride.account_id == account_id, NextReadyOverride.lang == lang)
            .first()
        )
        if not row:
            return False
        valid = bool(row.expires_at is None or row.expires_at > now)
        try:
            account_db.delete(row)
            account_db.commit()
        except Exception:
            account_db.rollback()
        return valid
