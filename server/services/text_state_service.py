"""
Text State Service

Consolidated service for text selection, readiness evaluation, and progress tracking.
Merges SelectionService, ReadinessService, and ProgressService into a single cohesive service.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..models import (
    Profile, 
    ReadingText, 
    ProfileTextRead,
    ReadingLookup,
    ReadingWordGloss,
    ReadingTextTranslation,
    NextReadyOverride,
)
from ..enums import TextUnit
from ..settings import get_settings

logger = logging.getLogger(__name__)


def _session_log_dir_root() -> Path:
    """Base directory for session logs"""
    base = os.getenv("ARC_OR_LOG_DIR", str(Path.cwd() / "data" / "session_logs"))
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_log_path(account_id: int, lang: str) -> Path:
    """Generate timestamped directory for a session log"""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d = _session_log_dir_root() / str(int(account_id)) / lang / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class ReadinessStatus:
    """Status of text readiness."""
    ready: bool
    text_id: Optional[int]
    reason: str
    status: str
    retry_info: Optional[Dict[str, Any]] = None


class TextStateService:
    """
    Consolidated service for text state management.
    
    Handles:
    - Text selection from global pool
    - Current text management
    - Text readiness evaluation
    - Progress tracking and analytics
    - Session recording
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    # SELECTION METHODS
    # -----------------------------------------------------------------
    
    def pick_current_or_new(
        self,
        account_db: Session,  # Per-account DB for Profile, ProfileTextRead
        global_db: Session,   # Global DB for ReadingText
        account_id: int,
        lang: str,
    ) -> Optional[ReadingText]:
        """
        Get the current text or pick a new one from global pool.
        
        Logic:
        1. If profile.current_text_id exists, return that text
        2. Otherwise pick next ready text from global pool (matching lang/target_lang)
        3. Skip texts the user has already read (unless reread is allowed)
        """
        prof = account_db.query(Profile).filter(
            Profile.account_id == account_id,
            Profile.lang == lang
        ).first()
        if not prof:
            return None
        
        target_lang = prof.target_lang
        
        # Case 1: User has a current_text_id - return that text
        if getattr(prof, "current_text_id", None):
            rt = global_db.get(ReadingText, int(prof.current_text_id))
            if rt and rt.lang == lang and rt.target_lang == target_lang:
                return rt
        
        # Get IDs of texts this profile has read
        read_text_ids = set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == prof.id)
            .all()
        )
        
        # Case 2: Find a ready text from global pool
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
                *ReadingText.ready_filter()
            )
        )
        
        # Filter out already-read texts (unless reread is allowed)
        if read_text_ids:
            query = query.filter(~ReadingText.id.in_(read_text_ids))
        
        rt = query.order_by(ReadingText.created_at.asc()).first()
        
        if rt:
            try:
                # Set current_text_id on profile
                prof.current_text_id = rt.id
                account_db.commit()
                logger.info(f"Set current_text_id={rt.id} for profile_id={prof.id}")
                return rt
            except Exception as e:
                logger.error(f"Failed to set current_text_id: {e}")
                account_db.rollback()
        
        # Case 3: No ready texts available - return None (will trigger generation)
        return None
    
    def get_next_text(
        self,
        account_db: Session,
        global_db: Session,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """
        Get the next unread text for a profile.
        
        Used when user clicks "Next" and we need to find another text.
        """
        # Get IDs of texts this profile has read
        read_text_ids = set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == profile.id)
            .all()
        )
        
        # Filter by language, target language, and readiness
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                *ReadingText.ready_filter()
            )
        )
        
        # Filter out already-read texts (unless reread is allowed)
        if read_text_ids:
            query = query.filter(~ReadingText.id.in_(read_text_ids))
        
        # Also exclude current text to avoid picking the same one
        if profile.current_text_id:
            query = query.filter(ReadingText.id != profile.current_text_id)
        
        rt = query.order_by(ReadingText.created_at.asc()).first()
        
        if rt:
            try:
                # Set current_text_id on profile
                profile.current_text_id = rt.id
                account_db.commit()
                logger.info(f"Set current_text_id={rt.id} for next text, profile_id={profile.id}")
                return rt
            except Exception as e:
                logger.error(f"Failed to set current_text_id for next text: {e}")
                account_db.rollback()
        
        return None
    
    def clear_current_text(self, account_db: Session, profile: Profile) -> None:
        """Clear the current text from a profile."""
        try:
            profile.current_text_id = None
            account_db.commit()
            logger.info(f"Cleared current_text_id for profile_id={profile.id}")
        except Exception as e:
            logger.error(f"Failed to clear current_text_id: {e}")
            account_db.rollback()
    
    # READINESS METHODS
    # -----------------------------------------------------------------
    
    def check_text_ready(self, global_db: Session, text_id: int) -> ReadinessStatus:
        """Check if a specific text is ready for reading."""
        text = global_db.get(ReadingText, text_id)
        if not text:
            return ReadinessStatus(
                ready=False,
                text_id=None,
                reason="text_not_found",
                status="error"
            )
        
        # Use the is_ready property from the model
        if text.is_ready:
            return ReadinessStatus(
                ready=True,
                text_id=text.id,
                reason="complete",
                status="ready"
            )
        else:
            return ReadinessStatus(
                ready=False,
                text_id=text.id,
                reason="incomplete_translations",
                status="translating"
            )
    
    def check_next_ready(self, global_db: Session, account_id: int, lang: str) -> ReadinessStatus:
        """
        Check if there's a next ready text for the user.
        
        Returns status and optionally retry information if generation is in progress.
        """
        # Check for manual override first
        override = global_db.query(NextReadyOverride).filter(
            NextReadyOverride.account_id == account_id
        ).first()
        
        if override:
            # Check if override is still valid
            if override.expires_at and override.expires_at > datetime.now(timezone.utc):
                text = global_db.get(ReadingText, override.text_id)
                if text and text.is_ready:
                    return ReadinessStatus(
                        ready=True,
                        text_id=text.id,
                        reason="manual_override",
                        status="ready"
                    )
            else:
                # Override expired, remove it
                global_db.delete(override)
                global_db.commit()
        
        # Look for the next ready text
        text = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                *ReadingText.ready_filter()
            )
            .order_by(ReadingText.created_at.asc())
            .first()
        )
        
        if text:
            return ReadinessStatus(
                ready=True,
                text_id=text.id,
                reason="ready_available",
                status="ready"
            )
        
        # No ready text - check if there's one being generated
        generating_text = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.content.isnot(None),
                ~ReadingText.words_complete | ~ReadingText.sentences_complete,
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )
        
        if generating_text:
            retry_info = {
                "text_id": generating_text.id,
                "translation_attempts": generating_text.translation_attempts,
                "last_attempt": generating_text.last_translation_attempt.isoformat() if generating_text.last_translation_attempt else None,
            }
            
            return ReadinessStatus(
                ready=False,
                text_id=generating_text.id,
                reason="incomplete_translations",
                status="translating",
                retry_info=retry_info
            )
        
        # No texts at all - need to generate
        return ReadinessStatus(
            ready=False,
            text_id=None,
            reason="no_texts_available",
            status="generating",
            retry_info={}
        )
    
    def has_words(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        """Check if a text has word translations."""
        return global_db.query(ReadingWordGloss).filter(
            ReadingWordGloss.text_id == text_id,
            ReadingWordGloss.target_lang == target_lang,
        ).first() is not None
    
    def has_sentences(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        """Check if a text has sentence translations."""
        return global_db.query(ReadingTextTranslation).filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.target_lang == target_lang,
            ReadingTextTranslation.unit == "sentence",
        ).first() is not None
    
    def get_manual_override(self, global_db: Session, account_id: int) -> Optional[int]:
        """Get manual override text ID for an account."""
        override = global_db.query(NextReadyOverride).filter(
            NextReadyOverride.account_id == account_id
        ).first()
        
        if override:
            # Check if override is still valid
            if override.expires_at and override.expires_at > datetime.now(timezone.utc):
                return override.text_id
            else:
                # Override expired, remove it
                global_db.delete(override)
                global_db.commit()
        
        return None
    
    def set_manual_override(
        self, 
        global_db: Session, 
        account_id: int, 
        text_id: int, 
        expires_seconds: int = 3600
    ) -> None:
        """Set a manual override for next ready text."""
        # Remove any existing override
        global_db.query(NextReadyOverride).filter(
            NextReadyOverride.account_id == account_id
        ).delete()
        
        # Create new override
        override = NextReadyOverride(
            account_id=account_id,
            text_id=text_id,
            expires_at=datetime.now(timezone.utc).replace(microsecond=0) + \
                     datetime.timedelta(seconds=expires_seconds)
        )
        global_db.add(override)
        global_db.commit()
        
        logger.info(f"Set manual override: text_id={text_id} for account_id={account_id}")
    
    # PROGRESS METHODS
    # -----------------------------------------------------------------
    
    def mark_text_read(
        self,
        account_db: Session,
        global_db: Session,
        account_id: int,
        text_id: int,
    ) -> bool:
        """
        Mark a text as read for a profile.
        """
        profile = account_db.query(Profile).filter(
            Profile.account_id == account_id
        ).first()
        
        if not profile:
            logger.error(f"No profile found for account_id={account_id}")
            return False
        
        try:
            # Check if already recorded
            existing = account_db.query(ProfileTextRead).filter(
                ProfileTextRead.profile_id == profile.id,
                ProfileTextRead.text_id == text_id,
            ).first()
            
            if existing:
                # Update last read time
                existing.last_read_at = datetime.now(timezone.utc)
                existing.read_count += 1
            else:
                # Create new record
                record = ProfileTextRead(
                    profile_id=profile.id,
                    text_id=text_id,
                    read_count=1,
                    first_read_at=datetime.now(timezone.utc),
                    last_read_at=datetime.now(timezone.utc),
                )
                account_db.add(record)
            
            account_db.commit()
            
            # Clear current_text_id from profile
            profile.current_text_id = None
            account_db.commit()
            
            logger.info(f"Marked text_id={text_id} as read for profile_id={profile.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark text as read: {e}")
            account_db.rollback()
            return False
    
    def record_session(
        self,
        db: Session,
        account_id: int,
        payload: Dict[str, Any],
    ) -> None:
        """
        Record session data for analytics and progress tracking.
        
        This is simplified compared to the full implementation and focuses on
        core progress tracking.
        """
        try:
            # Extract basic session info
            text_id = payload.get("text_id")
            if not text_id:
                logger.warning("Session payload missing text_id")
                return
            
            # Record word lookups if available
            lookups = payload.get("lookups", [])
            for lookup in lookups:
                try:
                    surface = lookup.get("surface")
                    span_start = lookup.get("span_start")
                    span_end = lookup.get("span_end")
                    
                    if not all([surface, span_start is not None, span_end is not None]):
                        continue
                    
                    # Check if lookup already recorded
                    existing = db.query(ReadingLookup).filter(
                        ReadingLookup.account_id == account_id,
                        ReadingLookup.text_id == text_id,
                        ReadingLookup.surface == surface,
                        ReadingLookup.span_start == span_start,
                        ReadingLookup.span_end == span_end,
                    ).first()
                    
                    if not existing:
                        record = ReadingLookup(
                            account_id=account_id,
                            text_id=text_id,
                            surface=surface,
                            span_start=span_start,
                            span_end=span_end,
                            lemma=lookup.get("lemma"),
                            pos=lookup.get("pos"),
                            translations=lookup.get("translations", {}),
                        )
                        db.add(record)
                
                except Exception as e:
                    logger.warning(f"Failed to record lookup: {e}")
                    continue
            
            db.commit()
            logger.info(f"Recorded session for account_id={account_id}, text_id={text_id}")
            
        except Exception as e:
            logger.error(f"Failed to record session: {e}")
            db.rollback()
    
    def get_reading_stats(
        self,
        account_db: Session,
        account_id: int,
    ) -> Dict[str, Any]:
        """
        Get reading statistics for a user.
        """
        try:
            profile = account_db.query(Profile).filter(
                Profile.account_id == account_id
            ).first()
            
            if not profile:
                return {}
            
            # Total texts read
            total_read = account_db.query(ProfileTextRead).filter(
                ProfileTextRead.profile_id == profile.id
            ).count()
            
            # Total word lookups
            total_lookups = account_db.query(ReadingLookup).filter(
                ReadingLookup.account_id == account_id
            ).count()
            
            # Recent activity (last 30 days)
            cutoff = datetime.now(timezone.utc) - datetime.timedelta(days=30)
            recent_lookups = account_db.query(ReadingLookup).filter(
                ReadingLookup.account_id == account_id,
                ReadingLookup.created_at >= cutoff,
            ).count()
            
            return {
                "total_texts_read": total_read,
                "total_word_lookups": total_lookups,
                "recent_lookups_30d": recent_lookups,
                "profile_level": profile.level_value,
                "profile_lang": profile.lang,
            }
            
        except Exception as e:
            logger.error(f"Failed to get reading stats: {e}")
            return {}
    
    def save_session_to_file(
        self,
        account_id: int,
        lang: str,
        payload: Dict[str, Any],
    ) -> Optional[Path]:
        """
        Save session data to file for later processing.
        
        This is a simplified version that saves basic session data.
        """
        try:
            log_dir = _session_log_path(account_id, lang)
            
            session_file = log_dir / "session.json"
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, default=str)
            
            return log_dir
            
        except Exception as e:
            logger.error(f"Failed to save session to file: {e}")
            return None


# LEGACY COMPATIBILITY
    # These methods provide backward compatibility with the old unified services
    
    def _has_words(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        """Legacy method - use has_words instead"""
        return self.has_words(global_db, text_id, target_lang)
    
    def _has_sentences(self, global_db: Session, text_id: int, target_lang: str) -> bool:
        """Legacy method - use has_sentences instead"""
        return self.has_sentences(global_db, text_id, target_lang)
    
    def evaluate(self, global_db: Session, text_or_id, target_lang: str) -> tuple[bool, str]:
        """Legacy method - use check_text_ready instead. Returns (ready, reason) tuple for backward compatibility."""
        # Handle both ReadingText object and text_id
        if hasattr(text_or_id, 'id'):
            text_id = text_or_id.id
        else:
            text_id = text_or_id
        
        status = self.check_text_ready(global_db, text_id)
        return (status.ready, status.reason)


# Singleton instance
_service = None

def get_text_state_service() -> TextStateService:
    """Get the singleton TextStateService instance."""
    global _service
    if _service is None:
        _service = TextStateService()
    return _service
