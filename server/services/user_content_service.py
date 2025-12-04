"""
User Content Service - Consolidated content selection, readiness, and progress management.

This service consolidates:
- SelectionService: Text selection from pool and current text management  
- ReadinessService: Text readiness evaluation and status checking
- ProgressService: Session recording and analytics

Responsibilities:
- Select appropriate texts for users (current and next)
- Evaluate text readiness and completion status
- Track user progress and analytics
- Manage reading session lifecycle
- Handle text pool interactions and user reading history
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, Set, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models import (
    Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss, 
    NextReadyOverride, ProfileTextRead, ReadingLookup
)
from ..enums import TextUnit
from ..settings import get_settings
from ..utils.migrations import ensure_reading_text_lifecycle_columns

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


class UserContentService:
    """
    Consolidated service for user content management.
    
    This service manages:
    1. Text selection from global pool
    2. Text readiness evaluation and status tracking
    3. User progress tracking and analytics
    4. Session lifecycle management
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    # SELECTION METHODS
    
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
                ReadingText.content.isnot(None),
                ReadingText.content != "",
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
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
        read_text_ids = set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == profile.id)
            .all()
        )
        
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                ReadingText.content.isnot(None),
                ReadingText.content != "",
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        if read_text_ids:
            query = query.filter(~ReadingText.id.in_(read_text_ids))
        
        return query.order_by(ReadingText.created_at.asc()).first()
    
    def first_ready_backup(
        self,
        global_db: Session,
        account_db: Session,
        lang: str,
        target_lang: Optional[str] = None,
        profile_id: Optional[int] = None,
        exclude_text_id: Optional[int] = None,
    ) -> Tuple[Optional[ReadingText], str]:
        """
        Find any ready backup text from the global pool.
        Returns (text, reason) where reason is 'both', 'grace', etc.
        """
        read_ids = set()
        if profile_id:
            read_ids = self._get_read_text_ids(account_db, profile_id)
        
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.content.isnot(None),
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        # Add target_lang filter if provided
        if target_lang:
            query = query.filter(ReadingText.target_lang == target_lang)
        
        if read_ids:
            query = query.filter(~ReadingText.id.in_(read_ids))
        
        if exclude_text_id:
            query = query.filter(ReadingText.id != exclude_text_id)
        
        text = query.first()
        if text:
            return text, "both"
        
        return None, "waiting"
    
    def mark_text_read(
        self,
        account_db: Session,
        profile: Profile,
        text_id: int,
    ) -> None:
        """Mark a text as read by the profile."""
        now = datetime.now(timezone.utc)
        
        existing = account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile.id,
            ProfileTextRead.text_id == text_id
        ).first()
        
        if existing:
            existing.read_count += 1
            existing.last_read_at = now
        else:
            account_db.add(ProfileTextRead(
                profile_id=profile.id,
                text_id=text_id,
                read_count=1,
                first_read_at=now,
                last_read_at=now,
            ))
        
        account_db.flush()
    
    # READINESS METHODS
    
    def check_text_ready(self, global_db: Session, text_id: int) -> 'ReadinessStatus':
        """Check if a specific text is ready for reading."""
        try:
            text = global_db.get(ReadingText, text_id)
            if not text:
                return ReadinessStatus(ready=False, text_id=text_id, reason="not_found")
            
            ready, reason = self.evaluate(global_db, text)
            return ReadinessStatus(
                ready=ready,
                text_id=text_id,
                reason=reason,
                status="ready" if ready else self._determine_status(global_db, text_id)
            )
            
        except Exception as e:
            logger.error(f"Error checking text readiness for text_id={text_id}: {e}")
            return ReadinessStatus(ready=False, text_id=text_id, reason="check_failed")
    
    def check_next_ready(self, global_db: Session, account_id: int, lang: str) -> 'ReadinessStatus':
        """Check if there's a next text ready after consuming current one."""
        try:
            # Check for manual override first
            with self._get_account_db(account_id) as account_db:
                profile = account_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
                if profile and profile.target_lang:
                    # Check if there are any ready texts
                    read_ids = self._get_read_text_ids(account_db, profile.id)
                    
                    query = global_db.query(ReadingText).filter(
                        ReadingText.lang == lang,
                        ReadingText.target_lang == profile.target_lang,
                        ReadingText.content.isnot(None),
                        ReadingText.words_complete == True,
                        ReadingText.sentences_complete == True,
                    )
                    
                    if read_ids:
                        query = query.filter(~ReadingText.id.in_(read_ids))
                    
                    next_text = query.order_by(ReadingText.created_at.asc()).first()
                    
                    if next_text:
                        return ReadinessStatus(ready=True, text_id=next_text.id, reason="available")
            
            return ReadinessStatus(ready=False, text_id=None, reason="no_texts_available")
            
        except Exception as e:
            logger.error(f"Error checking next ready: {e}")
            return ReadinessStatus(ready=False, text_id=None, reason="check_failed")
    
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
    
    # PROGRESS METHODS
    
    def record_session(self, db: Session, account_id: int, payload: Dict[str, Any]) -> None:
        """Process complete session data including word lookups, translations, and analytics."""
        if not payload or not isinstance(payload, dict):
            return
            
        try:
            logger.info(f"Recording session for account {account_id}: {payload.get('session_id')}")
            
            # Extract session metadata
            session_id = payload.get('session_id')
            text_id = payload.get('text_id')
            lang = payload.get('lang')
            target_lang = payload.get('target_lang')
            opened_at = payload.get('opened_at')
            analytics = payload.get('analytics', {})
            
            if not text_id or not session_id:
                logger.warning(f"Missing required session fields: {list(payload.keys())}")
                return
                
            # Verify text belongs to account
            text = db.get(ReadingText, text_id)
            if not text or text.account_id != account_id:
                logger.warning(f"Text {text_id} not found for account {account_id}")
                return
                
            # Process word lookups
            self._process_word_lookups(db, account_id, text_id, payload)
            
            # Store session analytics and save session data to files
            self._store_analytics(db, account_id, text_id, session_id, analytics)
            self._save_session_to_disk(account_id, lang, session_id, payload)
            
            logger.info(f"Successfully processed session {session_id} for account {account_id}")
            
        except Exception as e:
            logger.error(f"Error recording session for account {account_id}: {e}", exc_info=True)
    
    def complete_and_mark_read(self, db: Session, account_id: int, prior_text_id: Optional[int]) -> None:
        """Mark a text as read and record completion."""
        if not prior_text_id:
            return
        try:
            rt = db.get(ReadingText, int(prior_text_id))
            if rt and rt.account_id == int(account_id):
                rt.is_read = True
                rt.read_at = datetime.utcnow()
                db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
    
    # PRIVATE HELPER METHODS
    
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
                    # Note: ReadingWordGloss doesn't have target_lang in this schema
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
                    # Note: ReadingTextTranslation doesn't have target_lang in this schema
                    ReadingTextTranslation.unit == TextUnit.SENTENCE,
                )
                .first()
                is not None
            )
        except Exception:
            return False
    
    def _determine_status(self, global_db: Session, text_id: int, target_lang: Optional[str] = None) -> str:
        """Determine detailed status for unreadiness."""
        try:
            text = global_db.get(ReadingText, text_id)
            if not text:
                return "not_found"
            
            if not getattr(text, "content", None):
                return "no_content"
            
            tgt = target_lang or text.target_lang
            has_w = self._has_words(global_db, text_id, tgt)
            has_s = self._has_sentences(global_db, text_id, tgt)
            
            if has_w and has_s:
                return "ready"
            elif has_w and not has_s:
                return "missing_sentences"
            elif has_s and not has_w:
                return "missing_words"
            else:
                return "no_translations"
                
        except Exception:
            return "error"
    
    def _get_account_db(self, account_id: int):
        """Helper to get account DB session."""
        from ..utils.session_manager import db_manager
        return db_manager.get_db(account_id)
    
    def _process_word_lookups(self, db: Session, account_id: int, text_id: int, payload: Dict[str, Any]) -> None:
        """Process word lookup data from session."""
        lookups = []
        
        # Extract word lookups from title
        title = payload.get('title', {})
        if title and title.get('words'):
            for word in title['words']:
                if word.get('looked_up_at'):
                    lookups.append({
                        'word': word,
                        'is_title': True
                    })
        
        # Extract word lookups from paragraphs
        paragraphs = payload.get('paragraphs', [])
        for para in paragraphs:
            sentences = para.get('sentences', [])
            for sentence in sentences:
                words = sentence.get('words', [])
                for word in words:
                    if word.get('looked_up_at'):
                        lookups.append({
                            'word': word,
                            'is_title': False,
                            'sentence_text': sentence.get('text', ''),
                            'paragraph_text': para.get('text', '')
                        })
        
        # Create ReadingLookup records
        for lookup in lookups:
            word_data = lookup['word']
            try:
                record = ReadingLookup(
                    account_id=account_id,
                    text_id=text_id,
                    surface=word_data.get('surface'),
                    lemma=word_data.get('lemma'),
                    pos=word_data.get('pos'),
                    translations=[word_data.get('translation')] if word_data.get('translation') else [],
                    target_lang=payload.get('target_lang', 'en'),
                    created_at=datetime.fromtimestamp(word_data.get('looked_up_at', 0) / 1000.0)
                )
                db.add(record)
            except Exception as e:
                logger.debug(f"Error creating lookup record: {e}")
                continue
                
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving lookup records: {e}")
    
    def _store_analytics(self, db: Session, account_id: int, text_id: int, session_id: str, analytics: Dict[str, Any]) -> None:
        """Store session analytics data."""
        logger.info(f"Session Analytics - Text {text_id}, Account {account_id}:")
        logger.info(f"  Total words: {analytics.get('total_words', 0)}")
        logger.info(f"  Words looked up: {analytics.get('words_looked_up', 0)}")
        logger.info(f"  Lookup rate: {analytics.get('lookup_rate', 0):.3f}")
        logger.info(f"  Reading time (ms): {analytics.get('reading_time_ms', 0)}")
        logger.info(f"  Reading speed (WPM): {analytics.get('average_reading_speed_wpm', 0)}")
        logger.info(f"  Completion status: {analytics.get('completion_status', 'unknown')}")
    
    def _save_session_to_disk(self, account_id: int, lang: str, session_id: str, payload: Dict[str, Any]) -> None:
        """Save complete session data to disk similar to LLM stream logs"""
        try:
            session_dir = _session_log_path(account_id, lang)
            
            # Save meta information
            meta = {
                "account_id": account_id,
                "lang": lang,
                "session_id": session_id,
                "text_id": payload.get('text_id'),
                "created_at": datetime.utcnow().isoformat()
            }
            meta_file = session_dir / "meta.json"
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save complete session data
            session_file = session_dir / "session.json"
            session_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # Save analytics separately for easy access
            analytics_file = session_dir / "analytics.json"
            analytics = payload.get('analytics', {})
            if analytics:
                analytics_file.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding="utf-8")
            
            logger.debug(f"Saved session data to {session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save session data to disk: {e}", exc_info=True)


# Helper class for backward compatibility
class ReadinessStatus:
    def __init__(self, ready: bool, text_id: Optional[int], reason: str = "waiting", retry_info: Optional[Dict[str, Any]] = None, status: str = "waiting"):
        self.ready = ready
        self.text_id = text_id
        self.reason = reason
        self.retry_info = retry_info
        self.status = status


# Global instance for backward compatibility
def get_user_content_service() -> UserContentService:
    """Get the singleton UserContentService instance."""
    return UserContentService()
