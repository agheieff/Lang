"""
Session Management Service - Consolidated user session and reading context management.

This service consolidates:
- SessionProcessingService: User interaction processing from local storage
- ReadingViewService: Reading view data preparation and context management
- ReconstructionService: Data reconstruction from logs

Responsibilities:
- Process user session data (word lookups, interactions, learning progress)
- Prepare reading view context data
- Handle text selection and readiness state
- Reconstruct missing data from logs when needed
- Manage user level updates and CI preference adaptation
"""

import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    ReadingText,
    ReadingWordGloss,
    ReadingTextTranslation,
    ReadingLookup,
    ProfileTextRead,
    Lexeme,
)
from ..schemas.session import TextSessionState
from ..services.srs_service import SrsService
from ..services.text_state_service import get_text_state_service, ReadinessStatus
from ..services.title_extraction_service import TitleExtractionService

logger = logging.getLogger(__name__)


@dataclass
class ReadingContext:
    status: Literal["loading", "generating", "ready", "error"]
    text_id: Optional[int] = None
    content: Optional[str] = None
    words: List[ReadingWordGloss] = field(default_factory=list)
    title: Optional[str] = None
    title_words: List[Any] = field(default_factory=list)
    title_translation: Optional[str] = None
    is_fully_ready: bool = False
    sse_endpoint: Optional[str] = None
    session_state: Optional[Dict] = None
    is_next_ready: bool = False
    next_ready_reason: str = "waiting"


class SessionManagementService:
    """
    Consolidated service for user session and reading context management.
    
    This service manages:
    1. User session data processing (interactions, lookups, learning progress)
    2. Reading view preparation and context state
    3. Data reconstruction from logs when needed
    4. Text selection and readiness management
    """
    
    def __init__(self):
        self.srs_service = SrsService()
        self.user_content_service = get_text_state_service()
        self.title_service = TitleExtractionService()
    
    def process_session_data(
        self, 
        db: Session, 
        account_id: int, 
        text_id: int,
        session_data: Dict[str, Any]
    ) -> bool:
        """
        Process session data from local storage when user moves to next text.
        
        Args:
            db: Database session (per-account)
            account_id: User account ID
            text_id: Text ID that was just completed
            session_data: Session data containing lookups, interactions, time_spent, etc.
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            logger.info(f"Processing session data for account_id={account_id} text_id={text_id}")
            
            # Process word lookups
            lookups_processed = self._process_word_lookups(db, account_id, text_id, session_data.get("lookups", []))
            
            # Update SRS data
            srs_updated = self._update_srs_data(db, account_id, session_data)
            
            # Record user progress
            progress_updated = self._record_progress(db, account_id, text_id, session_data)
            
            # Update user level if needed
            self._update_user_level(db, account_id)
            
            # Adapt CI preference based on lookup rate
            self._adapt_ci_preference(db, account_id, text_id, session_data)
            
            logger.info(f"Session processing completed: lookups={lookups_processed} srs={srs_updated} progress={progress_updated}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing session data: {e}")
            try:
                db.rollback()
            except Exception:
                pass
            return False
    
    def persist_session_state(self, db: Session, account_id: int, state: TextSessionState) -> bool:
        """Persist session state from client."""
        try:
            # This would typically save the session state to database
            # For now, just log and return success
            logger.info(f"Persisting session state for account_id={account_id}")
            return True
        except Exception as e:
            logger.error(f"Error persisting session state: {e}")
            return False
    
    def get_current_reading_context(
        self,
        account_db: Session,  # Per-account DB
        global_db: Session,   # Global DB
        account_id: int,
    ) -> ReadingContext:
        """
        Prepare all data needed to render the current reading view.
        
        Args:
            account_db: Per-account database session
            global_db: Global database session
            account_id: User account ID
        
        Returns:
            ReadingContext with all necessary data for rendering
        """
        start_time = time.time()
        
        # Get user profile (per-account)
        prof = account_db.query(Profile).filter(Profile.account_id == account_id).first()
        if not prof:
            logger.error(f"Profile not found for account_id={account_id}")
            return ReadingContext(status="error")

        lang = prof.lang
        target_lang = prof.target_lang

        # Select text (current or new)
        text_obj = None
        try:
            text_obj = self.user_content_service.pick_current_or_new(
                account_db, global_db, account_id, lang
            )
        except Exception as e:
            logger.error(f"text_state_service failed: {e}")
            # Note: In the new consolidated approach, we might not have direct access to unopened/generating states
            # For now, return None to trigger generation
            text_obj = None

        # Determine status if no text
        if text_obj is None:
            # Check readiness status directly (this would normally detect generating state)
            try:
                readiness_status = self.user_content_service.check_next_ready(global_db, account_id, lang)
                if readiness_status.text_id:
                    return ReadingContext(status="generating")
            except Exception:
                pass
            return ReadingContext(status="loading", is_next_ready=True, next_ready_reason="trigger_generation")

        # We have a text object
        text_id = text_obj.id
        content = text_obj.content
        title = text_obj.title

        # Check readiness status
        try:
            readiness = self.user_content_service.check_text_ready(global_db, text_id)
        except Exception as e:
            logger.error(f"Readiness check failed for text_id={text_id}: {e}")
            readiness = ReadinessStatus(ready=False, text_id=text_id, reason="check_failed")

        if not readiness.ready:
            # Text not ready yet
            return ReadingContext(
                status="generating" if readiness.status == "generating" else "loading",
                text_id=text_id,
                content=content,
                is_next_ready=False,
                next_ready_reason=readiness.reason or "waiting",
            )

        # Text is ready - load full data
        try:
            # Load words from global DB
            words = global_db.query(ReadingWordGloss).filter(ReadingWordGloss.text_id == text_id).all()
            
            # Load translations if needed
            translations = global_db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.text_id == text_id
            ).all()
            
            # Setup title words and translation
            title_words = []
            title_translation = None
            if title and words:
                # Create title words from existing word glosses
                title_word_map = {w.word: w for w in words}
                for title_word in title.split():
                    if title_word in title_word_map:
                        title_words.append(title_word_map[title_word])
            
            # Check for title translations
            if translations:
                # Look for title translation (typically sentence_index 0 might contain title)
                for trans in translations:
                    if hasattr(trans, 'translation') and trans.translation:
                        title_translation = trans.translation
                        break

            # Check if fully ready (should be if readiness check passed)
            is_fully_ready = readiness.status == "ready"
            
            # Get session state for the text
            session_state = self._get_session_state_for_text(account_db, account_id, text_id)

            # Check if next text is ready
            next_ready_status = self.user_content_service.check_next_ready(global_db, account_id, lang)

            duration = time.time() - start_time
            logger.info(f"Reading context prepared in {duration:.3f}s for text_id={text_id}")

            return ReadingContext(
                status="ready",
                text_id=text_id,
                content=content,
                words=words,
                title=title,
                title_words=title_words,
                title_translation=title_translation,
                is_fully_ready=is_fully_ready,
                session_state=session_state,
                is_next_ready=next_ready_status.ready,
                next_ready_reason=next_ready_status.reason or "ready",
            )

        except Exception as e:
            logger.error(f"Error preparing reading context for text_id={text_id}: {e}")
            return ReadingContext(status="error", text_id=text_id)

    def ensure_words_from_logs(self, db: Session, account_id: int, text_id: int, *, text: Optional[str] = None, lang: Optional[str] = None) -> None:
        """Reconstruct word glosses from generation logs when data is missing."""
        if not text or not lang:
            rt = db.get(ReadingText, int(text_id))
            if not rt:
                return
            text = rt.content or ""
            lang = rt.lang
        
        try:
            from ..utils.gloss import reconstruct_glosses_from_logs
            reconstruct_glosses_from_logs(db, account_id=account_id, text_id=text_id, text=text or "", lang=str(lang), prefer_db=True)
        except Exception as e:
            logger.error(f"Failed to reconstruct words from logs for text_id={text_id}: {e}")

    def ensure_sentence_translations_from_logs(self, db: Session, account_id: int, text_id: int) -> None:
        """Reconstruct sentence translations from generation logs when data is missing."""
        rt = db.get(ReadingText, int(text_id))
        if not rt:
            return
        
        try:
            # This would typically be in a reading service, but we'll implement it inline
            from ..utils.gloss import reconstruct_sentences_from_logs
            reconstruct_sentences_from_logs(db, int(account_id), rt)
        except Exception as e:
            logger.error(f"Failed to reconstruct sentences from logs for text_id={text_id}: {e}")

    def _process_word_lookups(self, db: Session, account_id: int, text_id: int, lookups: List[Dict]) -> int:
        """Process word lookup events from session data."""
        if not text_id or text_id <= 0:
            return 0

        processed = 0
        for lookup_data in lookups:
            try:
                word = lookup_data.get("word")
                if not word:
                    continue

                timestamp_str = lookup_data.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except Exception:
                        timestamp = datetime.utcnow()
                else:
                    timestamp = datetime.utcnow()

                lookup = ReadingLookup(
                    account_id=account_id,
                    text_id=text_id,
                    surface=word,  # Changed from 'word' to 'surface' to match model
                    context_hash=lookup_data.get("context_hash"),
                    created_at=timestamp,
                )
                db.add(lookup)
                processed += 1

            except Exception as e:
                logger.warning(f"Failed to process lookup event: {e}")
                continue

        if processed > 0:
            db.commit()
        
        return processed

    def _update_srs_data(self, db: Session, account_id: int, session_data: Dict[str, Any]) -> bool:
        """Update SRS data based on session interactions."""
        try:
            # Extract interaction data for SRS processing
            interactions = session_data.get("interactions", [])
            if not interactions:
                return True  # No interactions to process

            # Convert session interactions to SRS format
            srs_events = []
            for interaction in interactions:
                if interaction.get("type") == "word_interaction":
                    srs_events.append({
                        "word": interaction.get("word"),
                        "action": interaction.get("action"),  # click, lookup, etc.
                        "timestamp": interaction.get("timestamp"),
                        "correct": interaction.get("correct", True),
                    })

            if srs_events:
                # Process SRS events
                profile = db.query(Profile).filter(Profile.account_id == account_id).first()
                if profile:
                    for event in srs_events:
                        self.srs_service.process_word_event(
                            db, account_id, profile.lang, event["word"], event["action"]
                        )

            return True

        except Exception as e:
            logger.error(f"Error updating SRS data: {e}")
            return False

    def _record_progress(self, db: Session, account_id: int, text_id: int, session_data: Dict[str, Any]) -> bool:
        """Record user progress for completed text."""
        try:
            # Mark text as read
            profile = db.query(Profile).filter(Profile.account_id == account_id).first()
            if profile:
                ptr = db.query(ProfileTextRead).filter(
                    ProfileTextRead.profile_id == profile.id,
                    ProfileTextRead.text_id == text_id
                ).first()
                
                if ptr:
                    ptr.last_read_at = datetime.utcnow()
                    ptr.read_count += 1
                else:
                    ptr = ProfileTextRead(
                        profile_id=profile.id,
                        text_id=text_id,
                        last_read_at=datetime.utcnow(),
                    )
                    db.add(ptr)

            db.commit()
            return True

        except Exception as e:
            logger.error(f"Error recording progress: {e}")
            return False

    def _update_user_level(self, db: Session, account_id: int) -> None:
        """Trigger user level update if needed."""
        try:
            profile = db.query(Profile).filter(Profile.account_id == account_id).first()
            if profile:
                from ..level import update_level_if_stale
                update_level_if_stale(profile)
                
                # Update current_text_id to None to allow next text selection
                profile.current_text_id = None
                db.commit()
        except Exception as e:
            logger.error(f"Error updating user level: {e}")

    def _adapt_ci_preference(self, db: Session, account_id: int, text_id: int, session_data: Dict[str, Any]) -> None:
        """Adapt CI preference based on user lookup rate."""
        try:
            profile = db.query(Profile).filter(Profile.account_id == account_id).first()
            if not profile:
                return

            time_spent = session_data.get("time_spent", 0)
            lookup_count = len(session_data.get("lookups", []))
            
            if time_spent > 0 and lookup_count > 0:
                # Calculate lookup rate (lookups per minute)
                lookup_rate = (lookup_count / time_spent) * 60
                
                # Adapt CI preference based on lookup rate
                # Higher lookup rate -> lower CI target (easier texts)
                # Lower lookup rate -> higher CI target (harder texts)
                current_ci = profile.ci_preference
                
                if lookup_rate > 3.0:  # Too many lookups
                    new_ci = max(0.85, current_ci - 0.02)
                elif lookup_rate < 1.0:  # Too few lookups
                    new_ci = min(0.98, current_ci + 0.02)
                else:
                    new_ci = current_ci
                
                if new_ci != current_ci:
                    profile.ci_preference = new_ci
                    db.commit()
                    logger.info(f"Adapted CI preference for account_id={account_id}: {current_ci} -> {new_ci}")

        except Exception as e:
            logger.error(f"Error adapting CI preference: {e}")

    def _get_session_state_for_text(self, db: Session, account_id: int, text_id: int) -> Optional[Dict]:
        """Get session state for a specific text."""
        try:
            # This would load any existing session data for the text
            # For now, return basic state
            return {
                "text_id": text_id,
                "account_id": account_id,
                "started_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting session state for text_id={text_id}: {e}")
            return None


# Global instance for backward compatibility  
def get_session_management_service() -> SessionManagementService:
    """Get the singleton SessionManagementService instance."""
    return SessionManagementService()
