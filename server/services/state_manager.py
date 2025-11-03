"""
Text generation state manager.
Tracks the progression of texts through generation phases.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session
from ..utils.session_manager import db_manager

from ..models import ReadingText, ReadingWordGloss, ReadingTextTranslation


class TextState(str, Enum):
    """States a text can be in during generation - simplified for clarity."""
    NONE = "none"  # No text exists
    GENERATING = "generating"  # Text content is being generated
    CONTENT_READY = "content_ready"  # Text content ready, translations pending
    FULLY_READY = "fully_ready"  # Both content and translations ready
    OPENED = "opened"  # User has opened the text
    READ = "read"  # User moved to next text
    FAILED = "failed"  # Generation failed and retries exhausted


class GenerationStateManager:
    """
    Manages the state of text generation.
    
    This service provides a clean interface for tracking where each text
    is in the generation pipeline, without being tied to the specific
    implementation of generation or translation.
    """
    
    def get_text_state(self, db: Session, account_id: int, text_id: int) -> TextState:
        """Get the current state of a text."""
        rt = db.query(ReadingText).filter(
            ReadingText.id == text_id,
            ReadingText.account_id == account_id
        ).first()
        
        if not rt:
            return TextState.NONE
        
        # Check if text has been read (moved past)
        if rt.read_at:
            return TextState.READ
        
        # Check if text has been opened by user
        if rt.opened_at:
            # Check if we have translations for opened texts
            has_words = db.query(ReadingWordGloss).filter(
                ReadingWordGloss.account_id == account_id,
                ReadingWordGloss.text_id == text_id
            ).first() is not None
            
            has_translations = db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.account_id == account_id,
                ReadingTextTranslation.text_id == text_id
            ).first() is not None
            
            if has_words or has_translations:
                return TextState.FULLY_READY
            else:
                return TextState.OPENED
        
        # Check if we have content
        if not rt.content or not rt.content.strip():
            return TextState.GENERATING
        
        # Check if we have translations
        has_words = db.query(ReadingWordGloss).filter(
            ReadingWordGloss.account_id == account_id,
            ReadingWordGloss.text_id == text_id
        ).first() is not None
        
        has_translations = db.query(ReadingTextTranslation).filter(
            ReadingTextTranslation.account_id == account_id,
            ReadingTextTranslation.text_id == text_id
        ).first() is not None
        
        if has_words or has_translations:
            return TextState.FULLY_READY
        else:
            return TextState.CONTENT_READY
    
    def is_generating(self, db: Session, account_id: int, text_id: int) -> bool:
        """Check if text is currently being generated."""
        return self.get_text_state(db, account_id, text_id) == TextState.GENERATING
    
    def has_content(self, db: Session, account_id: int, text_id: int) -> bool:
        """Check if text has content (may not have translations yet)."""
        state = self.get_text_state(db, account_id, text_id)
        return state in [TextState.CONTENT_READY, TextState.FULLY_READY, TextState.OPENED, TextState.READ]
    
    def is_ready(self, db: Session, account_id: int, text_id: int) -> bool:
        """Check if text is fully ready (content + translations)."""
        state = self.get_text_state(db, account_id, text_id)
        return state in [TextState.FULLY_READY, TextState.OPENED, TextState.READ]
    
    def get_unopened_text(self, db: Session, account_id: int, lang: str) -> Optional[ReadingText]:
        """Get the next unopened text that has content."""
        return db.query(ReadingText).filter(
            ReadingText.account_id == account_id,
            ReadingText.lang == lang,
            ReadingText.opened_at.is_(None),
            ReadingText.content.is_not(None),
            ReadingText.content != ""
        ).order_by(ReadingText.created_at.asc()).first()
    
    def get_generating_text(self, db: Session, account_id: int, lang: str) -> Optional[ReadingText]:
        """Get a text that's currently being generated."""
        return db.query(ReadingText).filter(
            ReadingText.account_id == account_id,
            ReadingText.lang == lang,
            ReadingText.opened_at.is_(None),
            ReadingText.content.is_(None)
        ).order_by(ReadingText.created_at.asc()).first()
    
    def count_unopened_ready(self, db: Session, account_id: int, lang: str) -> int:
        """Count unopened texts that have content but aren't opened yet."""
        return db.query(ReadingText).filter(
            ReadingText.account_id == account_id,
            ReadingText.lang == lang,
            ReadingText.opened_at.is_(None),
            ReadingText.content.is_not(None),
            ReadingText.content != ""
        ).count()
    
    def mark_opened(self, db: Session, account_id: int, text_id: int) -> bool:
        """Mark a text as opened when user first sees it."""
        rt = db.query(ReadingText).filter(
            ReadingText.id == text_id,
            ReadingText.account_id == account_id
        ).first()
        
        if rt and rt.opened_at is None:
            rt.opened_at = datetime.utcnow()
            db.commit()
            return True
        return False
    
    def mark_read(self, db: Session, account_id: int, text_id: int) -> bool:
        """Mark a text as read when user moves to next text."""
        rt = db.query(ReadingText).filter(
            ReadingText.id == text_id,
            ReadingText.account_id == account_id
        ).first()
        
        if rt:
            rt.read_at = datetime.utcnow()
            db.commit()
            return True
        return False
    
    def clear_current_pointer(self, db: Session, account_id: int) -> bool:
        """Clear the profile's current_text_id pointer."""
        from ..models import Profile  # Import here to avoid circular imports
        
        prof = db.query(Profile).filter(Profile.account_id == account_id).first()
        if prof and prof.current_text_id is not None:
            prof.current_text_id = None
            db.commit()
            return True
        return False
    
    def set_current_pointer(self, db: Session, account_id: int, text_id: int) -> bool:
        """Set the profile's current_text_id pointer."""
        from ..models import Profile  # Import here to avoid circular imports
        
        prof = db.query(Profile).filter(Profile.account_id == account_id).first()
        if prof:
            prof.current_text_id = text_id
            db.commit()
            return True
        return False
