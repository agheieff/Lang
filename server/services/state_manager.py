"""
Text generation state manager.
Tracks the progression of texts through generation phases.

Now supports global/per-account DB split:
- global_db: ReadingText, ReadingTextTranslation, ReadingWordGloss
- account_db: Profile, ProfileTextRead
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session

from ..models import ReadingText, ReadingWordGloss, ReadingTextTranslation, ProfileTextRead


class TextState(str, Enum):
    """States a text can be in during generation."""
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
    """
    
    def get_text_state(
        self,
        global_db: Session,
        account_db: Session,
        profile_id: int,
        text_id: int,
        target_lang: str,
    ) -> TextState:
        """Get the current state of a text."""
        rt = global_db.get(ReadingText, text_id)
        
        if not rt:
            return TextState.NONE
        
        # Check if text has been read by this profile
        read_entry = account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile_id,
            ProfileTextRead.text_id == text_id
        ).first()
        
        if read_entry:
            return TextState.READ
        
        # Check if we have content
        if not rt.content or not rt.content.strip():
            return TextState.GENERATING
        
        # Check if we have translations
        has_words = rt.words_complete or global_db.query(ReadingWordGloss).filter(
            ReadingWordGloss.text_id == text_id,
            ReadingWordGloss.target_lang == target_lang
        ).first() is not None
        
        has_translations = rt.sentences_complete or global_db.query(ReadingTextTranslation).filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.target_lang == target_lang
        ).first() is not None
        
        if has_words or has_translations:
            return TextState.FULLY_READY
        else:
            return TextState.CONTENT_READY
    
    def is_generating(
        self,
        global_db: Session,
        text_id: int,
    ) -> bool:
        """Check if text is currently being generated (no content yet)."""
        rt = global_db.get(ReadingText, text_id)
        if not rt:
            return False
        return not rt.content or not rt.content.strip()
    
    def has_content(
        self,
        global_db: Session,
        text_id: int,
    ) -> bool:
        """Check if text has content."""
        rt = global_db.get(ReadingText, text_id)
        if not rt:
            return False
        return bool(rt.content and rt.content.strip())
    
    def is_ready(
        self,
        global_db: Session,
        text_id: int,
    ) -> bool:
        """Check if text is fully ready (content + translations)."""
        rt = global_db.get(ReadingText, text_id)
        if not rt:
            return False
        return bool(
            rt.content and 
            rt.content.strip() and 
            rt.words_complete and 
            rt.sentences_complete
        )
    
    def get_unopened_text(
        self,
        global_db: Session,
        account_id: int,
        lang: str,
    ) -> Optional[ReadingText]:
        """Get the next unopened text that has content.
        
        Note: This method is simplified - it doesn't check per-profile read status.
        Use pick_current_or_new from SelectionService for full logic.
        """
        return global_db.query(ReadingText).filter(
            ReadingText.lang == lang,
            ReadingText.content.is_not(None),
            ReadingText.content != "",
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
        ).order_by(ReadingText.created_at.asc()).first()
    
    def get_generating_text(
        self,
        global_db: Session,
        account_id: int,
        lang: str,
    ) -> Optional[ReadingText]:
        """Get a text that's currently being generated."""
        return global_db.query(ReadingText).filter(
            ReadingText.lang == lang,
            ReadingText.generated_for_account_id == account_id,
            ReadingText.content.is_(None)
        ).order_by(ReadingText.created_at.asc()).first()
    
    def count_unopened_ready(
        self,
        global_db: Session,
        lang: str,
        target_lang: str,
    ) -> int:
        """Count ready texts in the global pool for a language pair."""
        return global_db.query(ReadingText).filter(
            ReadingText.lang == lang,
            ReadingText.target_lang == target_lang,
            ReadingText.content.is_not(None),
            ReadingText.content != "",
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
        ).count()
    
    def mark_text_read(
        self,
        account_db: Session,
        profile_id: int,
        text_id: int,
    ) -> bool:
        """Mark a text as read by a profile."""
        now = datetime.now(timezone.utc)
        
        existing = account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile_id,
            ProfileTextRead.text_id == text_id
        ).first()
        
        if existing:
            existing.read_count += 1
            existing.last_read_at = now
        else:
            account_db.add(ProfileTextRead(
                profile_id=profile_id,
                text_id=text_id,
                read_count=1,
                first_read_at=now,
                last_read_at=now,
            ))
        
        account_db.commit()
        return True
    
    def clear_current_pointer(
        self,
        account_db: Session,
        account_id: int,
    ) -> bool:
        """Clear the profile's current_text_id pointer."""
        from ..models import Profile
        
        prof = account_db.query(Profile).filter(Profile.account_id == account_id).first()
        if prof and prof.current_text_id is not None:
            prof.current_text_id = None
            account_db.commit()
            return True
        return False
    
    def set_current_pointer(
        self,
        account_db: Session,
        account_id: int,
        text_id: int,
    ) -> bool:
        """Set the profile's current_text_id pointer."""
        from ..models import Profile
        
        prof = account_db.query(Profile).filter(Profile.account_id == account_id).first()
        if prof:
            prof.current_text_id = text_id
            account_db.commit()
            return True
        return False
