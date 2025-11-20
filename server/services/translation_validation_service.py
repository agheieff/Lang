"""
Service for validating translation completeness after generation.

This service checks that required translation components are present
and triggers backfill or re-generation when needed.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime

from ..models import ReadingText, ReadingTextTranslation, ReadingWordGloss, Profile
from ..translation_backfill_service import TranslationBackfillService
from ..utils.session_manager import db_manager


class TranslationValidationService:
    """
    Service for validating translation completeness.
    
    This service:
    - Validates that all required translation components are present
    - Reports on translation completeness
    - Can trigger backfill for missing components
    """
    
    def __init__(self):
        self.backfill_service = TranslationBackfillService()
    
    def validate_and_backfill(self, account_id: int, text_id: int) -> Dict[str, bool]:
        """
        Validate translation completeness and backfill if needed.
        
        Returns:
            Dict with keys: 'words', 'sentences', 'title' indicating completion status
        """
        try:
            with db_manager.transaction(account_id) as account_db:
                # Check completeness
                completeness = self.backfill_service.check_translation_completeness(
                    account_db, account_id, text_id
                )
                
                # Backfill title if missing (easiest to fix)
                if not completeness["title"]:
                    title_success = self.backfill_service.backfill_title_from_existing_log(
                        account_db, account_id, text_id
                    )
                    if title_success:
                        completeness["title"] = True
                
                # TODO: Add more sophisticated validation for words and sentences
                # This could involve:
                # 1. Checking if the number of translations matches expected counts
                # 2. Verifying translation quality (non-empty, reasonable length)
                # 3. Ensuring all text segments are covered
                
                return completeness
                
        except Exception as e:
            print(f"[VALIDATION] Error validating translations for text {text_id}: {e}")
            return {"words": False, "sentences": False, "title": False}
    
    def get_translation_stats(self, account_db: Session, account_id: int, text_id: int) -> Dict[str, int]:
        """
        Get detailed translation statistics for a text.
        
        Returns:
            Dict with counts: 'sentences', 'words', 'title'
        """
        stats = {"sentences": 0, "words": 0, "title": 0}
        
        try:
            # Get text info
            text = account_db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if not text:
                return stats
            
            # Count sentence translations
            stats["sentences"] = account_db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.account_id == account_id,
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.unit == "sentence"
            ).count()
            
            # Count word translations
            stats["words"] = account_db.query(ReadingWordGloss).filter(
                ReadingWordGloss.account_id == account_id,
                ReadingWordGloss.text_id == text_id
            ).count()
            
            # Count title translations
            stats["title"] = account_db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.account_id == account_id,
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.unit == "text"
            ).count()
            
        except Exception as e:
            print(f"[VALIDATION] Error getting translation stats for text {text_id}: {e}")
        
        return stats
