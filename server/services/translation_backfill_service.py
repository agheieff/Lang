"""
Service for backfilling missing translations in existing texts.

This service identifies and fills gaps in:
- Sentence translations
- Word translations
- Title translations
"""

from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from sqlalchemy.orm import Session
from datetime import datetime

from ..models import ReadingText, ReadingTextTranslation, ReadingWordGloss, Profile
from .translation_service import TranslationService, TranslationResult
from .title_extraction_service import TitleExtractionService
from ..db import GlobalSessionLocal
from ..auth import Account
from ..llm.prompts import build_translation_contexts
from ..utils.text_segmentation import split_sentences


class TranslationBackfillService:
    """
    Service for identifying and backfilling missing translations.
    
    This service can:
    - Detect missing translation units for a text
    - Regenerate missing translations
    - Provide status on translation completeness
    """
    
    def __init__(self):
        self.translation_service = TranslationService()
        self.title_service = TitleExtractionService()
    
    def check_translation_completeness(self, account_db: Session, account_id: int, text_id: int) -> Dict[str, bool]:
        """
        Check which translation components are present for a text.
        
        Returns:
            Dict with keys: 'words', 'sentences', 'title'
        """
        result = {"words": False, "sentences": False, "title": False}
        
        # Check if text exists
        text = account_db.query(ReadingText).filter(
            ReadingText.id == text_id,
            ReadingText.account_id == account_id
        ).first()
        
        if not text:
            return result
        
        # Check sentence translations
        sentence_translations = account_db.query(ReadingTextTranslation).filter(
            ReadingTextTranslation.account_id == account_id,
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.unit == "sentence"
        ).count()
        
        # Count expected sentences
        if text.content:
            try:
                sentence_spans = split_sentences(text.content, text.lang)
                result["sentences"] = sentence_translations >= len(sentence_spans)
            except Exception:
                result["sentences"] = sentence_translations > 0
        else:
            result["sentences"] = sentence_translations > 0
        
        # Check word translations
        word_translations = account_db.query(ReadingWordGloss).filter(
            ReadingWordGloss.account_id == account_id,
            ReadingWordGloss.text_id == text_id
        ).count()
        
        # We consider words complete if we have at least some words
        result["words"] = word_translations > 0
        
        # Check title translations
        title_translations = account_db.query(ReadingTextTranslation).filter(
            ReadingTextTranslation.account_id == account_id,
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.unit == "text"
        ).count()
        
        result["title"] = title_translations > 0
        
        return result
    
    def backfill_missing_translations(self, account_id: int, text_id: int) -> Dict[str, bool]:
        """
        Backfill all missing translations for a text.
        
        Returns:
            Dict with keys: 'words', 'sentences', 'title' indicating success
        """
        results = {"words": False, "sentences": False, "title": False}
        
        # Get database sessions
        from ..utils.session_manager import db_manager
        
        try:
            with db_manager.transaction(account_id) as account_db:
                global_db = GlobalSessionLocal()
                try:
                    # Check what's missing
                    completeness = self.check_translation_completeness(account_db, account_id, text_id)
                    
                    # Get text details
                    text = account_db.query(ReadingText).filter(
                        ReadingText.id == text_id,
                        ReadingText.account_id == account_id
                    ).first()
                    
                    if not text:
                        print(f"[BACKFILL] Text {text_id} not found for account {account_id}")
                        return results
                    
                    # Get profile for target language
                    prof = account_db.query(Profile).filter(
                        Profile.account_id == account_id,
                        Profile.lang == text.lang
                    ).first()
                    
                    target_lang = prof.target_lang if prof else "en"
                    
                    # Backfill title translation if missing
                    if not completeness["title"]:
                        title, title_translation = self.title_service.get_title(account_db, account_id, text_id)
                        if title and title_translation:
                            success = self.title_service.persist_title_translation(
                                account_db, account_id, text_id, title, title_translation, target_lang
                            )
                            results["title"] = success
                    
                    # For words and sentences, we need to regenerate them completely
                    # This is more complex and would involve calling the translation service again
                    
                    # TODO: Implement full word and sentence backfill
                    # This would involve:
                    # 1. Getting the original reading messages from LLMRequestLog
                    # 2. Calling translation_service.generate_translations with just the missing components
                    # 3. Handling partial success scenarios
                    
                finally:
                    global_db.close()
        
        except Exception as e:
            print(f"[BACKFILL] Error backfilling translations for text {text_id}: {e}")
        
        return results
    
    def backfill_title_from_existing_log(self, account_db: Session, account_id: int, text_id: int) -> bool:
        """
        Backfill title translation from existing LLM logs if available.
        
        Returns True if title translation was added
        """
        try:
            # Get text details
            text = account_db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if not text:
                print(f"[BACKFILL] Text {text_id} not found for account {account_id}")
                return False
            
            # Get profile for target language
            prof = account_db.query(Profile).filter(
                Profile.account_id == account_id,
                Profile.lang == text.lang
            ).first()
            
            target_lang = prof.target_lang if prof else "en"
            
            # Extract title from existing logs
            title, title_translation = self.title_service.get_title(account_db, account_id, text_id)
            
            # Check if title translation already exists
            existing = account_db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.account_id == account_id,
                ReadingTextTranslation.text_id == text_id,
                ReadingTextTranslation.unit == "text"
            ).first()
            
            if existing:
                print(f"[BACKFILL] Title translation already exists for text {text_id}")
                return True
            
            # If no title translation, generate it
            if title:
                try:
                    import tempfile
                    from pathlib import Path
                    
                    # Generate title translation
                    success = self.translation_service._generate_title_translation(
                        account_db, account_id, text_id, text.lang, target_lang,
                        title, Path(tempfile.mkdtemp()), "openrouter", None, None
                    )
                    
                    if success:
                        print(f"[BACKFILL] Generated and saved title translation for text {text_id}")
                        return True
                    else:
                        print(f"[BACKFILL] Failed to generate title translation for text {text_id}")
                except Exception as e:
                    print(f"[BACKFILL] Error generating title translation: {e}")
            
            return False
            
        except Exception as e:
            print(f"[BACKFILL] Error backfilling title for text {text_id}: {e}")
            return False
