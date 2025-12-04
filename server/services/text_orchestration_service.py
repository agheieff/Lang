"""
Text Orchestration Service - Consolidated text and translation management.

This service consolidates:
- TextGenerationService: Text content generation 
- TranslationService: Word and sentence translations
- TranslationBackfillService: Missing translation detection and filling
- TranslationValidationService: Translation completeness validation

Responsibilities:
- Generate text content via LLM
- Generate word and sentence translations
- Validate and backfill missing translations
- Manage the complete text lifecycle from generation to readiness
- Log all operations for debugging and monitoring
"""

import json
import os
import random
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from fastapi import HTTPException

from ..llm import build_reading_prompt
from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from ..llm.prompts import build_translation_contexts, build_word_translation_prompt
from ..services.model_registry_service import get_model_registry
from ..services.llm_logging import log_llm_request, llm_call_and_log_to_file
from ..services.openrouter_key_service import get_openrouter_key_service
from ..services.title_extraction_service import TitleExtractionService
from ..auth import Account
from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from ..utils.json_parser import (
    extract_text_from_llm_response, 
    extract_json_from_text,
    extract_structured_translation,
    extract_word_translations,
)
from ..utils.text_segmentation import split_sentences
from ..utils.text_segmentation import split_sentences
from ..utils.gloss import compute_spans
from ..services.level_service import get_ci_target
from ..llm import compose_level_hint
from ..models import Lexeme, LexemeVariant
from sqlalchemy import select

logger = logging.getLogger(__name__)


# Word selection utility functions (integrated from word_selection.py)
class U:
    pass


def pick_words(account_db: Session, global_db: Session, user, lang: str, count: int = 12, new_ratio: float = 0.1) -> List[str]:
    """Select words for reading prompt, balancing new and known words."""
    if not account_db:
        return []
    
    try:
        # Get user profile
        profile = global_db.query(Profile).filter(Profile.account_id == user.id, Profile.lang == lang).first()
        if not profile:
            return []
        
        # Level-based word selection logic
        level_value = profile.level_value or 1.0
        level_var = profile.level_var or 1.0
        
        # Get appropriate word count based on level
        actual_count = min(count, max(3, int(count * (1 + level_value / 6))))
        
        # Simple word selection - in production this would be more sophisticated
        selected_words = []
        # Placeholder for actual word selection logic
        # This would typically query Lexeme table and apply level-based filtering
        
        return selected_words[:actual_count]
    except Exception as e:
        logger.error(f"Error selecting words: {e}")
        return []


# LLM common functions (integrated from llm_common.py)
def build_reading_prompt_spec(
    global_db: Session,
    *,
    account_id: int,
    lang: str,
    account_db: Optional[Session] = None,
    length: Optional[int] = None,
    include_words: Optional[List[str]] = None,
    ci_target_override: Optional[float] = None,
    topic: Optional[str] = None,
) -> Tuple:
    """Assemble PromptSpec and supporting values (words, level_hint)."""
    script = None
    if lang.startswith("zh"):
        prof = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        if prof and getattr(prof, "preferred_script", None) in ("Hans", "Hant"):
            script = prof.preferred_script
        else:
            script = "Hans"

    unit = "chars" if lang.startswith("zh") else "words"
    prof = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
    prof_len = None
    try:
        if prof and isinstance(prof.text_length, int) and prof.text_length and prof.text_length > 0:
            prof_len = int(prof.text_length)
    except Exception:
        prof_len = None
    approx_len = length if length is not None else (prof_len if prof_len is not None else (300 if unit == "chars" else 180))
    try:
        approx_len = max(50, min(2000, int(approx_len)))
    except Exception:
        approx_len = 300 if unit == "chars" else 180

    u = U(); u.id = account_id
    ci_target = ci_target_override if ci_target_override is not None else get_ci_target(global_db, account_id, lang)
    base_new_ratio = max(0.02, min(0.6, 1.0 - ci_target + 0.05))
    
    if account_db is not None:
        words = include_words or pick_words(account_db, global_db, u, lang, count=12, new_ratio=base_new_ratio)
        level_hint = compose_level_hint(global_db, u, lang)
    else:
        words = include_words or []
        level_hint = None

    from ..llm.prompts import PromptSpec
    spec = PromptSpec(
        lang=lang,
        unit=unit,
        approx_len=approx_len,
        user_level_hint=level_hint,
        include_words=words,
        script=script,
        ci_target=ci_target,
        recent_titles=[],  # Simplified for now
        topic=topic,
    )
    return spec, words, level_hint


def _get_recent_read_titles(global_db: Session, account_db: Session, account_id: int, lang: str, limit: int = 5) -> List[str]:
    """Fetch titles of the last N read texts for this user/language."""
    try:
        # Get recently read text IDs from account database
        from ..models import ProfileTextRead, ReadingText, Profile
        
        # First, get the user's profile for this language
        profile = global_db.query(Profile).filter(
            Profile.account_id == account_id,
            Profile.lang == lang
        ).first()
        
        if not profile:
            return []
        
        # Get recently read text IDs for this profile
        recent_reads = account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile.id,
            ProfileTextRead.read_at.isnot(None)
        ).order_by(ProfileTextRead.read_at.desc()).limit(limit).all()
        
        if not recent_reads:
            return []
        
        # Extract text IDs from recent reads
        text_ids = [read.text_id for read in recent_reads]
        
        # Fetch titles from global database for these text IDs
        texts_with_titles = global_db.query(ReadingText).filter(
            ReadingText.id.in_(text_ids),
            ReadingText.title.isnot(None),
            ReadingText.title != ""
        ).all()
        
        # Create a mapping of text_id to title for ordering
        title_map = {text.id: text.title for text in texts_with_titles}
        
        # Return titles in the same order as the recent reads (most recent first)
        titles = []
        for read in recent_reads:
            if read.text_id in title_map:
                titles.append(title_map[read.text_id])
        
        return titles
        
    except Exception as e:
        # Log error but don't fail the entire generation process
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to get recent read titles for account_id={account_id}, lang={lang}: {e}")
        return []


class TextGenerationResult:
    """Result of text generation attempt."""
    def __init__(self, 
                 success: bool,
                 text: str,
                 title: Optional[str] = None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 response_dict: Optional[Dict] = None,
                 error: Optional[str] = None,
                 log_dir: Optional[Path] = None,
                 messages: Optional[List[Dict]] = None):
        self.success = success
        self.text = text
        self.title = title
        self.provider = provider
        self.model = model
        self.response_dict = response_dict or {}
        self.error = error
        self.log_dir = log_dir
        self.messages = messages


class TranslationResult:
    """Result of translation generation attempt."""
    def __init__(self, 
                 success: bool,
                 words: bool = False,
                 sentences: bool = False,
                 error: Optional[str] = None,
                 log_dir: Optional[Path] = None):
        self.success = success
        self.words = words
        self.sentences = sentences
        self.error = error
        self.log_dir = log_dir


class TextOrchestrationService:
    """
    Consolidated service for text generation and translation.
    
    This service manages the complete text lifecycle:
    1. Text content generation
    2. Word and sentence translation generation
    3. Translation validation and backfill
    4. Logging and error handling
    """
    
    _instance: Optional['TextOrchestrationService'] = None
    _instance_lock = threading.Lock()
    
    def __new__(cls):
        # Singleton pattern to share in-memory state across requests
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._running: set[Tuple[int, str]] = set()  # (account_id, lang) pairs
        self._running_lock = threading.Lock()
        self.max_workers = 10  # Default parallelism for word translations
        self.title_service = TitleExtractionService()
        self._initialized = True
    
    def create_placeholder_text(
        self, 
        global_db: Session, 
        account_id: int, 
        lang: str,
        target_lang: str = "en",
        ci_target: Optional[float] = None,
        topic: Optional[str] = None,
    ) -> Optional[int]:
        """Create a placeholder ReadingText record in the GLOBAL database."""
        try:
            text_record = ReadingText(
                generated_for_account_id=account_id,
                lang=lang,
                target_lang=target_lang,
                source="llm",
                ci_target=ci_target,
                topic=topic,
                words_complete=False,
                sentences_complete=False,
            )
            global_db.add(text_record)
            global_db.commit()
            global_db.refresh(text_record)
            return text_record.id
        except Exception as e:
            logger.error(f"Failed to create placeholder text: {e}")
            global_db.rollback()
            return None
    
    def generate_text_content(
        self,
        global_db: Session,
        account_id: int,
        lang: str,
        text_id: int,
        account_db: Optional[Session] = None,
        length: Optional[int] = None,
        include_words: Optional[List[str]] = None,
        ci_target_override: Optional[float] = None,
        topic: Optional[str] = None,
    ) -> TextGenerationResult:
        """Generate text content for a placeholder ReadingText record."""
        try:
            # Get profile to determine target language
            profile = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
            target_lang = profile.target_lang if profile else "en"
            
            # Build prompt spec (using integrated function)
            spec, words, level_hint = build_reading_prompt_spec(
                global_db,
                account_id=account_id,
                lang=lang,
                account_db=account_db,
                length=length,
                include_words=include_words,
                ci_target_override=ci_target_override,
                topic=topic,
            )
            
            # Get model info
            model_registry = get_model_registry()
            model_config = model_registry.resolve_model(account_id, "text_generation", lang)
            
            # Generate prompt from spec
            from ..llm import build_reading_prompt
            prompt_messages = build_reading_prompt(spec)
            
            # Create log directory
            job_dir = Path("logs/text_generation") / f"text_{text_id}" / datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup key service
            key_service = get_openrouter_key_service()
            
            # Make LLM call
            response_dict = chat_complete_with_raw(
                api_base=model_config.base_url,
                api_key=key_service.get_active_key(),
                model=model_config.model_id,
                messages=prompt_messages,
                temperature=0.8,
                max_tokens=None,
                timeout=120,
            )
            
            # Log the request
            log_llm_request(
                job_dir / "text.json",
                request={
                    "account_id": account_id,
                    "lang": lang,
                    "target_lang": target_lang,
                    "text_id": text_id,
                    "messages": prompt_messages,
                    "model": model_config.model_id,
                    "provider": model_config.provider,
                },
                response=response_dict,
            )
            
            # Extract content
            content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Parse the generated content
            generated_text = extract_text_from_llm_response(content)
            generated_title = extract_json_from_text(content).get("title")
            
            # Update the database record
            text_record = global_db.query(ReadingText).filter(ReadingText.id == text_id).first()
            if text_record:
                text_record.content = generated_text
                text_record.title = generated_title
                text_record.request_sent_at = datetime.utcnow()
                text_record.generated_at = datetime.utcnow()
                
                # Update pool selection fields
                if ci_target_override is not None:
                    text_record.ci_target = ci_target_override
                if topic:
                    text_record.topic = topic
                
                global_db.commit()
            
            return TextGenerationResult(
                success=True,
                text=generated_text,
                title=generated_title,
                provider=model_config.provider,
                model=model_config.model_id,
                response_dict=response_dict,
                log_dir=job_dir,
                messages=prompt_messages,
            )
            
        except Exception as e:
            logger.error(f"Text generation failed for text_id={text_id}: {e}")
            return TextGenerationResult(
                success=False,
                text="",
                error=str(e),
            )
    
    def generate_translations(
        self,
        account_db: Session,
        global_db: Session,
        account_id: int,
        lang: str,
        text_id: int,
        text_content: str,
        text_title: Optional[str],
        job_dir: Path,
        reading_messages: List[Dict],
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> TranslationResult:
        """Generate all translations for a text."""
        try:
            # Get profile for target language
            profile = global_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
            target_lang = profile.target_lang if profile else "en"
            
            # Get model config if not provided
            if not provider or not model_id or not base_url:
                model_registry = get_model_registry()
                model_config = model_registry.resolve_model(account_id, "translation", lang)
                provider = model_config.provider
                model_id = model_config.model_id
                base_url = model_config.base_url
            
            # Split text into sentences with logging
            sentences = split_sentences(text_content, lang)
            logger.info(f"Split text into {len(sentences)} sentences for text_id={text_id}")
            
            # Generate word translations (parallel)
            words_success = self._generate_word_translations_parallel(
                account_db, global_db, account_id, lang, target_lang, text_id,
                sentences, job_dir, provider, model_id, base_url, reading_messages
            )
            
            # Generate sentence translations
            sentences_success = self._generate_sentence_translations(
                account_db, global_db, account_id, lang, target_lang, text_id,
                sentences, job_dir, provider, model_id, base_url, reading_messages, text_title
            )
            
            # Update text completion flags
            text_record = global_db.query(ReadingText).filter(ReadingText.id == text_id).first()
            if text_record:
                text_record.words_complete = words_success
                text_record.sentences_complete = sentences_success
                global_db.commit()
            
            return TranslationResult(
                success=words_success and sentences_success,
                words=words_success,
                sentences=sentences_success,
                log_dir=job_dir,
            )
            
        except Exception as e:
            logger.error(f"Translation generation failed for text_id={text_id}: {e}")
            return TranslationResult(
                success=False,
                error=str(e),
                log_dir=job_dir,
            )
    
    def _generate_word_translations_parallel(
        self, account_db: Session, global_db: Session, account_id: int, lang: str,
        target_lang: str, text_id: int, sentences: List[str], job_dir: Path,
        provider: str, model_id: str, base_url: str, reading_messages: List[Dict]
    ) -> bool:
        """Generate word translations in parallel for multiple sentences."""
        try:
            key_service = get_openrouter_key_service()
            
            def process_sentence(sentence_idx: int, sentence: str) -> bool:
                try:
                    # Build context for this sentence
                    contexts = build_translation_contexts(
                        lang, reading_messages, sentence, []
                    )
                    
                    # Build word translation prompt
                    prompt_messages = build_word_translation_prompt(
                        lang, target_lang, contexts
                    )
                    
                    # Make LLM call
                    response_dict = chat_complete_with_raw(
                        api_base=base_url,
                        api_key=key_service.get_active_key(),
                        model=model_id,
                        messages=prompt_messages,
                        temperature=0.3,
                        max_tokens=2000,
                        timeout=60,
                    )
                    
                    # Log the request
                    log_llm_request(
                        job_dir / f"words_sentence_{sentence_idx}.json",
                        request={
                            "account_id": account_id,
                            "text_id": text_id,
                            "sentence_idx": sentence_idx,
                            "messages": prompt_messages,
                            "model": model_id,
                        },
                        response=response_dict,
                    )
                    
                    # Extract word translations
                    content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
                    word_translations = extract_word_translations(content)
                    
                    # Save to database
                    for word_trans in word_translations:
                        gloss = ReadingWordGloss(
                            text_id=text_id,
                            sentence_index=sentence_idx,
                            word=word_trans["word"],
                            gloss=word_trans["gloss"],
                            context=word_trans.get("context", ""),
                        )
                        account_db.add(gloss)
                    
                    account_db.commit()
                    return True
                    
                except Exception as e:
                    logger.error(f"Word translation failed for sentence {sentence_idx}: {e}")
                    return False
            
            # Process sentences in parallel
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sentences))) as executor:
                futures = [executor.submit(process_sentence, i, s) for i, s in enumerate(sentences)]
                results = [f.result() for f in futures]
            
            return all(results)
            
        except Exception as e:
            logger.error(f"Parallel word translation failed: {e}")
            return False
    
    def _generate_sentence_translations(
        self, account_db: Session, global_db: Session, account_id: int, lang: str,
        target_lang: str, text_id: int, sentences: List[str], job_dir: Path,
        provider: str, model_id: str, base_url: str, reading_messages: List[Dict],
        text_title: Optional[str]
    ) -> bool:
        """Generate structured sentence translations."""
        try:
            # Build translation contexts for all sentences
            contexts = build_translation_contexts(
                lang, reading_messages, " ".join(sentences), []
            )
            
            # Request structured translation
            request_payload = {
                "sentences": contexts,
                "output_format": "structured_json",
                "target_language": target_lang,
                "text_title": text_title,
            }
            
            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are a professional translator. Translate the given sentences and provide structured output.",
                },
                {
                    "role": "user", 
                    "content": json.dumps(request_payload, ensure_ascii=False, indent=2),
                }
            ]
            
            # Make LLM call
            key_service = get_openrouter_key_service()
            response_dict = chat_complete_with_raw(
                api_base=base_url,
                api_key=key_service.get_active_key(),
                model=model_id,
                messages=prompt_messages,
                temperature=0.2,
                max_tokens=4000,
                timeout=90,
            )
            
            # Log the request
            log_llm_request(
                job_dir / "sentences.json",
                request={
                    "account_id": account_id,
                    "text_id": text_id,
                    "messages": prompt_messages,
                    "model": model_id,
                },
                response=response_dict,
            )
            
            # Extract translations
            content = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "")
            translations = extract_structured_translation(content)
            
            # Save to database
            for i, translation in enumerate(translations):
                if i < len(sentences):
                    record = ReadingTextTranslation(
                        text_id=text_id,
                        sentence_index=i,
                        original=sentences[i],
                        translation=translation,
                    )
                    account_db.add(record)
            
            account_db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Sentence translation failed for text_id={text_id}: {e}")
            return False
    
    def check_translation_completeness(self, account_db: Session, account_id: int, text_id: int) -> Dict[str, bool]:
        """Check which translation components are present for a text."""
        result = {"words": False, "sentences": False, "title": False}
        
        try:
            # Check if text exists
            text = account_db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if not text:
                return result
            
            # Check sentence translations
            sentence_count = account_db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.text_id == text_id
            ).count()
            result["sentences"] = sentence_count > 0
            
            # Check word translations
            word_count = account_db.query(ReadingWordGloss).filter(
                ReadingWordGloss.text_id == text_id
            ).count()
            result["words"] = word_count > 0
            
            # Check title
            result["title"] = bool(text.title and text.title.strip())
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking translation completeness for text_id={text_id}: {e}")
            return result
    
    def validate_and_backfill(self, account_id: int, text_id: int) -> Dict[str, bool]:
        """Validate translation completeness and backfill if needed."""
        from ..utils.session_manager import db_manager
        
        try:
            with db_manager.transaction(account_id) as account_db:
                # Check completeness
                completeness = self.check_translation_completeness(account_db, account_id, text_id)
                
                # Backfill title if missing
                if not completeness["title"]:
                    title_success = self._backfill_title_from_existing_log(account_db, account_id, text_id)
                    if title_success:
                        completeness["title"] = True
                
                # For words and sentences, we would need the original generation context
                # This is a simplified implementation - full backfill would require access to logs
                logger.info(f"Translation completeness for text_id={text_id}: {completeness}")
                
                return completeness
                
        except Exception as e:
            logger.error(f"Error in validate_and_backfill for text_id={text_id}: {e}")
            return {"words": False, "sentences": False, "title": False}
    
    def _backfill_title_from_existing_log(self, account_db: Session, account_id: int, text_id: int) -> bool:
        """Attempt to extract and backfill title from existing generation logs."""
        try:
            # This is a simplified implementation
            # In practice, would check logs/ directories for generation logs
            text = account_db.query(ReadingText).filter(
                ReadingText.id == text_id,
                ReadingText.account_id == account_id
            ).first()
            
            if text and text.content:
                # Generate title from content
                generated_title = self.title_service.extract_title_from_content(text.content, "en")
                if generated_title:
                    text.title = generated_title
                    account_db.commit()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error backfilling title for text_id={text_id}: {e}")
            return False


# Global instance for backward compatibility
def get_text_orchestration_service() -> TextOrchestrationService:
    """Get the singleton TextOrchestrationService instance."""
    return TextOrchestrationService()
