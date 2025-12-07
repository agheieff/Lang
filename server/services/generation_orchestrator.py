"""
Generation Orchestrator Service

Consolidated service that handles:
- Text content generation via LLM
- Word and sentence translation generation
- Orchestration between generation and translation
- High-level text lifecycle management

Merges functionality from:
- GenerationService
- TranslationService
- UnifiedOrchestrationService
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session
from ..llm import build_reading_prompt
from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from ..llm.prompts import build_translation_contexts, build_word_translation_prompt
from ..services.model_registry_service import get_model_registry
from ..services.llm_logging import log_llm_request, llm_call_and_log_to_file
from ..services.openrouter_key_service import get_openrouter_key_service
from ..services.title_extraction_service import TitleExtractionService
from ..services.notification_service import get_notification_service
from ..services.state_manager import GenerationStateManager
from ..services.pool_selection_service import get_pool_selection_service
from ..services.usage_service import get_usage_service
from ..services.level_service import get_ci_target
from ..llm import compose_level_hint
from ..db import db_manager, GlobalSessionLocal
from ..utils.json_parser import (
    extract_json_from_text,
    extract_text_from_llm_response,
    extract_structured_translation,
    extract_word_translations,
)
from ..utils.text_segmentation import split_sentences
from ..utils.gloss import compute_spans
from ..models import (
    Profile,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    TextUnit,
)

logger = logging.getLogger(__name__)


class TextGenerationResult:
    """Result of text generation operation."""
    def __init__(self, success: bool, content: Optional[str] = None, title: Optional[str] = None, error: Optional[str] = None, messages=None):
        self.success = success
        self.content = content
        self.title = title
        self.error = error
        self.messages = messages or []


class TranslationResult:
    """Result of translation operation."""
    def __init__(self, success: bool, word_count: int = 0, sentence_count: int = 0, error: Optional[str] = None):
        self.success = success
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.error = error


class GenerationOrchestrator:
    """
    Consolidated service that orchestrates text generation and translation.
    
    Handles:
    - Text content generation via LLM
    - Word and sentence translation generation
    - Orchestration and lifecycle management
    - Legacy compatibility for old service interfaces
    """
    
    _instance: Optional['GenerationOrchestrator'] = None
    _instance_lock = threading.Lock()
    
    def __new__(cls):
        # Singleton pattern to share state across requests
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        self.state_manager = GenerationStateManager()
        self.pool_service = get_pool_selection_service()
        self.notification_service = get_notification_service()
        self.title_service = TitleExtractionService()
        # Add any other service instances we need
        self._initialized = True
    
    # TEXT GENERATION METHODS
    # -----------------------------------------------------------------
    
    def ensure_text_available(self, global_db: Session, account_id: int, lang: str) -> None:
        """
        Ensure the text pool has available texts. Triggers generation if needed.
        """
        # Check if there are any ready texts in the pool
        ready_text = global_db.query(ReadingText).filter(
            ReadingText.lang == lang,
            ReadingText.content.isnot(None),
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
        ).first()
        
        if not ready_text:
            # Trigger generation of a new text - profile comes from global DB
            from ..models import Profile
            profile = global_db.query(Profile).filter(
                Profile.account_id == account_id,
                Profile.lang == lang
            ).first()

            if profile:
                # Start generation - run the async job
                try:
                    asyncio.run(
                        self._run_generation_job(
                            global_db=global_db,
                            account_id=account_id,
                            lang=lang,
                            profile=profile
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to run generation job: {e}")
    
    async def _run_generation_job(self, global_db: Session, account_id: int, lang: str, profile: Profile) -> None:
        """
        Run the complete text generation and translation job.
        """
        try:
            # Step 1: Generate text content
            generation_result = await self._generate_text_content(
                global_db=global_db,
                account_id=account_id,
                lang=lang,
                profile=profile
            )
            
            if not generation_result.success:
                self.notification_service.send_generation_failed(account_id, lang, None, generation_result.error)
                return
            
            text_id = generation_result.messages.get('text_id') if generation_result.messages else None
            if not text_id:
                logger.error("No text_id returned from generation")
                return
            
            # Step 2: Generate translations
            translation_result = await self.generate_translations(
                text_id=text_id,
                account_id=account_id
            )
            
            if translation_result.success:
                self.notification_service.send_translations_ready(account_id, lang, text_id)
            else:
                logger.error(f"Translation failed for text_id={text_id}: {translation_result.error}")
                
        except Exception as e:
            logger.error(f"Generation job failed: {e}")
            self.notification_service.send_generation_failed(account_id, lang, None, str(e))
    
    async def _generate_text_content(
        self,
        global_db: Session,
        account_id: int,
        lang: str,
        profile: Profile
    ) -> TextGenerationResult:
        """Generate text content via LLM."""
        try:
            # Check quota before generation
            from ..auth import Account
            from .usage_service import get_usage_service
            
            account = global_db.get(Account, account_id)
            tier = account.subscription_tier if account else "Free"
            
            usage_service = get_usage_service()
            can_generate, reason = usage_service.check_quota(global_db, account_id, tier)
            
            if not can_generate:
                logger.warning(f"Quota exceeded for account {account_id}: {reason}")
                return TextGenerationResult(success=False, error=f"Quota exceeded: {reason}")
            
            # Create placeholder text
            placeholder_text = ReadingText(
                lang=lang,
                target_lang=profile.target_lang,
                generated_for_account_id=account_id,
                created_at=datetime.now(timezone.utc)
            )
            global_db.add(placeholder_text)
            global_db.flush()
            
            text_id = placeholder_text.id
            
            # Notify generation started
            self.notification_service.send_generation_started(account_id, lang, text_id)
            
            # Build prompt
            from ..llm.prompts import build_reading_prompt_spec
            spec, messages, _ = build_reading_prompt_spec(
                db=global_db,
                account_id=account_id,
                lang=lang,
            )
            
            # Get generation parameters
            ci_target, topic = self.pool_service.get_generation_params(profile, vary=True)
            
            # Make LLM call - get appropriate API key based on user tier
            import os
            from ..auth import Account
            
            # Get account to check tier and get appropriate key
            account = global_db.get(Account, account_id)
            api_key = None
            
            if account:
                key_service = get_openrouter_key_service()
                user_key = key_service.get_user_key(account)
                
                if user_key:
                    # Paid tier user with their own key
                    api_key = user_key
                    logger.info(f"Using user's own OpenRouter key for account {account_id}")
                else:
                    # Free tier or no individual key - use shared key
                    api_key = os.getenv("OPENROUTER_API_KEY")
                    if api_key:
                        logger.info(f"Using shared OpenRouter key for account {account_id}")
            
            if not api_key:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("No OpenRouter API key available")
            
            model = _pick_openrouter_model(None) 
            logger.info(f"Now using key ending in {api_key[-8:]} for text generation")
            result, stats = await chat_complete_with_raw(
                messages=messages,
                model=model,
                user_api_key=api_key,
                temperature=ci_target,
                max_tokens=4096
            )
            
            # Extract content
            content = extract_text_from_llm_response(result)
            if content:
                # Extract title
                title = self.title_service.extract_title(content)
                
                # Update text
                placeholder_text.content = content
                placeholder_text.title = title
                placeholder_text.content_length = len(content)
                placeholder_text.ci_preference = ci_target
                placeholder_text.topic = topic
                
                global_db.commit()
                
                # Record usage for quota tracking
                usage_service.record_usage(global_db, account_id, len(content))
                
                # Log LLM request (stats is a dict from the API response)
                if stats:
                    log_llm_request(
                        endpoint="/llm/generate_text",
                        request_id=str(text_id),
                        account_id=account_id,
                        input_tokens_before=stats.get("usage", {}).get("prompt_tokens"),
                        output_tokens_before=stats.get("usage", {}).get("completion_tokens"),
                        input_tokens_after=None,
                        output_tokens_after=None,
                        input_cost_before=None,
                        output_cost_before=None,
                        input_cost_after=None,
                        output_cost_after=None,
                        model=model,
                        temperature=ci_target,
                        max_tokens=4096
                    )
                
                logger.info(f"Generated text {text_id}: {len(content)} chars")
                return TextGenerationResult(
                    success=True,
                    content=content,
                    title=title,
                    messages={'text_id': text_id}
                )
            else:
                return TextGenerationResult(
                    success=False,
                    error="No content extracted from LLM response"
                )
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return TextGenerationResult(success=False, error=str(e))
    
    # TRANSLATION METHODS
    # -----------------------------------------------------------------
    
    async def generate_translations(self, text_id: int, account_id: int) -> TranslationResult:
        """
        Generate word and sentence translations for a text.
        """
        try:
            async with asyncio.timeout(300):  # 5 minute timeout
                with db_manager.transaction() as global_db:
                    text = global_db.get(ReadingText, text_id)
                    if not text:
                        return TranslationResult(success=False, error="Text not found")
                    
                    if not text.content:
                        return TranslationResult(success=False, error="Text has no content")
                    
                    # Start both translation tasks in parallel
                    word_task = asyncio.create_task(
                        self._generate_word_translations(global_db, text, account_id)
                    )
                    sentence_task = asyncio.create_task(
                        self._generate_sentence_translations(global_db, text, account_id)
                    )
                    
                    # Wait for both to complete
                    word_result, sentence_result = await asyncio.gather(
                        word_task, sentence_task, return_exceptions=True
                    )
                    
                    word_count = 0
                    sentence_count = 0
                    error = None
                    
                    # Handle word translation result
                    if isinstance(word_result, Exception):
                        error = f"Word translation error: {word_result}"
                        logger.error(error)
                    elif not word_result.success:
                        error = f"Word translation failed: {word_result.error}"
                        logger.error(error)
                    else:
                        word_count = word_result.word_count or 0
                        text.words_complete = True
                    
                    # Handle sentence translation result
                    if isinstance(sentence_result, Exception):
                        if not error:
                            error = f"Sentence translation error: {sentence_result}"
                            logger.error(error)
                    elif not sentence_result.success:
                        if not error:
                            error = f"Sentence translation failed: {sentence_result.error}"
                            logger.error(error)
                    else:
                        sentence_count = sentence_result.sentence_count or 0
                        text.sentences_complete = True
                    
                    # Update completion status
                    text.translation_attempts += 1
                    text.last_translation_attempt = datetime.now(timezone.utc)
                    
                    global_db.commit()
                    
                    return TranslationResult(
                        success=error is None,
                        word_count=word_count,
                        sentence_count=sentence_count,
                        error=error
                    )
                    
        except asyncio.TimeoutError:
            logger.error(f"Translation timeout for text {text_id}")
            return TranslationResult(success=False, error="Translation timeout")
        except Exception as e:
            logger.error(f"Translation failed for text {text_id}: {e}")
            return TranslationResult(success=False, error=str(e))
    
    async def _generate_word_translations(self, global_db: Session, text: ReadingText, account_id: int) -> TranslationResult:
        """Generate word translations for a text."""
        try:
            # Compute word spans
            spans = compute_spans(text.content, text.lang)
            
            if not spans:
                return TranslationResult(success=True, word_count=0, sentence_count=0)
            
            # Extract surfaces for translation
            surfaces = [span.surface for span in spans]
            
            # Build translation prompt
            translation_contexts = build_translation_contexts(text.lang, text.content or "", surfaces)
            model = _pick_openrouter_model(account_id=account_id)
            
            # Make LLM call
            key_service = get_openrouter_key_service()
            api_key = await key_service.get_valid_key()
            messages = build_word_translation_prompt(translation_contexts)
            result, stats = await chat_complete_with_raw(
                messages=messages, model=model, api_key=api_key, temperature=0.3, max_tokens=8000
            )
            
            # Extract translations
            translation_data = extract_json_from_text(result)
            word_translations = extract_word_translations(translation_data)
            
            if not word_translations:
                return TranslationResult(success=False, error="Could not extract word translations")
            
            # Store translations
            word_count = 0
            for surface in surfaces:
                if surface in word_translations:
                    for translation_dict in word_translations[surface]:
                        # Find corresponding spans
                        matching_spans = [span for span in spans if span.surface == surface]
                        for span in matching_spans:
                            word_gloss = ReadingWordGloss(
                                text_id=text.id,
                                target_lang=text.target_lang,
                                lang=text.lang,
                                span_start=span.start,
                                span_end=span.end,
                                lemma=translation_dict.get('lemma', surface),
                                pos=translation_dict.get('pos', 'unknown'),
                                pinyin=translation_dict.get('pinyin', ''),
                                translation=translation_dict.get('translation', ''),
                                lemma_translation=translation_dict.get('lemma_translation', ''),
                                grammar=translation_dict.get('grammar', ''),
                            )
                            global_db.add(word_gloss)
                            word_count += 1
            
            global_db.flush()
            return TranslationResult(success=True, word_count=word_count)
            
        except Exception as e:
            logger.error(f"Word translation failed: {e}")
            return TranslationResult(success=False, error=str(e))
    
    async def _generate_sentence_translations(self, global_db: Session, text: ReadingText, account_id: int) -> TranslationResult:
        """Generate sentence translations for a text."""
        try:
            # Split text into sentences
            sentences = split_sentences(text.content, text.lang)
            
            if not sentences:
                return TranslationResult(success=True, word_count=0, sentence_count=0)
            
            sentence_count = 0
            for i, sentence in enumerate(sentences):
                try:
                    # Build translation context for this sentence
                    contexts = build_translation_contexts(text.lang, sentence, [])
                    
                    # Simple translation: prompt LLM to translate the sentence
                    messages = [
                        {"role": "system", "content": "You are a helpful translator. Translate the given text to the target language, preserving meaning and tone."},
                        {"role": "user", "content": f"Translate this {text.lang} sentence to {text.target_lang}:\n\n{sentence}\n\nTranslation:"}
                    ]
                    
                    # Make LLM call
                    model = _pick_openrouter_model(account_id=account_id)
                    key_service = get_openrouter_key_service()
                    api_key = await key_service.get_valid_key()
                    result, _ = await chat_complete_with_raw(
                        messages=messages, model=model, api_key=api_key, temperature=0.3, max_tokens=2000
                    )
                    
                    # Extract translation (simple - just take the result as is)
                    translation = result.strip()
                    
                    if translation:
                        # Store translation
                        text_translation = ReadingTextTranslation(
                            text_id=text.id,
                            target_lang=text.target_lang,
                            lang=text.lang,
                            unit="sentence",
                            segment_index=i,
                            source_text=sentence,
                            translated_text=translation,
                        )
                        global_db.add(text_translation)
                        sentence_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to translate sentence {i}: {e}")
                    continue
            
            global_db.flush()
            return TranslationResult(success=True, sentence_count=sentence_count)
            
        except Exception as e:
            logger.error(f"Sentence translation failed: {e}")
            return TranslationResult(success=False, error=str(e))
    
    # ORCHESTRATION METHODS
    # -----------------------------------------------------------------
    
    def retry_incomplete_texts(self, global_db: Session, account_id: int, lang: str) -> int:
        """Retry failed translations for incomplete texts."""
        try:
            incomplete_texts = global_db.query(ReadingText).filter(
                ReadingText.lang == lang,
                ~ReadingText.words_complete | ~ReadingText.sentences_complete
            ).all()
            
            retried_count = 0
            for text in incomplete_texts:
                # Reset completion flags
                text.words_complete = False
                text.sentences_complete = False
                text.translation_attempts += 1
                text.last_translation_attempt = datetime.now(timezone.utc)
                
                # Start translation retry
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                loop.create_task(self.generate_translations(text.id, text.generated_for_account_id))
                retried_count += 1
            
            global_db.commit()
            return retried_count
            
        except Exception as e:
            logger.error(f"Error retrying incomplete texts: {e}")
            return 0
    
    def is_text_ready(self, text_id: int) -> bool:
        """Check if a text is ready for reading."""
        global_db = GlobalSessionLocal()
        try:
            text = global_db.get(ReadingText, text_id)
            return text.is_ready if text else False
        finally:
            global_db.close()
    
    def get_text(self, text_id: int) -> Optional[ReadingText]:
        """Get a text by ID."""
        global_db = GlobalSessionLocal()
        try:
            return global_db.get(ReadingText, text_id)
        finally:
            global_db.close()
    
    # LEGACY COMPATIBILITY
    # -----------------------------------------------------------------
    
    def is_generation_in_progress(self, account_id: int, lang: str) -> bool:
        """Check if generation is in progress for an account."""
        return self.state_manager.is_generating(account_id, lang)
    
    # Legacy method compatibility
    async def generate_translations_legacy(self, account_db, global_db, text_id: int):
        """Legacy method for backwards compatibility."""
        from ..deps import get_current_account
        # Extract account_id from account_db somehow
        account = account_db.query(Profile).filter(Profile.id == 1).first()  # Simplified
        return await self.generate_translations(text_id, account.account_id if account else 1)


# Singleton instance for now
_orchestrator = None

def get_generation_orchestrator() -> GenerationOrchestrator:
    """Get the singleton GenerationOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = GenerationOrchestrator()
    return _orchestrator

# Individual service getters for backward compatibility
def get_generation_service() -> GenerationOrchestrator:
    """Backward compatibility - returns the orchestrator with generation methods."""
    return get_generation_orchestrator()

def get_translation_service() -> GenerationOrchestrator:
    """Backward compatibility - returns the orchestrator with translation methods.""" 
    return get_generation_orchestrator()
