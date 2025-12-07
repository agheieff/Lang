"""
Reading services consolidating generation, state, pool, and retry functionality.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Optional, Set, List, Tuple

from sqlalchemy.orm import Session
from server.llm import build_reading_prompt
from server.llm.client import _pick_openrouter_model, chat_complete_with_raw
from server.llm.prompts import build_translation_contexts, build_word_translation_prompt
from server.services.model_registry_service import get_model_registry
from server.services.llm_logging import log_llm_request, llm_call_and_log_to_file
from server.services.openrouter_key_service import get_openrouter_key_service
from server.services.title_extraction_service import TitleExtractionService
from server.services.notification_service import get_notification_service
from server.services.usage_service import get_usage_service
from server.services.level_service import get_ci_target
from server.llm import compose_level_hint
from server.db import db_manager, GlobalSessionLocal
from server.utils.json_parser import (
    extract_json_from_text,
    extract_text_from_llm_response,
    extract_structured_translation,
    extract_word_translations,
)
from server.utils.text_segmentation import split_sentences
from server.utils.gloss import compute_spans
from server.models import (
    Profile,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    TextVocabulary,
    ProfileTextRead,
    ProfileTextQueue,
    GenerationLog,
    TranslationLog,
    GenerationRetryAttempt,
    TextState,
)

logger = logging.getLogger(__name__)


# Text State Management
class TextState(str):
    """States a text can be in during generation."""
    NONE = "none"
    GENERATING = "generating"
    CONTENT_READY = "content_ready"
    FULLY_READY = "fully_ready"
    OPENED = "opened"
    READ = "read"
    FAILED = "failed"


class GenerationStateManager:
    """Manages the state of text generation."""
    
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
            ProfileTextRead.text_id == text_id,
        ).first()
        
        if read_entry:
            return TextState.READ
        
        # Check generation state
        if not rt.content or not rt.words_complete or not rt.sentences_complete:
            if rt.request_sent_at and not rt.generated_at:
                return TextState.GENERATING
            elif rt.content and not (rt.words_complete and rt.sentences_complete):
                return TextState.CONTENT_READY
            else:
                return TextState.FAILED
        
        return TextState.FULLY_READY
    
    def transition_to_content_ready(
        self,
        global_db: Session,
        text_id: int,
        content: str,
        words: List[Dict],
        level_hint: Optional[str] = None,
    ) -> ReadingText:
        """Transition text to content_ready state."""
        rt = global_db.get(ReadingText, text_id)
        if not rt:
            raise ValueError(f"Text {text_id} not found")
        
        rt.content = content
        rt.generated_at = datetime.now(timezone.utc)
        rt.prompt_words = words
        rt.prompt_level_hint = level_hint
        rt.words_complete = False
        rt.sentences_complete = False
        
        global_db.commit()
        global_db.refresh(rt)
        
        return rt
    
    def mark_fully_ready(
        self,
        global_db: Session,
        text_id: int,
    ) -> ReadingText:
        """Mark text as fully ready."""
        rt = global_db.get(ReadingText, text_id)
        if not rt:
            raise ValueError(f"Text {text_id} not found")
        
        rt.words_complete = True
        rt.sentences_complete = True
        
        global_db.commit()
        global_db.refresh(rt)
        
        return rt


# Reading Generation Service
class ReadingGenerationService:
    """Handles text and translation generation."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def generate_text_content(
        self,
        account_id: int,
        profile_id: int,
        lang: str,
        target_lang: str,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """Generate text content using LLM."""
        try:
            # Get configuration
            ci_target = get_ci_target(profile.level_value, profile.level_var)
            words = self._get_word_list_from_profile(profile)
            level_hint = compose_level_hint(profile.level_value, profile.level_code)
            
            # Build prompt
            prompt = build_reading_prompt(
                lang=lang,
                target_lang=target_lang,
                ci_target=ci_target,
                words=words,
                level_hint=level_hint,
                text_preferences=profile.text_preferences,
            )
            
            # Get LLM model and make request
            model = _pick_openrouter_model()
            
            logger.info(f"Generating text for account {account_id}, profile {profile_id}")
            
            response = await chat_complete_with_raw(
                messages=prompt,
                model=model,
            )
            
            # Extract content
            content = extract_text_from_llm_response(response)
            
            if not content:
                logger.error(f"Failed to extract content from LLM response: {response}")
                return None
            
            # Create ReadingText record
            with GlobalSessionLocal() as global_db:
                rt = ReadingText(
                    account_id=account_id,
                    generated_for_account_id=account_id,
                    lang=lang,
                    target_lang=target_lang,
                    content=content,
                    source="llm",
                    ci_target=ci_target,
                    request_sent_at=datetime.now(timezone.utc),
                    generated_at=datetime.now(timezone.utc),
                    prompt_words=words,
                    prompt_level_hint=level_hint,
                    translation_attempts=0,
                    words_complete=False,
                    sentences_complete=False,
                )
                
                global_db.add(rt)
                global_db.commit()
                global_db.refresh(rt)
                
                # Extract title
                title_service = TitleExtractionService()
                title = title_service.extract_title(content, lang)
                if title:
                    rt.title = title
                    global_db.commit()
                
                # Mark as content ready
                state_manager = GenerationStateManager()
                state_manager.transition_to_content_ready(
                    global_db, rt.id, content, words, level_hint
                )
                
                logger.info(f"Generated text content {rt.id} for account {account_id}")
                return rt
                
        except Exception as e:
            logger.error(f"Error generating text content: {e}")
            return None
    
    async def generate_translations(
        self,
        text_id: int,
        lang: str,
        target_lang: str,
    ) -> bool:
        """Generate word and sentence translations for a text."""
        try:
            with GlobalSessionLocal() as global_db:
                rt = global_db.get(ReadingText, text_id)
                if not rt:
                    logger.error(f"Text {text_id} not found")
                    return False
                
                if not rt.content:
                    logger.error(f"Text {text_id} has no content")
                    return False
                
                # Generate word translations
                word_success = await self._generate_word_translations(
                    global_db, rt, lang, target_lang
                )
                
                # Generate sentence translations
                sentence_success = await self._generate_sentence_translations(
                    global_db, rt, lang, target_lang
                )
                
                if word_success and sentence_success:
                    # Mark as fully ready
                    state_manager = GenerationStateManager()
                    state_manager.mark_fully_ready(global_db, text_id)
                    
                    logger.info(f"Completed translations for text {text_id}")
                    return True
                else:
                    logger.error(f"Failed translations for text {text_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error generating translations for text {text_id}: {e}")
            return False
    
    async def _generate_word_translations(
        self,
        global_db: Session,
        rt: ReadingText,
        lang: str,
        target_lang: str,
    ) -> bool:
        """Generate word translations."""
        try:
            # Extract spans and compute glosses
            spans = compute_spans(rt.content)
            
            if not spans:
                logger.warning(f"No spans found for text {rt.id}")
                return True
            
            # Generate translations in batches
            batch_size = 10
            for i in range(0, len(spans), batch_size):
                batch = spans[i:i + batch_size]
                
                # Build translation prompt
                prompt = build_word_translation_prompt(
                    lang=lang,
                    target_lang=target_lang,
                    words=batch,
                )
                
                # Make LLM request
                model = _pick_openrouter_model()
                response = await chat_complete_with_raw(
                    messages=prompt,
                    model=model,
                )
                
                # Extract translations
                translations = extract_word_translations(response)
                
                # Store glosses
                for word_data in batch:
                    surface = word_data["surface"]
                    start = word_data["span_start"]
                    end = word_data["span_end"]
                    
                    # Find translation for this word
                    translation = None
                    for trans in translations:
                        if trans.get("surface") == surface:
                            translation = trans.get("translation")
                            break
                    
                    # Create gloss entry
                    gloss = ReadingWordGloss(
                        text_id=rt.id,
                        lang=lang,
                        target_lang=target_lang,
                        surface=surface,
                        lemma=word_data.get("lemma"),
                        pos=word_data.get("pos"),
                        pinyin=word_data.get("pinyin"),
                        translation=translation,
                        grammar=word_data.get("grammar", {}),
                        span_start=start,
                        span_end=end,
                    )
                    
                    global_db.add(gloss)
                
                global_db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating word translations: {e}")
            return False
    
    async def _generate_sentence_translations(
        self,
        global_db: Session,
        rt: ReadingText,
        lang: str,
        target_lang: str,
    ) -> bool:
        """Generate sentence translations."""
        try:
            # Split text into sentences
            sentences = split_sentences(rt.content, lang)
            
            if not sentences:
                logger.warning(f"No sentences found for text {rt.id}")
                return True
            
            # Build translation contexts
            contexts = build_translation_contexts(
                text=rt.content,
                sentences=sentences,
                target_lang=target_lang,
                lang=lang,
                existing_translations=[],  # TODO: Add existing translations
            )
            
            # Generate translations in batches
            batch_size = 5
            for i in range(0, len(contexts), batch_size):
                batch = contexts[i:i + batch_size]
                
                # Make LLM request
                model = _pick_openrouter_model()
                response = await chat_complete_with_raw(
                    messages=[{"role": "user", "content": batch}],
                    model=model,
                )
                
                # Extract translations
                translations = extract_structured_translation(response)
                
                # Store translations
                for j, sentence in enumerate(sentences[i:i + batch_size]):
                    if j < len(translations):
                        trans = ReadingTextTranslation(
                            text_id=rt.id,
                            target_lang=target_lang,
                            unit="sentence",
                            segment_index=i + j,
                            source_text=sentence,
                            translated_text=translations[j],
                            provider="llm",
                            model=model,
                        )
                        global_db.add(trans)
                
                global_db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating sentence translations: {e}")
            return False
    
    def _get_word_list_from_profile(self, profile: Profile) -> List[Dict]:
        """Get word list for text generation from profile."""
        # TODO: Implement vocabulary selection based on profile
        return []


# Pool Selection Service
class PoolSelectionService:
    """Manages global text pool and selection."""
    
    def get_available_texts(
        self,
        global_db: Session,
        account_db: Session,
        profile: Profile,
        limit: int = 10,
    ) -> List[Tuple[ReadingText, float]]:
        """Get available texts for this profile, scored by match."""
        # Get texts this profile hasn't read
        read_texts = {ptr.text_id for ptr in account_db.query(ProfileTextRead).filter(
            ProfileTextRead.profile_id == profile.id
        ).all()}
        
        # Query available texts
        available_texts = global_db.query(ReadingText).filter(
            ReadingText.lang == profile.lang,
            ReadingText.target_lang == profile.target_lang,
            ReadingText.content.isnot(None),
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
            ~ReadingText.id.in_(read_texts),
        ).all()
        
        # Score and sort texts
        scored_texts = []
        for text in available_texts:
            score = self._calculate_match_score(profile, text)
            scored_texts.append((text, score))
        
        scored_texts.sort(key=lambda x: x[1])
        
        return scored_texts[:limit]
    
    def _calculate_match_score(self, profile: Profile, text: ReadingText) -> float:
        """Calculate how well text matches profile preferences."""
        # Simple scoring based on CI target
        if text.ci_target and profile.ci_preference:
            ci_diff = abs(text.ci_target - profile.ci_preference)
            return ci_diff
        
        return 0.5  # Default middle score
    
    def select_next_text(
        self,
        global_db: Session,
        account_db: Session,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """Select the next text for the profile."""
        scored_texts = self.get_available_texts(global_db, account_db, profile, limit=1)
        
        if scored_texts:
            return scored_texts[0][0]
        
        return None


# Retry Service
class RetryService:
    """Handles retry logic for failed generation components."""
    
    async def retry_failed_translations(
        self,
        text_id: int,
    ) -> bool:
        """Retry failed translations for a text."""
        try:
            with GlobalSessionLocal() as global_db:
                rt = global_db.get(ReadingText, text_id)
                if not rt:
                    return False
                
                # Check retry limits
                if rt.translation_attempts >= 3:
                    logger.error(f"Text {text_id} exceeded retry limit")
                    return False
                
                # Create retry attempt record
                retry = GenerationRetryAttempt(
                    account_id=rt.account_id or 0,
                    text_id=text_id,
                    attempt_number=rt.translation_attempts + 1,
                    status="pending",
                )
                global_db.add(retry)
                
                # Update text
                rt.translation_attempts += 1
                rt.last_translation_attempt = datetime.now(timezone.utc)
                
                global_db.commit()
                
                # Retry generation
                generation_service = ReadingGenerationService()
                success = await generation_service.generate_translations(
                    text_id,
                    rt.lang,
                    rt.target_lang,
                )
                
                # Update retry record
                retry.status = "completed" if success else "failed"
                retry.completed_at = datetime.now(timezone.utc)
                global_db.commit()
                
                return success
                
        except Exception as e:
            logger.error(f"Error retrying translations for text {text_id}: {e}")
            return False


# Orchestration Service
class ReadingOrchestrator:
    """High-level orchestration of reading text generation and delivery."""
    
    def __init__(self):
        self.generation_service = ReadingGenerationService()
        self.pool_service = PoolSelectionService()
        self.retry_service = RetryService()
        self.state_manager = GenerationStateManager()
    
    async def ensure_next_text_ready(
        self,
        account_id: int,
        profile_id: int,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """Ensure there's a ready text for the profile."""
        with GlobalSessionLocal() as global_db, db_manager.get_account_db(account_id) as account_db:
            # Check for existing ready text
            next_text = self.pool_service.select_next_text(global_db, account_db, profile)
            
            if next_text:
                # Mark as opened
                next_text.opened_at = datetime.now(timezone.utc)
                global_db.commit()
                
                # Update profile current text
                profile.current_text_id = next_text.id
                account_db.commit()
                
                return next_text
            
            # No ready text, start generation
            await self._trigger_background_generation(account_id, profile_id, profile)
            
            return None
    
    async def _trigger_background_generation(
        self,
        account_id: int,
        profile_id: int,
        profile: Profile,
    ) -> None:
        """Trigger background generation of new text."""
        try:
            # Generate text content
            rt = await self.generation_service.generate_text_content(
                account_id,
                profile_id,
                profile.lang,
                profile.target_lang,
                profile,
            )
            
            if not rt:
                logger.error(f"Failed to generate text for account {account_id}")
                return
            
            # Generate translations
            success = await self.generation_service.generate_translations(
                rt.id,
                profile.lang,
                profile.target_lang,
            )
            
            if success:
                # Notify client
                notification_service = get_notification_service()
                await notification_service.notify_text_ready(account_id, rt.id)
            else:
                # Schedule retry
                asyncio.create_task(self.retry_service.retry_failed_translations(rt.id))
                
        except Exception as e:
            logger.error(f"Error in background generation: {e}")


# Service instances
_generating_service = None


def get_reading_orchestrator() -> ReadingOrchestrator:
    """Get the reading orchestrator service instance."""
    global _generating_service
    if _generating_service is None:
        _generating_service = ReadingOrchestrator()
    return _generating_service


def get_state_manager() -> GenerationStateManager:
    """Get the state manager service instance."""
    return GenerationStateManager()


def get_pool_selection_service() -> PoolSelectionService:
    """Get the pool selection service instance."""
    return PoolSelectionService()


def get_retry_service() -> RetryService:
    """Get the retry service instance."""
    return RetryService()
