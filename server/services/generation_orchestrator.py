from __future__ import annotations

import asyncio
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from ..utils.session_manager import db_manager
from ..db import GlobalSessionLocal
from sqlalchemy.orm import Session

from .readiness_service import ReadinessService
from .state_manager import GenerationStateManager
from .text_generation_service import TextGenerationService, TextGenerationResult
from .translation_service import TranslationService
from .translation_validation_service import TranslationValidationService
from .notification_service import get_notification_service
from .llm_common import build_reading_prompt_spec
from .pool_selection_service import get_pool_selection_service
from .usage_service import get_usage_service, QuotaExceededError

logger = logging.getLogger(__name__)

# Cross-process locking utilities
from ..utils.file_lock import FileLock

class GenerationOrchestrator:
    """
    Orchestrates the text generation pipeline.
    
    This service coordinates between:
    - State management (where we are in the process)
    - Text generation (creating content)
    - Translation generation (creating translations)
    - Notifications (keeping the client updated)
    - Retry logic (recovering from failures)
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
        self.readiness_service = ReadinessService()
        self.state_manager = GenerationStateManager()
        self.text_gen_service = TextGenerationService()
        self.translation_service = TranslationService()
        self.validation_service = TranslationValidationService()
        self.notification_service = get_notification_service()
        self.file_lock = FileLock(ttl_seconds=float(os.getenv("ARC_GEN_LOCK_TTL_SEC", "300")))
        self._initialized = True
    
    def ensure_text_available(self, global_db: Session, account_id: int, lang: str) -> None:
        """
        Ensure the text pool has available texts. Triggers generation if needed.
        Called by background worker to maintain pool.
        """
        from ..models import Profile
        
        pool_service = get_pool_selection_service()
        
        # Get profile from global DB
        profile = global_db.query(Profile).filter(
            Profile.account_id == account_id,
            Profile.lang == lang
        ).first()
        
        if not profile:
            return  # No profile for this account/lang
        
        # Pool full or generation already running? Nothing to do.
        # needs_backfill expects (global_db, account_db, profile)
        # For background worker, we don't have per-account DB open, so pass global_db for both
        # (the method only uses account_db for ProfileTextRead which we'll handle separately)
        with db_manager.read_only(account_id) as account_db:
            if pool_service.needs_backfill(global_db, account_db, profile) <= 0:
                return
        
        if self.text_gen_service.is_generation_in_progress(account_id, lang):
            return
        if self.state_manager.get_generating_text(global_db, account_id, lang):
            return
        
        self._start_generation_job(global_db, account_id, lang)
    
    def retry_incomplete_texts(self, global_db: Session, account_id: int, lang: str) -> int:
        """
        Retry translation for incomplete texts that are due for retry.
        Returns number of texts queued for retry.
        """
        from ..models import ReadingText
        
        MAX_ATTEMPTS = 5
        now = datetime.utcnow()
        retried = 0
        
        # Find texts with content but incomplete translations (from GLOBAL DB)
        incomplete = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.content.isnot(None),
                # Not fully complete
                ((ReadingText.words_complete == False) | (ReadingText.sentences_complete == False)),
            )
            .all()
        )
        
        for text in incomplete:
            attempts = text.translation_attempts or 0
            last_attempt = text.last_translation_attempt
            
            # Check if max attempts reached - delete it
            if attempts >= MAX_ATTEMPTS:
                logger.warning(f"[ORCHESTRATOR] Deleting text {text.id} - max attempts ({MAX_ATTEMPTS}) reached")
                from ..models import ReadingWordGloss, ReadingTextTranslation, TextVocabulary
                global_db.query(ReadingWordGloss).filter(ReadingWordGloss.text_id == text.id).delete()
                global_db.query(ReadingTextTranslation).filter(ReadingTextTranslation.text_id == text.id).delete()
                global_db.query(TextVocabulary).filter(TextVocabulary.text_id == text.id).delete()
                global_db.delete(text)
                continue
            
            # Check backoff - exponential: 1, 2, 4, 8, 16 minutes
            if last_attempt:
                backoff_seconds = 60 * (2 ** attempts)  # 60, 120, 240, 480, 960 seconds
                seconds_since = (now - last_attempt).total_seconds()
                if seconds_since < backoff_seconds:
                    logger.debug(f"[ORCHESTRATOR] Text {text.id} in backoff ({seconds_since:.0f}s < {backoff_seconds}s)")
                    continue
            
            # Queue for retry
            logger.info(f"[ORCHESTRATOR] Retrying translation for text {text.id} (attempt {attempts + 1})")
            job_dir = self._get_job_dir(account_id, lang)
            
            # Start translation job (will increment attempt counter)
            self._start_translation_job(
                account_id, lang, text.id,
                text.content, text.title if hasattr(text, 'title') else None,
                job_dir, []  # Empty messages - translation service will handle
            )
            retried += 1
        
        if retried > 0 or incomplete:
            try:
                global_db.commit()
            except Exception:
                global_db.rollback()
        
        return retried
    
    def _start_generation_job(self, db: Session, account_id: int, lang: str) -> None:
        """Start a full generation job in a background thread."""
        if not self.text_gen_service.mark_generation_started(account_id, lang):
            return  # Already in progress
        
        # Fire-and-forget via a dedicated thread (use asyncio.run for any async stubs)
        def _worker():
            try:
                asyncio.run(self._run_generation_job(db, account_id, lang))
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Worker exception: {e}", exc_info=True)
            finally:
                # Mark generation as completed (even on error)
                self.text_gen_service.mark_generation_completed(account_id, lang)
        
        thread = threading.Thread(
            target=_worker, 
            name=f"gen-orchestrator-{account_id}-{lang}", 
            daemon=True
        )
        thread.start()
    
    async def _run_generation_job(self, db: Session, account_id: int, lang: str) -> None:
        """Run the complete generation pipeline for a text."""
        job_dir = None
        lock_path = None
        text_id = None
        
        try:
            logger.info(f"[ORCHESTRATOR] Starting generation job for account_id={account_id} lang={lang}")
            
            # Step 1: Acquire cross-process lock
            lock_path = await self._acquire_generation_lock(account_id, lang)
            if lock_path is None:
                return
            
            # Step 2: Create placeholder text and set up job
            text_id, job_dir, ci_target, topic = await self._create_placeholder_and_setup_job(account_id, lang)
            
            # Step 3: Generate text content
            result = await self._generate_main_text_content(
                account_id, lang, text_id, job_dir, ci_target, topic
            )
            
            if not result.success:
                await self._handle_generation_failure(
                    account_id, lang, text_id, result.error or "Generation failed"
                )
                return
            
            # Step 4: Record usage for Free tier tracking
            if result.text:
                with db_manager.transaction(account_id) as usage_db:
                    usage_service = get_usage_service()
                    usage_service.record_usage(usage_db, account_id, len(result.text))
            
            # Step 5: Notify content is ready
            await self._notify_content_ready(account_id, lang, text_id)
            
            # Step 6: Start translation job in background
            await self._start_translation_job_async(
                account_id, lang, text_id, 
                result.text, result.title, job_dir, result.messages
            )
            
            logger.info(f"[ORCHESTRATOR] Generation job completed successfully for text_id={text_id}")
            
            # Mark this generation as complete so background worker can start another if needed
            self.text_gen_service.mark_generation_completed(account_id, lang)
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Generation job failed: {e}", exc_info=True)
            if text_id:
                await self._handle_generation_failure(account_id, lang, text_id, str(e))
        finally:
            # Always release cross-process lock if held (both success and failure)
            await self._release_generation_lock(lock_path)
    
    async def _acquire_generation_lock(self, account_id: int, lang: str) -> Optional[Path]:
        """Acquire cross-process lock to avoid duplicate jobs across workers."""
        try:
            lock_path = self.file_lock.acquire(account_id, lang)
            if lock_path is None:
                logger.info(f"[ORCHESTRATOR] Could not acquire lock for account_id={account_id} lang={lang}")
            return lock_path
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error acquiring lock: {e}", exc_info=True)
            return None
    
    async def _release_generation_lock(self, lock_path: Optional[Path]) -> None:
        """Release cross-process lock if held."""
        try:
            self.file_lock.release(lock_path)
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error releasing lock: {e}", exc_info=True)
    
    async def _create_placeholder_and_setup_job(self, account_id: int, lang: str) -> Tuple[int, Path, float, Optional[str]]:
        """Create placeholder text and set up job directory.
        
        Returns:
            (text_id, job_dir, ci_target, topic) - params for generation
        """
        from ..models import Profile
        from ..auth import Account
        
        # Get global DB session for text creation
        global_db = GlobalSessionLocal()
        
        try:
            with db_manager.transaction(account_id) as gen_db:
                # Check Free tier quota before generating
                account = global_db.query(Account).filter(Account.id == account_id).first()
                user_tier = account.subscription_tier if account else "Free"
                
                usage_service = get_usage_service()
                can_generate, reason = usage_service.check_quota(gen_db, account_id, user_tier)
                if not can_generate:
                    raise QuotaExceededError(reason)
                
                # Get profile for pool generation params
                prof = gen_db.query(Profile).filter(
                    Profile.account_id == account_id, 
                    Profile.lang == lang
                ).first()
                
                # Get target language from profile (default to 'en')
                target_lang = prof.target_lang if prof else "en"
                
                # Get varied generation params from pool service
                pool_service = get_pool_selection_service()
                if prof:
                    ci_target, topic = pool_service.get_generation_params(prof, vary=True)
                else:
                    # Default values if no profile exists
                    ci_target, topic = 0.92, "daily_life"
            
            # Create placeholder text in GLOBAL DB
            text_id = self.text_gen_service.create_placeholder_text(
                global_db, account_id, lang,
                target_lang=target_lang,
                ci_target=ci_target, topic=topic
            )
            if not text_id:
                raise Exception("Failed to create placeholder text")
            
            global_db.commit()
            
            # Set up logging directory
            job_dir = self._get_job_dir(account_id, lang)
            
            # Notify client generation has started
            self.notification_service.send_generation_started(account_id, lang, text_id)
            
            return text_id, job_dir, ci_target, topic
        finally:
            global_db.close()
    
    async def _generate_main_text_content(
        self, 
        account_id: int, 
        lang: str, 
        text_id: int, 
        job_dir: Path,
        ci_target: Optional[float] = None,
        topic: Optional[str] = None,
    ) -> 'TextGenerationResult':
        """Generate the main text content."""
        with db_manager.transaction(account_id) as gen_db:
            # Get global database session for account lookups
            global_db = GlobalSessionLocal()
            try:
                # Build prompt specification with pool params
                spec, words, level_hint = build_reading_prompt_spec(
                    gen_db, 
                    account_id=account_id, 
                    lang=lang,
                    ci_target_override=ci_target,
                    topic=topic,
                )
                from ..llm import build_reading_prompt
                messages = build_reading_prompt(spec)
                
                # Generate text content
                result = self.text_gen_service.generate_text_content(
                    gen_db, global_db, account_id, lang, text_id, job_dir, messages
                )
                
                # Store prompt data in text record for analysis/reproducibility
                if result.success:
                    from ..models import ReadingText
                    text = global_db.get(ReadingText, text_id)
                    if text:
                        text.prompt_words = words or {}
                        text.prompt_level_hint = level_hint
                        global_db.commit()
                
                # Store messages for later use in translation
                result.messages = messages
                
                return result
            finally:
                # Always close global database session
                global_db.close()
    
    async def _notify_content_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify client that content is ready."""
        self.notification_service.send_content_ready(account_id, lang, text_id)
    
    async def _start_translation_job_async(self,
                                          account_id: int,
                                          lang: str,
                                          text_id: int,
                                          text_content: str,
                                          text_title: Optional[str],
                                          job_dir: Path,
                                          reading_messages: list) -> None:
        """Start translation job in background."""
        self._start_translation_job(
            account_id, lang, text_id, 
            text_content, text_title, job_dir, reading_messages
        )
    
    async def _handle_generation_failure(self, account_id: int, lang: str, text_id: int, error_message: str) -> None:
        """Handle generation failure by notifying client."""
        self.notification_service.send_generation_failed(account_id, lang, text_id, error_message)
    
    def _start_translation_job(self,
                               account_id: int,
                               lang: str,
                               text_id: int,
                               text_content: str,
                               text_title: Optional[str],
                               job_dir: Path,
                               reading_messages: list) -> None:
        """Start translation job in background thread (non-daemon to ensure completion)."""
        thread = threading.Thread(
            target=self._run_translation_job,
            args=(account_id, lang, text_id, text_content, text_title, job_dir, reading_messages),
            name=f"translations-{account_id}-{text_id}",
            daemon=False,  # Non-daemon so thread completes even if main exits
        )
        thread.start()
    
    def _run_translation_job(self,
                            account_id: int,
                            lang: str,
                            text_id: int,
                            text_content: str,
                            text_title: Optional[str],
                            job_dir: Path,
                            reading_messages: list) -> None:
        """Run the translation job (words + sentences) using TranslationService."""
        from ..models import ReadingText
        MAX_ATTEMPTS = 5
        
        # Track this attempt (ReadingText is now in GLOBAL DB)
        try:
            global_db = GlobalSessionLocal()
            try:
                text = global_db.get(ReadingText, text_id)
                if text:
                    text.translation_attempts = (text.translation_attempts or 0) + 1
                    text.last_translation_attempt = datetime.utcnow()
                    current_attempt = text.translation_attempts
                    global_db.commit()
                    logger.info(f"[ORCHESTRATOR] Translation attempt {current_attempt}/{MAX_ATTEMPTS} for text_id={text_id}")
                else:
                    current_attempt = 1
            finally:
                global_db.close()
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Failed to track translation attempt: {e}")
            current_attempt = 1
        
        translation_success = False
        try:
            # Get database sessions - use separate transaction for actual work
            with db_manager.transaction(account_id) as account_db:
                global_db = GlobalSessionLocal()
                try:
                    # Use the new TranslationService
                    translation_result = self.translation_service.generate_translations(
                        account_db, global_db,
                        account_id=account_id,
                        lang=lang,
                        text_id=text_id,
                        text_content=text_content,
                        text_title=text_title,
                        job_dir=job_dir,
                        reading_messages=reading_messages,
                        provider=None,
                        model_id=None,
                        base_url=None
                    )
                    translation_success = translation_result.success
                finally:
                    global_db.close()
            
            if translation_result.success:
                # Validate and potentially backfill missing translations
                completeness = self.validation_service.validate_and_backfill(account_id, text_id)
                
                logger.info(f"[ORCHESTRATOR] Translation validation for text_id={text_id}: "
                      f"words={completeness['words']}, sentences={completeness['sentences']}, title={completeness['title']}")
                
                # Notify translations are ready
                self.notification_service.send_translations_ready(account_id, lang, text_id)
                self._notify_next_ready_if_backup(account_id, lang, text_id)
                
                logger.info(f"[ORCHESTRATOR] Translations completed for text_id={text_id}")
            else:
                logger.error(f"[ORCHESTRATOR] Translation job failed for text_id={text_id}: {translation_result.error}")
                if translation_result.words or translation_result.sentences:
                    completeness = self.validation_service.validate_and_backfill(account_id, text_id)
                    self.notification_service.send_translations_ready(account_id, lang, text_id)
                    self._notify_next_ready_if_backup(account_id, lang, text_id)
            
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Translation job exception: {e}", exc_info=True)
        
        # Handle failure - check if we should delete or schedule retry
        if not translation_success:
            self._handle_translation_failure(account_id, lang, text_id, current_attempt, MAX_ATTEMPTS)
    
    def _handle_translation_failure(self, account_id: int, lang: str, text_id: int, 
                                    current_attempt: int, max_attempts: int) -> None:
        """Handle translation failure - delete text if max attempts reached."""
        from ..models import ReadingText, ReadingWordGloss, ReadingTextTranslation, TextVocabulary
        
        try:
            # ReadingText is now in GLOBAL DB
            global_db = GlobalSessionLocal()
            try:
                text = global_db.get(ReadingText, text_id)
                if not text:
                    return
                
                # Check if text is now complete (might have succeeded partially)
                if text.words_complete and text.sentences_complete:
                    logger.info(f"[ORCHESTRATOR] Text {text_id} completed despite earlier issues")
                    return
                
                if current_attempt >= max_attempts:
                    # Delete the broken text and its translations
                    logger.warning(f"[ORCHESTRATOR] Deleting text {text_id} after {max_attempts} failed attempts")
                    
                    # Delete associated data from global DB
                    global_db.query(ReadingWordGloss).filter(ReadingWordGloss.text_id == text_id).delete()
                    global_db.query(ReadingTextTranslation).filter(ReadingTextTranslation.text_id == text_id).delete()
                    global_db.query(TextVocabulary).filter(TextVocabulary.text_id == text_id).delete()
                    global_db.delete(text)
                    global_db.commit()
                    
                    logger.info(f"[ORCHESTRATOR] Deleted broken text {text_id}")
                else:
                    # Schedule retry with exponential backoff
                    backoff_minutes = 2 ** (current_attempt - 1)  # 1, 2, 4, 8, 16 minutes
                    logger.info(f"[ORCHESTRATOR] Text {text_id} will be retried (attempt {current_attempt}/{max_attempts}, backoff {backoff_minutes}m)")
            finally:
                global_db.close()
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Error handling translation failure: {e}")
    
    def _notify_next_ready_if_backup(self, account_id: int, lang: str, text_id: int) -> None:
        """Send next_ready event if this text is a backup (ready but not assigned) text."""
        try:
            from ..models import ReadingText
            # ReadingText is now in GLOBAL DB
            global_db = GlobalSessionLocal()
            try:
                text = global_db.get(ReadingText, text_id)
                
                # Send notification if text is ready
                if text and text.words_complete and text.sentences_complete:
                    self.notification_service.send_next_ready(account_id, lang, text_id)
            finally:
                global_db.close()
        except Exception as e:
            logger.warning(f"[ORCHESTRATOR] Failed to check/send next_ready: {e}")
    
    def _get_job_dir(self, account_id: int, lang: str) -> Path:
        """Get/create directory for job logs."""
        base = Path.cwd() / "data" / "llm_stream_logs" / str(account_id) / lang
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        job_dir = base / ts
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir


def get_generation_orchestrator() -> GenerationOrchestrator:
    """Get the singleton GenerationOrchestrator instance."""
    return GenerationOrchestrator()
