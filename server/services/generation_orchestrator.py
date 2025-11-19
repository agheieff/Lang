from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from ..utils.session_manager import db_manager
from ..db import GlobalSessionLocal
from sqlalchemy.orm import Session

from .retry_service import RetryService
from .readiness_service import ReadinessService
from .state_manager import GenerationStateManager
from .text_generation_service import TextGenerationService, TextGenerationResult
from .translation_service import TranslationService
from .notification_service import get_notification_service
from .llm_common import build_reading_prompt_spec


# Cross-process locking utilities
from ..utils.file_lock import FileLock


def retry_failed_components(account_id: int, text_id: int) -> dict:
    """Main retry entry point - retry all missing components for a text.
    Uses retry_actions to regenerate only the missing parts.
    """
    from .retry_service import RetryService
    from .readiness_service import ReadinessService
    from .retry_actions import retry_missing_words as _retry_words, retry_missing_sentences as _retry_sents

    retry_service = RetryService()
    readiness = ReadinessService()

    results = {"words": False, "sentences": False, "error": None}

    try:
        with db_manager.transaction(account_id) as db:
            failed_components = readiness.get_failed_components(db, account_id, text_id)
            if not (failed_components.get("words") or failed_components.get("sentences")):
                return results

            can_retry, reason = retry_service.can_retry(db, account_id, text_id, failed_components)
            if not can_retry:
                results["error"] = reason or "cannot_retry"
                return results

            retry_record = retry_service.create_retry_attempt(db, account_id, text_id, failed_components)
            if not retry_record:
                results["error"] = "failed_to_create_retry_record"
                return results

            log_dir = retry_service.get_existing_log_directory(account_id, text_id)
            if not log_dir:
                results["error"] = "no_existing_log_directory"
                return results

            completed = {}
            if failed_components.get("words"):
                results["words"] = _retry_words(account_id, text_id, log_dir)
                completed["words"] = results["words"]
            if failed_components.get("sentences"):
                results["sentences"] = _retry_sents(account_id, text_id, log_dir)
                completed["sentences"] = results["sentences"]

            error_msg = None if all(
                [not failed_components.get("words") or results["words"],
                 not failed_components.get("sentences") or results["sentences"]]
            ) else "partial_failure"
            retry_service.mark_retry_completed(db, getattr(retry_record, "id", None), completed, error_details=error_msg)

    except Exception as e:
        try:
            print(f"[RETRY] General error in retry_failed_components for text_id={text_id}: {e}")
        except Exception:
            pass
        results["error"] = str(e)

    return results


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
    
    def __init__(self):
        self.retry_service = RetryService()
        self.readiness_service = ReadinessService()
        self.state_manager = GenerationStateManager()
        self.text_gen_service = TextGenerationService()
        self.translation_service = TranslationService()
        self.notification_service = get_notification_service()
        self.file_lock = FileLock(ttl_seconds=float(os.getenv("ARC_GEN_LOCK_TTL_SEC", "300")))
    
    def ensure_text_available(self, db: Session, account_id: int, lang: str) -> None:
        """
        SINGLE entry point for ensuring text availability.
        
        This is THE ONLY method that should start text generation.
        All other parts of the system call this method.
        
        Always generates a new text if and only if there are no unopened texts.
        This maintains exactly one backup text without race conditions.
        
        Args:
            db: Database session
            account_id: User account ID  
            lang: Language code
        """
        try:
            # Count unopened texts that have content
            unopened_count = self.state_manager.count_unopened_ready(db, account_id, lang)
            
            # Always skip generation if we have unopened texts
            if unopened_count > 0:
                return  # Already have texts available
            
            # Check if generation is already in progress
            if self.text_gen_service.is_generation_in_progress(account_id, lang):
                return
            
            # Check if there's a text being generated
            generating_text = self.state_manager.get_generating_text(db, account_id, lang)
            if generating_text:
                return  # Already generating
            
            # Start new generation
            self._start_generation_job(db, account_id, lang)
            
        except Exception as e:
            print(f"[ORCHESTRATOR] Error in ensure_text_available: {e}")
            try:
                db.rollback()
            except Exception:
                pass
    
    def _start_generation_job(self, db: Session, account_id: int, lang: str) -> None:
        """Start a full generation job in a background thread."""
        if not self.text_gen_service.mark_generation_started(account_id, lang):
            return  # Already in progress
        
        # Fire-and-forget via a dedicated thread (use asyncio.run for any async stubs)
        def _worker():
            try:
                asyncio.run(self._run_generation_job(db, account_id, lang))
            except Exception as e:
                print(f"[ORCHESTRATOR] Worker exception: {e}")
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
            print(f"[ORCHESTRATOR] Starting generation job for account_id={account_id} lang={lang}")
            
            # Step 1: Acquire cross-process lock
            lock_path = await self._acquire_generation_lock(account_id, lang)
            if lock_path is None:
                return
            
            # Step 2: Create placeholder text and set up job
            text_id, job_dir = await self._create_placeholder_and_setup_job(account_id, lang)
            
            # Step 3: Generate text content
            result = await self._generate_main_text_content(
                account_id, lang, text_id, job_dir
            )
            
            if not result.success:
                await self._handle_generation_failure(
                    account_id, lang, text_id, result.error or "Generation failed"
                )
                return
            
            # Step 4: Notify content is ready
            await self._notify_content_ready(account_id, lang, text_id)
            
            # Step 5: Start translation job in background
            await self._start_translation_job_async(
                account_id, lang, text_id, 
                result.text, result.title, job_dir, result.messages
            )
            
            print(f"[ORCHESTRATOR] Generation job completed successfully for text_id={text_id}")
            
        except Exception as e:
            print(f"[ORCHESTRATOR] Generation job failed: {e}")
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
                print(f"[ORCHESTRATOR] Could not acquire lock for account_id={account_id} lang={lang}")
            return lock_path
        except Exception as e:
            print(f"[ORCHESTRATOR] Error acquiring lock: {e}")
            return None
    
    async def _release_generation_lock(self, lock_path: Optional[Path]) -> None:
        """Release cross-process lock if held."""
        try:
            self.file_lock.release(lock_path)
        except Exception as e:
            print(f"[ORCHESTRATOR] Error releasing lock: {e}")
    
    async def _create_placeholder_and_setup_job(self, account_id: int, lang: str) -> Tuple[int, Path]:
        """Create placeholder text and set up job directory."""
        with db_manager.transaction(account_id) as gen_db:
            # Create placeholder text
            text_id = self.text_gen_service.create_placeholder_text(gen_db, account_id, lang)
            if not text_id:
                raise Exception("Failed to create placeholder text")
            
            # Set up logging directory
            job_dir = self._get_job_dir(account_id, lang)
            
            # Notify client generation has started
            self.notification_service.send_generation_started(account_id, lang, text_id)
            
            return text_id, job_dir
    
    async def _generate_main_text_content(self, account_id: int, lang: str, text_id: int, job_dir: Path) -> 'TextGenerationResult':
        """Generate the main text content."""
        with db_manager.transaction(account_id) as gen_db:
            # Get global database session for account lookups
            global_db = GlobalSessionLocal()
            try:
                # Build prompt specification
                spec, words, level_hint = build_reading_prompt_spec(gen_db, account_id=account_id, lang=lang)
                from ..llm import build_reading_prompt
                messages = build_reading_prompt(spec)
                
                # Generate text content
                result = self.text_gen_service.generate_text_content(
                    gen_db, global_db, account_id, lang, text_id, job_dir, messages
                )
                
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
        """Start translation job in background thread."""
        thread = threading.Thread(
            target=self._run_translation_job,
            args=(account_id, lang, text_id, text_content, text_title, job_dir, reading_messages),
            name=f"translations-{account_id}-{text_id}",
            daemon=True,
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
        try:
            # Get database sessions
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
                        provider="openrouter",  # TODO: Make this configurable
                        model_id=None,  # Let translation service pick model
                        base_url=None
                    )
                finally:
                    # Always close global database session
                    global_db.close()
            
            if translation_result.success:
                # Notify translations are ready
                self.notification_service.send_translations_ready(account_id, lang, text_id)
                print(f"[ORCHESTRATOR] Translations completed for text_id={text_id}")
            else:
                print(f"[ORCHESTRATOR] Translation job failed for text_id={text_id}: {translation_result.error}")
                # For partial failures, still notify that something is ready
                if translation_result.words or translation_result.sentences:
                    self.notification_service.send_translations_ready(account_id, lang, text_id)
            
        except Exception as e:
            print(f"[ORCHESTRATOR] Translation job failed: {e}")
    
    def _get_job_dir(self, account_id: int, lang: str) -> Path:
        """Get/create directory for job logs."""
        base = Path.cwd() / "data" / "llm_stream_logs" / str(account_id) / lang
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        job_dir = base / ts
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir
    
    def check_and_retry_failed_texts(self, db: Session, account_id: int, lang: str) -> list:
        """Check for failed texts and attempt retries if allowed."""
        try:
            print(f"[RETRY] Checking for failed texts to retry for account_id={account_id} lang={lang}")
        except Exception:
            pass
        
        failed_texts = self.retry_service.get_failed_texts_for_retry(db, account_id, lang, limit=3)
        
        try:
            if not failed_texts:
                print("[RETRY] No failed texts found - everything looks good!")
            else:
                print(f"[RETRY] Found {len(failed_texts)} failed texts, retrying: {[t.id for t in failed_texts]}")
        except Exception:
            pass
        
        # Trigger retries in background threads to avoid blocking
        for text in failed_texts:
            def retry_in_background():
                try:
                    result = retry_failed_components(account_id, text.id)
                    if result.get("error"):
                        try:
                            print(f"[RETRY] Retry failed for text_id={text.id}: {result['error']}")
                        except Exception:
                            pass
                    else:
                        try:
                            print(f"[RETRY] Successfully initiated retry for text_id={text.id}")
                        except Exception:
                            pass
                            
                except Exception as e:
                    try:
                        print(f"[RETRY] Error retrying text_id={text.id}: {e}")
                    except Exception:
                        pass
            
            thread = threading.Thread(target=retry_in_background, daemon=True, name=f"retry-{account_id}-{text.id}")
            thread.start()
        
        return [{"text_id": t.id, "status": "retry_initiated"} for t in failed_texts]
