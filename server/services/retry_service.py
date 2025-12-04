from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import json

from sqlalchemy.orm import Session

from ..models import GenerationRetryAttempt, ReadingText
from ..enums import RetryComponent
from ..services.user_content_service import UserContentService


def _job_dir(account_id: int, lang: str) -> Path:
    """Create job directory for generation logging."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = Path.cwd() / "data" / "llm_stream_logs"
    base.mkdir(parents=True, exist_ok=True)
    d = base / str(int(account_id)) / lang / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_dir_root() -> Path:
    """Get root directory for LLM stream logs."""
    base = os.getenv("ARC_OR_LOG_DIR", str(Path.cwd() / "data" / "llm_stream_logs"))
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


class RetryService:
    """Service for managing retry attempts for failed generation components."""
    
    def __init__(self):
        self.settings = self._get_settings()
        self.user_content = UserContentService()
    
    def _get_settings(self) -> dict:
        """Load retry-related settings."""
        return {
            "max_attempts": int(os.getenv("ARC_GEN_RETRY_MAX_ATTEMPTS", "3")),
            "cooldown_base": float(os.getenv("ARC_GEN_RETRY_COOLDOWN_BASE", "60")),
            "cooldown_max": float(os.getenv("ARC_GEN_RETRY_COOLDOWN_MAX", "300")),
        }
    
    def can_retry(self, db: Session, account_id: int, text_id: int, components: dict) -> tuple[bool, Optional[str]]:
        """Check if we can retry failed components.
        
        Returns:
            (can_retry, reason) where reason is human-readable if can_retry is False
        """
        if not components.get("words", False) and not components.get("sentences", False):
            return (False, "no_failed_components")
        
        # Check existing attempts - handle case where table doesn't exist yet
        attempts = []
        try:
            attempts = (
                db.query(GenerationRetryAttempt)
                .filter(
                    GenerationRetryAttempt.account_id == account_id,
                    GenerationRetryAttempt.text_id == text_id,
                )
                .order_by(GenerationRetryAttempt.attempt_number.desc())
                .all()
            )
        except Exception as e:
            # Table doesn't exist - this is expected for first-time use
            if "no such table" in str(e):
                pass
            else:
                raise
        
        if len(attempts) >= self.settings["max_attempts"]:
            return (False, f"max_attempts_reached ({self.settings['max_attempts']})")
        
        # Check cooldown
        if attempts:
            last_attempt = attempts[0]
            if last_attempt.status in ["pending"]:
                return (False, "retry_in_progress")
            
            # Exponential backoff: cooldown = base * (2 ^ (attempt - 1))
            cooldown_seconds = min(
                self.settings["cooldown_max"],
                self.settings["cooldown_base"] * (2 ** (len(attempts) - 1))
            )
            
            if last_attempt.completed_at:
                time_since = datetime.utcnow() - last_attempt.completed_at
                if time_since.total_seconds() < cooldown_seconds:
                    remaining = cooldown_seconds - time_since.total_seconds()
                    return (False, f"cooldown_remaining ({remaining:.0f}s)")
        
        return (True, None)
    
    def create_retry_attempt(self, db: Session, account_id: int, text_id: int, components: dict) -> Optional[GenerationRetryAttempt]:
        """Create a new retry attempt record."""
        try:
            failed_bitmap = 0
            if components.get("words", False):
                failed_bitmap |= RetryComponent.WORDS
            if components.get("sentences", False):
                failed_bitmap |= RetryComponent.SENTENCES
            
            # Get next attempt number
            last_attempt = None
            try:
                last_attempt = (
                    db.query(GenerationRetryAttempt)
                    .filter(
                        GenerationRetryAttempt.account_id == account_id,
                        GenerationRetryAttempt.text_id == text_id,
                    )
                    .order_by(GenerationRetryAttempt.attempt_number.desc())
                    .first()
                )
            except Exception as e:
                # Table doesn't exist - skip tracking
                if "no such table" in str(e):
                    pass
                else:
                    raise
            
            attempt_num = (last_attempt.attempt_number + 1) if last_attempt else 1
            
            retry = GenerationRetryAttempt(
                account_id=account_id,
                text_id=text_id,
                failed_components=failed_bitmap,
                attempt_number=attempt_num,
                status="pending",
            )
            
            db.add(retry)
            try:
                db.commit()
                return retry
            except Exception:
                db.rollback()
                return None
        except Exception as e:
            # Table doesn't exist - return dummy retry object
            if "no such table" in str(e):
                class DummyRetry:
                    def __init__(self):
                        self.id = None
                        self.status = "pending"
                return DummyRetry()
            else:
                raise
    
    def mark_retry_completed(self, db: Session, retry_id: int, completed_components: dict, error_details: Optional[str] = None) -> bool:
        """Mark a retry attempt as completed."""
        # Disable retry tracking when table doesn't exist
        if retry_id is None:
            return True  # Dummy retry - return success
            
        try:
            retry = db.query(GenerationRetryAttempt).filter(GenerationRetryAttempt.id == retry_id).first()
            if not retry:
                return False
            
            completed_bitmap = 0
            if completed_components.get("words", False):
                completed_bitmap |= RetryComponent.WORDS
            if completed_components.get("sentences", False):
                completed_bitmap |= RetryComponent.SENTENCES
            
            retry.completed_components = completed_bitmap
            retry.completed_at = datetime.utcnow()
            retry.error_details = error_details
            
            if error_details:
                retry.status = "failed"
            elif completed_bitmap == retry.failed_components:
                retry.status = "completed"
            else:
                retry.status = "partial"
            
            try:
                db.commit()
                return True
            except Exception:
                db.rollback()
                return False
        except Exception as e:
            if "no such table" in str(e):
                return True  # Table doesn't exist - skip tracking
            raise
    
    def get_existing_log_directory(self, account_id: int, text_id: int) -> Optional[Path]:
        """Find the existing log directory for a text across all language folders.
        Picks the most recent job dir with matching meta.text_id.
        """
        acc_root = _log_dir_root() / str(account_id)
        if not acc_root.exists():
            return None

        candidates: list[Path] = []
        for lang_dir in sorted([p for p in acc_root.iterdir() if p.is_dir()]):
            for job_dir in sorted([p for p in lang_dir.iterdir() if p.is_dir()], reverse=True):
                meta_file = job_dir / "meta.json"
                if not meta_file.exists():
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("text_id") == text_id:
                        candidates.append(job_dir)
                except Exception:
                    continue
        if not candidates:
            return None
        # Return most recent by mtime
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    
    def get_failed_texts_for_retry(self, db: Session, account_id: int, lang: str, limit: int = 5) -> list[ReadingText]:
        """Get texts that need retry, ordered by creation date."""
        # Get texts that are not opened but have content and are missing components
        # Also check that the text is old enough and likely finished generation
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)  # Only retry texts older than 5 mins
        texts = (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.opened_at.is_(None),
                ReadingText.content.is_not(None),
                ReadingText.created_at < cutoff_time,  # Only retry older texts
            )
            .all()
        )
        
        failed_texts = []
        try:
            for text in texts:
                failed_components = self.user_content.get_failed_components(db, text.id, text.target_lang)
                needs_retry = failed_components.get("words") or failed_components.get("sentences")
                
                if needs_retry:
                    failed_components = self.user_content.get_failed_components(db, text.id, text.target_lang)
                    can_retry, reason = self.can_retry(db, account_id, text.id, failed_components)
                    
                    if can_retry:
                        failed_texts.append(text)
                        if len(failed_texts) >= limit:
                            break
        except Exception as e:
            # Log error but don't break the flow
            pass
                
        return failed_texts

    def retry_failed_components(self, account_id: int, text_id: int) -> dict:
        """Main retry entry point - retry all missing components for a text.
        Uses retry_actions to regenerate only the missing parts.
        """
        from ..utils.session_manager import db_manager
        from .retry_actions import retry_missing_words as _retry_words, retry_missing_sentences as _retry_sents

        results = {"words": False, "sentences": False, "error": None}

        try:
            with db_manager.transaction(account_id) as db:
                # Get text to determine target_lang
                text = db.query(ReadingText).filter(ReadingText.id == text_id).first()
                if not text:
                    results["error"] = "text_not_found"
                    return results
                    
                target_lang = text.target_lang
                failed_components = self.user_content.get_failed_components(db, text.id, target_lang)
                if not (failed_components.get("words") or failed_components.get("sentences")):
                    return results

                can_retry, reason = self.can_retry(db, account_id, text_id, failed_components)
                if not can_retry:
                    results["error"] = reason or "cannot_retry"
                    return results

                retry_record = self.create_retry_attempt(db, account_id, text_id, failed_components)
                if not retry_record:
                    results["error"] = "failed_to_create_retry_record"
                    return results

                log_dir = self.get_existing_log_directory(account_id, text_id)
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
                self.mark_retry_completed(db, getattr(retry_record, "id", None), completed, error_details=error_msg)

        except Exception as e:
            results["error"] = str(e)

        return results
