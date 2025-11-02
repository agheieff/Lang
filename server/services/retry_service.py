from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import json

from sqlalchemy.orm import Session

from ..models import GenerationRetryAttempt, ReadingText
from .readiness_service import ReadinessService
from .gen_queue import _job_dir, _log_dir_root


class RetryService:
    """Service for managing retry attempts for failed generation components."""
    
    COMPONENT_BITMASKS = {
        "words": 1,
        "sentences": 2,
        "structured": 4,
    }
    
    def __init__(self):
        self.settings = self._get_settings()
        self.readiness = ReadinessService()
    
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
        
        # Check existing attempts
        attempts = (
            db.query(GenerationRetryAttempt)
            .filter(
                GenerationRetryAttempt.account_id == account_id,
                GenerationRetryAttempt.text_id == text_id,
            )
            .order_by(GenerationRetryAttempt.attempt_number.desc())
            .all()
        )
        
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
        failed_bitmap = 0
        if components.get("words", False):
            failed_bitmap |= self.COMPONENT_BITMASKS["words"]
        if components.get("sentences", False):
            failed_bitmap |= self.COMPONENT_BITMASKS["sentences"]
        
        # Get next attempt number
        last_attempt = (
            db.query(GenerationRetryAttempt)
            .filter(
                GenerationRetryAttempt.account_id == account_id,
                GenerationRetryAttempt.text_id == text_id,
            )
            .order_by(GenerationRetryAttempt.attempt_number.desc())
            .first()
        )
        
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
    
    def mark_retry_completed(self, db: Session, retry_id: int, completed_components: dict, error_details: Optional[str] = None) -> bool:
        """Mark a retry attempt as completed."""
        retry = db.query(GenerationRetryAttempt).filter(GenerationRetryAttempt.id == retry_id).first()
        if not retry:
            return False
        
        completed_bitmap = 0
        if completed_components.get("words", False):
            completed_bitmap |= self.COMPONENT_BITMASKS["words"]
        if completed_components.get("sentences", False):
            completed_bitmap |= self.COMPONENT_BITMASKS["sentences"]
        
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
    
    def get_existing_log_directory(self, account_id: int, text_id: int) -> Optional[Path]:
        """Find the existing log directory for a text to reuse for retry."""
        base = _log_dir_root() / str(account_id) / "zh"  # TODO: pass lang properly
        if not base.exists():
            return None
        
        # Look for the most recent directory containing this text_id
        for job_dir in sorted(base.iterdir(), reverse=True):
            if not job_dir.is_dir():
                continue
            
            meta_file = job_dir / "meta.json"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("text_id") == text_id:
                        return job_dir
                except Exception:
                    continue
        
        return None
    
    def get_failed_texts_for_retry(self, db: Session, account_id: int, lang: str, limit: int = 5) -> list[ReadingText]:
        """Get texts that need retry, ordered by creation date."""
        # Get texts that are not opened but have content and are missing components
        texts = (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.opened_at.is_(None),
                ReadingText.content.is_not(None),
            )
            .all()
        )
        
        failed_texts = []
        for text in texts:
            if self.readiness.needs_retry(db, account_id, text.id):
                can_retry, reason = self.can_retry(db, account_id, text.id, {})
                if can_retry:
                    failed_texts.append(text)
                    if len(failed_texts) >= limit:
                        break
        
        return failed_texts
