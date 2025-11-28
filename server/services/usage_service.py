"""
Usage tracking service for Free tier quota enforcement.
Tracks monthly text generation usage by character count and text count.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from ..config import FREE_TIER_CHAR_LIMIT, FREE_TIER_TEXT_LIMIT
from ..models import UsageTracking

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when user exceeds their monthly quota"""
    pass


class UsageService:
    """Manages usage tracking and quota enforcement for Free tier users"""

    def _get_period_start(self) -> datetime:
        """Get the first moment of the current month (UTC)"""
        now = datetime.now(timezone.utc)
        return datetime(now.year, now.month, 1, tzinfo=timezone.utc)

    def _get_or_create_usage(self, db: Session, account_id: int) -> UsageTracking:
        """Get or create usage record for current period"""
        period_start = self._get_period_start()
        
        usage = db.query(UsageTracking).filter(
            UsageTracking.account_id == account_id,
            UsageTracking.period_start == period_start
        ).first()
        
        if not usage:
            usage = UsageTracking(
                account_id=account_id,
                period_start=period_start,
                texts_generated=0,
                chars_generated=0,
            )
            db.add(usage)
            db.flush()
        
        return usage

    def check_quota(self, db: Session, account_id: int, tier: str) -> Tuple[bool, str]:
        """
        Check if user can generate another text.
        
        Args:
            db: Database session
            account_id: User's account ID
            tier: User's subscription tier
            
        Returns:
            (can_generate, reason) tuple
        """
        if tier != "Free":
            return (True, "ok")
        
        try:
            usage = self._get_or_create_usage(db, account_id)
            
            if usage.texts_generated >= FREE_TIER_TEXT_LIMIT:
                return (False, f"Monthly text limit reached ({FREE_TIER_TEXT_LIMIT} texts)")
            
            if usage.chars_generated >= FREE_TIER_CHAR_LIMIT:
                return (False, f"Monthly character limit reached ({FREE_TIER_CHAR_LIMIT:,} chars)")
            
            return (True, "ok")
            
        except Exception as e:
            logger.error(f"Error checking quota for account {account_id}: {e}")
            return (True, "ok")

    def record_usage(self, db: Session, account_id: int, text_length: int) -> None:
        """
        Record usage after successful text generation.
        
        Args:
            db: Database session
            account_id: User's account ID
            text_length: Length of generated text in characters
        """
        try:
            usage = self._get_or_create_usage(db, account_id)
            usage.texts_generated += 1
            usage.chars_generated += text_length
            usage.last_updated = datetime.now(timezone.utc)
            db.flush()
            
            logger.info(
                f"[USAGE] Recorded for account {account_id}: "
                f"+1 text, +{text_length} chars "
                f"(total: {usage.texts_generated} texts, {usage.chars_generated:,} chars)"
            )
            
        except Exception as e:
            logger.error(f"Error recording usage for account {account_id}: {e}")

    def get_usage_stats(self, db: Session, account_id: int, tier: str) -> dict:
        """
        Get usage statistics for display.
        
        Args:
            db: Database session
            account_id: User's account ID
            tier: User's subscription tier
            
        Returns:
            Dict with usage stats and limits
        """
        if tier != "Free":
            return {
                "tier": tier,
                "texts_generated": None,
                "chars_generated": None,
                "text_limit": None,
                "char_limit": None,
                "texts_remaining": None,
                "chars_remaining": None,
            }
        
        try:
            usage = self._get_or_create_usage(db, account_id)
            
            return {
                "tier": tier,
                "texts_generated": usage.texts_generated,
                "chars_generated": usage.chars_generated,
                "text_limit": FREE_TIER_TEXT_LIMIT,
                "char_limit": FREE_TIER_CHAR_LIMIT,
                "texts_remaining": max(0, FREE_TIER_TEXT_LIMIT - usage.texts_generated),
                "chars_remaining": max(0, FREE_TIER_CHAR_LIMIT - usage.chars_generated),
                "period_start": usage.period_start.isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error getting usage stats for account {account_id}: {e}")
            return {
                "tier": tier,
                "error": str(e),
            }


_usage_service: Optional[UsageService] = None


def get_usage_service() -> UsageService:
    global _usage_service
    if _usage_service is None:
        _usage_service = UsageService()
    return _usage_service
