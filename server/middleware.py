"""Rate limiting middleware using token bucket algorithm."""

from __future__ import annotations

import time
import logging
from collections import OrderedDict
from typing import Dict

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from server.config import RATE_LIMITS, RATE_WINDOW_SEC, SubscriptionTier

logger = logging.getLogger(__name__)

# Rate limiter configuration
MAX_BUCKETS = 10000  # Maximum number of buckets to store
BUCKET_TIMEOUT_SECONDS = 3600  # 1 hour - remove buckets after this time


class TokenBucket:
    """Simple token bucket for rate limiting."""

    def __init__(self, rate_limit: int, window_seconds: int = 60):
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.tokens = rate_limit
        self.last_update = time.time()
        self.last_access = time.time()  # Track last access for cleanup

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""

        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on elapsed time
        tokens_to_add = int(elapsed / self.window_seconds * self.rate_limit)
        self.tokens = min(self.rate_limit, self.tokens + tokens_to_add)
        self.last_update = now
        self.last_access = now  # Update access time

        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        self.tokens = self.rate_limit
        self.last_update = time.time()
        self.last_access = time.time()

    def is_stale(self, timeout_seconds: int = BUCKET_TIMEOUT_SECONDS) -> bool:
        """Check if bucket is stale and should be cleaned up."""
        return (time.time() - self.last_access) > timeout_seconds


class RateLimiter:
    """Rate limiter for API endpoints."""

    def __init__(self):
        # Use OrderedDict for LRU eviction and efficient cleanup
        self.buckets: OrderedDict[int, TokenBucket] = OrderedDict()
        self._cleanup_counter = 0

    def _cleanup_stale_buckets(self) -> None:
        """Remove stale buckets to prevent memory leak."""
        try:
            stale_keys = [
                account_id
                for account_id, bucket in self.buckets.items()
                if bucket.is_stale()
            ]

            for key in stale_keys:
                del self.buckets[key]

            if stale_keys:
                logger.info(f"Cleaned up {len(stale_keys)} stale rate limit buckets")

        except Exception as e:
            logger.error(f"Error cleaning up stale buckets: {e}")

    def _enforce_max_buckets(self) -> None:
        """Evict oldest buckets if we exceed the maximum."""
        if len(self.buckets) > MAX_BUCKETS:
            # Remove oldest entries (FIFO - OrderedDict maintains insertion order)
            excess = len(self.buckets) - MAX_BUCKETS
            for _ in range(excess):
                self.buckets.popitem(last=False)
            logger.warning(f"Evicted {excess} old rate limit buckets to enforce max limit")

    def get_bucket(self, account_id: int, tier: str) -> TokenBucket:
        """Get or create a token bucket for the account."""

        # Periodic cleanup (every 100 requests)
        self._cleanup_counter += 1
        if self._cleanup_counter >= 100:
            self._cleanup_counter = 0
            self._cleanup_stale_buckets()
            self._enforce_max_buckets()

        if account_id not in self.buckets:
            # Get rate limit based on tier
            try:
                tier_enum = SubscriptionTier(tier) if isinstance(tier, str) else tier
                rate_limit = RATE_LIMITS.get(tier_enum, 60)
            except (ValueError, KeyError):
                rate_limit = 60  # Default fallback

            self.buckets[account_id] = TokenBucket(rate_limit, RATE_WINDOW_SEC)
            logger.info(
                f"Created rate limit bucket for account {account_id}: "
                f"{rate_limit} requests per {RATE_WINDOW_SEC}s"
            )

        else:
            # Move to end to mark as recently used (for LRU eviction)
            self.buckets.move_to_end(account_id)

        return self.buckets[account_id]

    def check_rate_limit(self, account_id: int, tier: str, tokens: int = 1) -> bool:
        """Check if request is allowed."""

        bucket = self.get_bucket(account_id, tier)
        return bucket.consume(tokens)

    def reset_bucket(self, account_id: int) -> None:
        """Reset rate limit for an account."""

        if account_id in self.buckets:
            self.buckets[account_id].reset()
            logger.info(f"Reset rate limit bucket for account {account_id}")

    def get_bucket_count(self) -> int:
        """Get current number of buckets (for monitoring)."""
        return len(self.buckets)


# Global rate limiter instance
_rate_limiter = RateLimiter()


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency to check rate limits.

    Raises HTTPException if rate limit exceeded.
    """
    # Try to get account_id from request state
    account_id = getattr(request.state, "account_id", None)
    tier = getattr(request.state, "tier", "Free")

    if account_id:
        allowed = _rate_limiter.check_rate_limit(account_id, tier)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for account {account_id} (tier: {tier})"
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "status": "error",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": RATE_WINDOW_SEC,
                },
            )
