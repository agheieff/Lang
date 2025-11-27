"""
Pool-based text selection service.
Manages a pool of pre-generated texts with varied stats and selects
the best match based on proximity to user's current state.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from ..config import POOL_SIZE, POOL_CI_VARIANCE
from ..models import Profile, ReadingText

logger = logging.getLogger(__name__)

# Available topics for text generation
TOPICS = ["fiction", "news", "science", "history", "daily_life", "culture"]

# Default topic weights (all equal)
DEFAULT_TOPIC_WEIGHTS = {t: 1.0 for t in TOPICS}


class PoolSelectionService:
    """
    Manages text pool and selection based on user state.
    
    Key concepts:
    - Pool: Pre-generated texts waiting to be read (pooled=True, opened_at=None)
    - Selection: Pick text closest to user's current preferences
    - Backfill: Generate new texts when pool is depleted
    """
    
    def get_pool(self, db: Session, account_id: int, lang: str) -> List[ReadingText]:
        """Get all pooled texts for a user/language."""
        return (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.pooled == True,
                ReadingText.content.isnot(None),  # Only fully generated texts
                ReadingText.opened_at.is_(None),  # Not yet opened
            )
            .all()
        )
    
    def get_pool_size(self, db: Session, account_id: int, lang: str) -> int:
        """Get count of pooled texts (including those still generating)."""
        return (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.pooled == True,
                ReadingText.opened_at.is_(None),
            )
            .count()
        )
    
    def select_from_pool(
        self,
        db: Session,
        account_id: int,
        lang: str,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """
        Select the best text from pool based on user's current state.
        Returns None if pool is empty.
        """
        pool = self.get_pool(db, account_id, lang)
        if not pool:
            return None
        
        # Get user preferences
        ci_pref = getattr(profile, 'ci_preference', None) or 0.92
        topic_weights = getattr(profile, 'topic_weights', None) or DEFAULT_TOPIC_WEIGHTS
        
        # Find text with minimum distance to user state
        best_text = min(pool, key=lambda t: self._distance(t, ci_pref, topic_weights))
        return best_text
    
    def _distance(
        self,
        text: ReadingText,
        ci_pref: float,
        topic_weights: dict,
    ) -> float:
        """
        Calculate distance between text and user preferences.
        Lower distance = better match.
        """
        # CI distance (weighted heavily)
        text_ci = getattr(text, 'ci_target', None) or 0.92
        d_ci = abs(text_ci - ci_pref) * 2.0
        
        # Topic distance
        text_topic = getattr(text, 'topic', None) or 'general'
        topic_weight = topic_weights.get(text_topic, 0.5)
        # Higher weight = lower distance (preferred topic)
        d_topic = 1.0 - min(topic_weight, 1.0) if topic_weight > 0 else 1.0
        
        return d_ci + d_topic * 0.5
    
    def get_generation_params(
        self,
        profile: Profile,
        vary: bool = True,
    ) -> Tuple[float, str]:
        """
        Get CI target and topic for generating a new pool text.
        
        Args:
            profile: User profile with preferences
            vary: If True, add variance to create diversity in pool
            
        Returns:
            (ci_target, topic)
        """
        # Base CI from profile
        ci_pref = getattr(profile, 'ci_preference', None) or 0.92
        topic_weights = getattr(profile, 'topic_weights', None) or DEFAULT_TOPIC_WEIGHTS
        
        if vary:
            # Add variance to CI target
            ci_target = ci_pref + random.uniform(-POOL_CI_VARIANCE, POOL_CI_VARIANCE)
            ci_target = max(0.80, min(0.98, ci_target))  # Clamp to valid range
        else:
            ci_target = ci_pref
        
        # Weighted random topic selection
        topic = self._weighted_random_topic(topic_weights)
        
        return ci_target, topic
    
    def _weighted_random_topic(self, weights: dict) -> str:
        """Select a topic based on weights."""
        # Filter out zero/negative weights
        valid = {k: v for k, v in weights.items() if v > 0}
        if not valid:
            return random.choice(TOPICS)
        
        topics = list(valid.keys())
        probs = list(valid.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        
        return random.choices(topics, weights=probs, k=1)[0]
    
    def needs_backfill(self, db: Session, account_id: int, lang: str) -> int:
        """
        Check how many texts need to be generated to fill pool.
        Returns count of texts needed (0 if pool is full).
        """
        current_size = self.get_pool_size(db, account_id, lang)
        return max(0, POOL_SIZE - current_size)
    
    def mark_selected(self, db: Session, text: ReadingText) -> None:
        """Mark a text as selected (no longer in pool)."""
        text.pooled = False
        db.flush()


# Singleton
_pool_service: Optional[PoolSelectionService] = None


def get_pool_selection_service() -> PoolSelectionService:
    global _pool_service
    if _pool_service is None:
        _pool_service = PoolSelectionService()
    return _pool_service
