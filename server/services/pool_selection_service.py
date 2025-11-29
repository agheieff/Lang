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

from ..config import POOL_SIZE, POOL_CI_VARIANCE, TOPICS, DEFAULT_TOPIC_WEIGHTS
from ..models import Profile, ReadingText

logger = logging.getLogger(__name__)


class PoolSelectionService:
    """
    Manages text pool and selection based on user state.
    
    Key concepts:
    - Pool: Pre-generated texts waiting to be read (pooled=True, opened_at=None)
    - Selection: Pick text closest to user's current preferences
    - Backfill: Generate new texts when pool is depleted
    """
    
    def get_pool(self, db: Session, account_id: int, lang: str) -> List[ReadingText]:
        """Get all ready pooled texts for a user/language."""
        return (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account_id,
                ReadingText.lang == lang,
                ReadingText.pooled == True,
                ReadingText.content.isnot(None),  # Has content
                ReadingText.opened_at.is_(None),  # Not yet opened
                ReadingText.words_complete == True,  # Words translated
                ReadingText.sentences_complete == True,  # Sentences translated
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
        
        # Topic distance - handle multiple comma-separated topics
        text_topic_str = getattr(text, 'topic', None) or 'general'
        text_topics = [t.strip() for t in text_topic_str.split(',') if t.strip()]
        
        if text_topics:
            # Average weight across all topics in the text
            weights_sum = sum(topic_weights.get(t, 0.5) for t in text_topics)
            avg_weight = weights_sum / len(text_topics)
        else:
            avg_weight = 0.5
        
        # Higher weight = lower distance (preferred topics)
        d_topic = 1.0 - min(avg_weight, 1.0) if avg_weight > 0 else 1.0
        
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
        """
        Select 1-3 topics based on weights.
        ~50% chance of 1 topic, ~35% chance of 2, ~15% chance of 3.
        Returns comma-separated string like "technology,science".
        """
        # Filter out zero/negative weights
        valid = {k: v for k, v in weights.items() if v > 0}
        if not valid:
            return random.choice(TOPICS)
        
        topics = list(valid.keys())
        probs = list(valid.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        
        # Decide how many topics: 50% → 1, 35% → 2, 15% → 3
        roll = random.random()
        if roll < 0.50:
            count = 1
        elif roll < 0.85:
            count = 2
        else:
            count = 3
        
        # Don't pick more topics than available
        count = min(count, len(topics))
        
        # Pick topics without replacement
        selected = []
        remaining_topics = topics.copy()
        remaining_probs = probs.copy()
        
        for _ in range(count):
            if not remaining_topics:
                break
            # Normalize remaining probs
            total_prob = sum(remaining_probs)
            if total_prob <= 0:
                break
            norm_probs = [p / total_prob for p in remaining_probs]
            
            # Pick one
            idx = random.choices(range(len(remaining_topics)), weights=norm_probs, k=1)[0]
            selected.append(remaining_topics[idx])
            
            # Remove from pool
            remaining_topics.pop(idx)
            remaining_probs.pop(idx)
        
        return ",".join(selected)
    
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
