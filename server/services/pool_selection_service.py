"""
Pool-based text selection service.
Manages global text pool and selection based on user state.
"""
from __future__ import annotations

import logging
import random
from typing import List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from ..config import POOL_SIZE, POOL_CI_VARIANCE, TOPICS, DEFAULT_TOPIC_WEIGHTS
from ..models import Profile, ReadingText, ProfileTextRead

logger = logging.getLogger(__name__)


class PoolSelectionService:
    """
    Manages global text pool and selection based on user state.
    
    Key concepts:
    - Global Pool: Ready texts (lang/target_lang) shared across users
    - Selection: Pick text closest to user's current preferences that user hasn't read
    - Backfill: Generate new texts when pool is depleted
    """
    
    def get_pool(
        self,
        global_db: Session,
        lang: str,
        target_lang: str,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[ReadingText]:
        """Get all ready texts from global pool for a language pair."""
        query = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
                ReadingText.content.isnot(None),
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
        )
        
        if exclude_ids:
            query = query.filter(~ReadingText.id.in_(exclude_ids))
        
        return query.all()
    
    def get_pool_size(
        self,
        global_db: Session,
        lang: str,
        target_lang: str,
    ) -> int:
        """Get count of ready texts in global pool for a language pair."""
        return (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == lang,
                ReadingText.target_lang == target_lang,
                ReadingText.content.isnot(None),
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
            .count()
        )
    
    def get_unread_text_ids(
        self,
        account_db: Session,
        profile_id: int,
    ) -> Set[int]:
        """Get IDs of texts the profile has read."""
        return set(
            r.text_id for r in account_db.query(ProfileTextRead.text_id)
            .filter(ProfileTextRead.profile_id == profile_id)
            .all()
        )
    
    def select_from_pool(
        self,
        global_db: Session,
        account_db: Session,
        lang: str,
        target_lang: str,
        profile: Profile,
    ) -> Optional[ReadingText]:
        """
        Select the best text from global pool based on user's current state.
        Excludes texts the user has already read.
        Returns None if no unread texts available.
        """
        # Get texts user has already read
        read_ids = self.get_unread_text_ids(account_db, profile.id)
        
        # Get available pool
        pool = self.get_pool(global_db, lang, target_lang, exclude_ids=read_ids)
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
        """
        ci_pref = getattr(profile, 'ci_preference', None) or 0.92
        topic_weights = getattr(profile, 'topic_weights', None) or DEFAULT_TOPIC_WEIGHTS
        
        if vary:
            ci_target = ci_pref + random.uniform(-POOL_CI_VARIANCE, POOL_CI_VARIANCE)
            ci_target = max(0.80, min(0.98, ci_target))
        else:
            ci_target = ci_pref
        
        topic = self._weighted_random_topic(topic_weights)
        
        return ci_target, topic
    
    def _weighted_random_topic(self, weights: dict) -> str:
        """Select 1-3 topics based on weights."""
        valid = {k: v for k, v in weights.items() if v > 0}
        if not valid:
            return random.choice(TOPICS)
        
        topics = list(valid.keys())
        probs = list(valid.values())
        total = sum(probs)
        probs = [p / total for p in probs]
        
        roll = random.random()
        if roll < 0.50:
            count = 1
        elif roll < 0.85:
            count = 2
        else:
            count = 3
        
        count = min(count, len(topics))
        
        selected = []
        remaining_topics = topics.copy()
        remaining_probs = probs.copy()
        
        for _ in range(count):
            if not remaining_topics:
                break
            total_prob = sum(remaining_probs)
            if total_prob <= 0:
                break
            norm_probs = [p / total_prob for p in remaining_probs]
            
            idx = random.choices(range(len(remaining_topics)), weights=norm_probs, k=1)[0]
            selected.append(remaining_topics[idx])
            
            remaining_topics.pop(idx)
            remaining_probs.pop(idx)
        
        return ",".join(selected)
    
    def needs_backfill(
        self,
        global_db: Session,
        account_db: Session,
        profile: Profile,
    ) -> int:
        """
        Check how many texts need to be generated for this profile.
        Returns count of texts needed (0 if enough unread texts exist).
        """
        read_ids = self.get_unread_text_ids(account_db, profile.id)
        
        available = (
            global_db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                ReadingText.content.isnot(None),
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
                ~ReadingText.id.in_(read_ids) if read_ids else True,
            )
            .count()
        )
        
        return max(0, POOL_SIZE - available)


# Singleton
_pool_service: Optional[PoolSelectionService] = None


def get_pool_selection_service() -> PoolSelectionService:
    global _pool_service
    if _pool_service is None:
        _pool_service = PoolSelectionService()
    return _pool_service
