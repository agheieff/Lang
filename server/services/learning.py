"""
Learning services consolidating SRS, level, and lexeme functionality.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session
from server.models import (
    Profile,
    Lexeme,
    LexemeVariant,
    WordEvent,
    UserLexemeContext,
    Card,
    TextVocabulary,
    ProfileTextRead,
    Account,
    GenerationLog,
)

logger = logging.getLogger(__name__)


# Level Service
class LevelService:
    """Manages user language level estimation and tracking."""
    
    def get_ci_target(
        self,
        level_value: float,
        level_var: float = 1.0,
    ) -> float:
        """Get target comprehension index based on user level."""
        # CI targets range from 0.85 (challenging) to 0.98 (easy)
        # Lower levels get easier texts
        if level_value < 2.0:
            return 0.95  # Very easy for beginners
        elif level_value < 4.0:
            return 0.92  # Easy for elementary
        elif level_value < 6.0:
            return 0.88  # Moderate for intermediate
        elif level_value < 8.0:
            return 0.86  # Challenging for advanced
        else:
            return 0.85  # Most challenging for expert
    
    def update_level_from_text(
        self,
        account_db: Session,
        profile: Profile,
        text_id: int,
        interactions: List[Dict],
    ) -> Tuple[float, float]:
        """Update user level based on text interactions."""
        try:
            # Calculate interaction metrics
            total_words = len(interactions)
            if total_words == 0:
                return profile.level_value, profile.level_var
            
            clicked_words = sum(1 for i in interactions if i.get("clicked", False))
            click_rate = clicked_words / total_words
            
            # Get text difficulty estimate
            # TODO: Implement text difficulty calculation based on known vocabulary
            
            # Update level using Bayesian adaptation
            current_level = profile.level_value
            current_var = profile.level_var
            
            # Simplified Bayesian update
            # Higher click rate suggests text is too hard -> decrease level
            # Lower click rate suggests text is too easy -> increase level
            if click_rate > 0.3:  # Too many clicks, decrease level
                level_delta = -0.1 * (click_rate - 0.3)
            else:  # Good performance, increase level
                level_delta = 0.05 * (0.3 - click_rate)
            
            new_level = max(0.0, min(10.0, current_level + level_delta))
            new_var = max(0.1, current_var * 0.95)  # Decrease uncertainty
            
            # Update profile
            profile.level_value = new_level
            profile.level_var = new_var
            
            account_db.commit()
            
            logger.info(f"Updated level for profile {profile.id}: {current_level}->{new_level}")
            return new_level, new_var
            
        except Exception as e:
            logger.error(f"Error updating level: {e}")
            return profile.level_value, profile.level_var


# Lexeme Service
class LexemeService:
    """Manages user-specific vocabulary and lexeme tracking."""
    
    def track_word_exposure(
        self,
        account_db: Session,
        account_id: int,
        profile_id: int,
        text_id: int,
        word_data: List[Dict],
    ) -> None:
        """Track word exposure events from reading."""
        try:
            for word_info in word_data:
                surface = word_info.get("surface", "")
                lemma = word_info.get("lemma", surface)
                pos = word_info.get("pos", "NOUN")
                span_start = word_info.get("span_start", 0)
                span_end = word_info.get("span_end", 0)
                
                # Get or create lexeme
                lexeme = account_db.query(Lexeme).filter(
                    Lexeme.account_id == account_id,
                    Lexeme.profile_id == profile_id,
                    Lexeme.lang == word_info.get("lang", ""),
                    Lexeme.lemma == lemma,
                    Lexeme.pos == pos,
                ).first()
                
                if not lexeme:
                    lexeme = Lexeme(
                        account_id=account_id,
                        profile_id=profile_id,
                        lang=word_info.get("lang", ""),
                        lemma=lemma,
                        pos=pos,
                        surface=surface,
                        first_seen_at=datetime.now(timezone.utc),
                        last_seen_at=datetime.now(timezone.utc),
                        exposures=0,
                        clicks=0,
                        distinct_texts=0,
                    )
                    account_db.add(lexeme)
                
                # Update exposure counts
                lexeme.exposures += 1
                lexeme.last_seen_at = datetime.now(timezone.utc)
                lexeme.distinct_texts += 1
                
                # Update SRS parameters
                self._update_srs_parameters(lexeme)
                
                # Create word event
                event = WordEvent(
                    account_id=account_id,
                    profile_id=profile_id,
                    event_type="exposure",
                    surface=surface,
                    span_start=span_start,
                    span_end=span_end,
                    text_id=text_id,
                    meta={"lemma": lemma, "pos": pos},
                )
                account_db.add(event)
            
            account_db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking word exposure: {e}")
            account_db.rollback()
    
    def track_word_click(
        self,
        account_db: Session,
        account_id: int,
        profile_id: int,
        text_id: int,
        word_info: Dict,
    ) -> None:
        """Track word click/lookup events."""
        try:
            surface = word_info.get("surface", "")
            lemma = word_info.get("lemma", surface)
            pos = word_info.get("pos", "NOUN")
            span_start = word_info.get("span_start", 0)
            span_end = word_info.get("span_end", 0)
            
            # Get lexeme
            lexeme = account_db.query(Lexeme).filter(
                Lexeme.account_id == account_id,
                Lexeme.profile_id == profile_id,
                Lexeme.lang == word_info.get("lang", ""),
                Lexeme.lemma == lemma,
                Lexeme.pos == pos,
            ).first()
            
            if not lexeme:
                # Create new lexeme for clicked word
                lexeme = Lexeme(
                    account_id=account_id,
                    profile_id=profile_id,
                    lang=word_info.get("lang", ""),
                    lemma=lemma,
                    pos=pos,
                    surface=surface,
                    first_seen_at=datetime.now(timezone.utc),
                    last_seen_at=datetime.now(timezone.utc),
                    exposures=1,
                    clicks=1,
                    distinct_texts=1,
                    last_clicked_at=datetime.now(timezone.utc),
                )
                account_db.add(lexeme)
            else:
                # Update existing lexeme
                lexeme.clicks += 1
                lexeme.last_clicked_at = datetime.now(timezone.utc)
                
                # Update SRS parameters
                self._update_srs_parameters(lexeme, clicked=True)
            
            # Create click event
            event = WordEvent(
                account_id=account_id,
                profile_id=profile_id,
                event_type="click",
                surface=surface,
                span_start=span_start,
                span_end=span_end,
                text_id=text_id,
                meta={"lemma": lemma, "pos": pos},
            )
            account_db.add(event)
            
            account_db.commit()
            
        except Exception as e:
            logger.error(f"Error tracking word click: {e}")
            account_db.rollback()
    
    def _update_srs_parameters(
        self,
        lexeme: Lexeme,
        clicked: bool = False,
    ) -> None:
        """Update SRS parameters using Bayesian model."""
        try:
            # Current parameters
            alpha = lexeme.alpha
            beta = lexeme.beta
            clicks = lexeme.clicks
            exposures = lexeme.exposures
            
            # Update Bayesian posterior
            if clicked:
                # Click increases belief in future clicks
                alpha += 1.0
                beta += 0.1
            else:
                # Exposure without click suggests knowledge
                alpha += 0.1
                beta += 1.0
            
            # Update difficulty (lower = easier)
            click_rate = clicks / max(exposures, 1)
            difficulty = max(0.1, 1.0 - click_rate)
            
            # Calculate familiarity
            familiarity = 1.0 - (alpha / (alpha + beta))
            
            # Calculate next due date
            # More familiar words have longer intervals
            interval_days = max(1, int(7 * familiarity))
            next_due = datetime.now(timezone.utc) + timedelta(days=interval_days)
            
            # Update lexeme
            lexeme.alpha = alpha
            lexeme.beta = beta
            lexeme.difficulty = difficulty
            lexeme.familiarity = familiarity
            lexeme.next_due_at = next_due
            lexeme.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Error updating SRS parameters: {e}")
    
    def get_due_lexemes(
        self,
        account_db: Session,
        profile_id: int,
        limit: int = 20,
    ) -> List[Lexeme]:
        """Get lexemes due for SRS review."""
        now = datetime.now(timezone.utc)
        
        return account_db.query(Lexeme).filter(
            Lexeme.profile_id == profile_id,
            Lexeme.next_due_at <= now,
            Lexeme.is_active == True,
        ).order_by(
            Lexeme.next_due_at.asc(),
        ).limit(limit).all()
    
    def add_lexeme_variant(
        self,
        account_db: Session,
        lexeme_id: int,
        script: str,
        form: str,
    ) -> LexemeVariant:
        """Add a variant form for a lexeme (e.g., simplified/traditional Chinese)."""
        try:
            variant = LexemeVariant(
                lexeme_id=lexeme_id,
                script=script,
                form=form,
                created_at=datetime.now(timezone.utc),
            )
            account_db.add(variant)
            account_db.commit()
            account_db.refresh(variant)
            
            return variant
            
        except Exception as e:
            logger.error(f"Error adding lexeme variant: {e}")
            account_db.rollback()
            raise
    
    def get_lexemes_by_importance(
        self,
        account_db: Session,
        profile_id: int,
        limit: int = 100,
    ) -> List[Lexeme]:
        """Get lexemes sorted by importance and difficulty."""
        return account_db.query(Lexeme).filter(
            Lexeme.profile_id == profile_id,
            Lexeme.is_active == True,
        ).order_by(
            Lexeme.importance.desc(),
            Lexeme.difficulty.asc(),
            Lexeme.familiarity.asc(),
        ).limit(limit).all()


# SRS Service
class SRSService:
    """Spaced Repetition System for vocabulary review."""
    
    def __init__(self):
        self.lexeme_service = LexemeService()
    
    def get_srs_cards(
        self,
        account_db: Session,
        profile_id: int,
        limit: int = 10,
    ) -> List[Dict]:
        """Get flashcards for SRS review."""
        try:
            # Get due lexemes
            due_lexemes = self.lexeme_service.get_due_lexemes(account_db, profile_id, limit)
            
            cards = []
            for lexeme in due_lexemes:
                # Get variants if available
                variants = account_db.query(LexemeVariant).filter(
                    LexemeVariant.lexeme_id == lexeme.id,
                ).all()
                
                card_data = {
                    "id": lexeme.id,
                    "surface": lexeme.surface,
                    "lemma": lexeme.lemma,
                    "pos": lexeme.pos,
                    "difficulty": lexeme.difficulty,
                    "familiarity": lexeme.familiarity,
                    "variants": [
                        {"script": v.script, "form": v.form}
                        for v in variants
                    ],
                    "exposures": lexeme.exposures,
                    "clicks": lexeme.clicks,
                }
                cards.append(card_data)
            
            # Sort by difficulty and familiarity
            cards.sort(key=lambda x: (
                x["difficulty"],  # Easier first
                x["familiarity"],  # Less familiar first
            ))
            
            return cards
            
        except Exception as e:
            logger.error(f"Error getting SRS cards: {e}")
            return []
    
    def submit_srs_review(
        self,
        account_db: Session,
        lexeme_id: int,
        rating: int,  # 1-5 rating (1=hard, 5=easy)
    ) -> bool:
        """Submit SRS review and update lexeme."""
        try:
            lexeme = account_db.query(Lexeme).filter(
                Lexeme.id == lexeme_id,
            ).first()
            
            if not lexeme:
                return False
            
            # Update based on rating
            if rating <= 2:  # Hard review
                lexeme.stability = max(0.1, lexeme.stability * 0.8)
                lexeme.difficulty = min(2.0, lexeme.difficulty * 1.1)
            elif rating >= 4:  # Easy review
                lexeme.stability = min(1.0, lexeme.stability * 1.2)
                lexeme.difficulty = max(0.1, lexeme.difficulty * 0.9)
            else:  # Medium review
                lexeme.stability = max(0.1, min(1.0, lexeme.stability * 1.05))
            
            # Calculate next interval using SM-2 algorithm (simplified)
            if rating <= 2:
                # Reset interval for difficult cards
                next_interval = 1
            else:
                # Increase interval based on stability
                next_interval = max(1, int(7 * lexeme.stability))
            
            lexeme.next_due_at = datetime.now(timezone.utc) + timedelta(days=next_interval)
            lexeme.updated_at = datetime.now(timezone.utc)
            
            account_db.commit()
            
            logger.info(f"Updated SRS for lexeme {lexeme_id} with rating {rating}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting SRS review: {e}")
            account_db.rollback()
            return False
    
    def get_review_stats(
        self,
        account_db: Session,
        profile_id: int,
    ) -> Dict:
        """Get SRS statistics for a profile."""
        try:
            total_lexemes = account_db.query(Lexeme).filter(
                Lexeme.profile_id == profile_id,
            ).count()
            
            due_lexemes = account_db.query(Lexeme).filter(
                Lexeme.profile_id == profile_id,
                Lexeme.next_due_at <= datetime.now(timezone.utc),
            ).count()
            
            new_lexemes = account_db.query(Lexeme).filter(
                Lexeme.profile_id == profile_id,
                Lexeme.clicks == 0,
            ).count()
            
            learning_lexemes = account_db.query(Lexeme).filter(
                Lexeme.profile_id == profile_id,
                Lexeme.familiarity < 0.5,
            ).count()
            
            mature_lexemes = account_db.query(Lexeme).filter(
                Lexeme.profile_id == profile_id,
                Lexeme.familiarity >= 0.8,
            ).count()
            
            return {
                "total": total_lexemes,
                "due": due_lexemes,
                "new": new_lexemes,
                "learning": learning_lexemes,
                "mature": mature_lexemes,
                "due_percentage": (due_lexemes / max(total_lexemes, 1)) * 100,
            }
            
        except Exception as e:
            logger.error(f"Error getting review stats: {e}")
            return {}


# Service instances
_level_service = None
_lexeme_service = None
_srs_service = None


def get_level_service() -> LevelService:
    """Get the level service instance."""
    global _level_service
    if _level_service is None:
        _level_service = LevelService()
    return _level_service


def get_lexeme_service() -> LexemeService:
    """Get the lexeme service instance."""
    global _lexeme_service
    if _lexeme_service is None:
        _lexeme_service = LexemeService()
    return _lexeme_service


def get_srs_service() -> SRSService:
    """Get the SRS service instance."""
    global _srs_service
    if _srs_service is None:
        _srs_service = SRSService()
    return _srs_service
