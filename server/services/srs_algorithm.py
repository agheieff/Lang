"""Simple Spaced Repetition System (SRS) algorithm module.

This module is designed to be easily replaceable with other algorithms.
The interface is kept minimal and clean:

Input: Reading session data (word exposures, clicks, timestamps)
Output: SRS schedule parameters (stability, difficulty, next review time)

Usage:
    from server.services.srs_algorithm import update_srs_from_session

    # After user reads a text
    for interaction in session_interactions:
        srs_data = update_srs_from_session(
            lexeme=lexeme,
            was_clicked=interaction["clicked"],
            click_count=interaction["click_count"],
            was_viewed=interaction["translation_viewed"],
        )
        # Update lexeme with srs_data
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from server.models import Lexeme


@dataclass
class SRSUpdate:
    """Result of SRS algorithm update."""

    stability: float
    alpha: float
    beta: float
    difficulty: float
    familiarity: float
    importance: float
    next_due_at: datetime


class SimpleSRS:
    """Simple spaced repetition algorithm.

    Algorithm overview:
    - Stability: How stable a memory is (higher = more stable)
    - Difficulty: How hard a word is for the user (higher = harder)
    - Familiarity: Overall familiarity score (0-1)
    - Importance: Priority based on frequency and user interaction

    The algorithm increases stability when words are recognized without help,
    and decreases stability when words are looked up.
    """

    INITIAL_STABILITY = 1.0
    INITIAL_DIFFICULTY = 5.0
    INITIAL_FAMILIARITY = 0.0
    INITIAL_IMPORTANCE = 5.0
    INITIAL_ALPHA = 1.0
    INITIAL_BETA = 1.0

    MIN_STABILITY = 0.5
    MAX_STABILITY = 365.0  # days
    MIN_DIFFICULTY = 0.0
    MAX_DIFFICULTY = 10.0

    def update(
        self,
        lexeme: Lexeme,
        was_clicked: bool,
        click_count: int = 0,
        was_viewed: bool = False,
    ) -> SRSUpdate:
        """Update SRS parameters based on user interaction.

        Args:
            lexeme: Current lexeme state
            was_clicked: Whether user clicked the word for translation
            click_count: Number of times clicked in this session
            was_viewed: Whether user viewed the full translation

        Returns:
            SRSUpdate with new parameters
        """
        # Get current values or initialize
        stability = lexeme.stability or self.INITIAL_STABILITY
        alpha = lexeme.alpha or self.INITIAL_ALPHA
        beta = lexeme.beta or self.INITIAL_BETA
        difficulty = lexeme.difficulty or self.INITIAL_DIFFICULTY
        familiarity = lexeme.familiarity or self.INITIAL_FAMILIARITY
        importance = lexeme.importance or self.INITIAL_IMPORTANCE

        # Update based on interaction type
        if was_clicked or click_count > 0:
            # User needed help - word is harder than thought
            stability *= 0.8
            difficulty = min(self.MAX_DIFFICULTY, difficulty + 1.0)
            familiarity = max(0.0, familiarity - 0.1)
            alpha += click_count * 0.5
            beta += 0.1
        else:
            # User knew the word - memory is strengthening
            stability *= 1.2
            difficulty = max(self.MIN_DIFFICULTY, difficulty - 0.2)
            familiarity = min(1.0, familiarity + 0.05)
            alpha += 0.1
            beta += 1.0

        # Clamp values
        stability = max(self.MIN_STABILITY, min(self.MAX_STABILITY, stability))
        difficulty = max(self.MIN_DIFFICULTY, min(self.MAX_DIFFICULTY, difficulty))
        familiarity = max(0.0, min(1.0, familiarity))

        # Update importance based on exposures and recent activity
        exposures = (lexeme.exposures or 0) + 1
        clicks = (lexeme.clicks or 0) + click_count

        # Importance balances frequency (exposures) with difficulty
        # Words that appear often but are hard should be prioritized
        click_ratio = clicks / exposures if exposures > 0 else 0.5
        importance = (difficulty * 0.6) + (click_ratio * 10.0 * 0.4)

        # Calculate next review time
        next_due_at = self._calculate_next_due(stability)

        return SRSUpdate(
            stability=stability,
            alpha=alpha,
            beta=beta,
            difficulty=difficulty,
            familiarity=familiarity,
            importance=importance,
            next_due_at=next_due_at,
        )

    def _calculate_next_due(self, stability: float) -> datetime:
        """Calculate next review time based on stability.

        Higher stability = longer interval.
        Formula: hours = stability * 2.5
        """
        interval_hours = stability * 2.5
        next_due = datetime.now(timezone.utc) + timedelta(hours=interval_hours)
        return next_due


# Global instance for easy import
_srs = SimpleSRS()


def update_srs_from_session(
    lexeme: Lexeme,
    was_clicked: bool = False,
    click_count: int = 0,
    was_viewed: bool = False,
) -> SRSUpdate:
    """Update SRS parameters from a reading session interaction.

    This is the main entry point for the SRS module.
    Replace this function to switch to a different algorithm.

    Args:
        lexeme: Current lexeme state
        was_clicked: Whether user clicked the word for translation
        click_count: Number of times clicked in this session
        was_viewed: Whether user viewed the full translation

    Returns:
        SRSUpdate with new parameters to apply to lexeme
    """
    return _srs.update(
        lexeme=lexeme,
        was_clicked=was_clicked,
        click_count=click_count,
        was_viewed=was_viewed,
    )


def get_words_for_llm(
    db,
    profile_id: int,
    count: int = 20,
    min_difficulty: Optional[float] = None,
    max_difficulty: Optional[float] = None,
) -> list[dict]:
    """Get words to include in text generation prompts.

    This bridges SRS to LLM text generation.
    Returns words that need practice based on SRS schedule.

    Args:
        db: Database session
        profile_id: User's profile ID
        count: Maximum number of words to return
        min_difficulty: Optional minimum difficulty filter
        max_difficulty: Optional maximum difficulty filter

    Returns:
        List of word dictionaries with surface, lemma, pos, and priority
    """
    from server.models import Lexeme

    now = datetime.now(timezone.utc)

    query = db.query(Lexeme).filter(
        Lexeme.profile_id == profile_id,
    )

    # Filter by difficulty if specified
    if min_difficulty is not None:
        query = query.filter(Lexeme.difficulty >= min_difficulty)
    if max_difficulty is not None:
        query = query.filter(Lexeme.difficulty <= max_difficulty)

    # Prioritize by: due date first, then importance
    lexemes = (
        query.order_by(
            Lexeme.next_due_at.asc().nullsfirst(),
            Lexeme.importance.desc(),
        )
        .limit(count)
        .all()
    )

    # Convert to simple format for LLM
    words = []
    for lex in lexemes:
        words.append(
            {
                "surface": lex.surface or "",
                "lemma": lex.lemma or "",
                "pos": lex.pos or "",
                "priority": lex.importance or 0.0,
            }
        )

    return words
