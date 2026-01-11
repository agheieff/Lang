"""Global word frequency tracking service."""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from server.models import GlobalWordFrequency

logger = logging.getLogger(__name__)


def get_word_frequency(
    db: Session,
    lang: str,
    lemma: str,
    pos: str,
) -> float:
    """Get frequency percentile for a word (0-1).

    Returns 0.5 (middle) if word not found.
    """
    try:
        gwf = db.query(GlobalWordFrequency).filter(
            GlobalWordFrequency.lang == lang,
            GlobalWordFrequency.lemma == lemma,
            GlobalWordFrequency.pos == pos
        ).first()

        return gwf.frequency_percentile if gwf else 0.5
    except Exception as e:
        logger.error(f"Error getting word frequency: {e}")
        return 0.5


def update_frequency_from_text(
    db: Session,
    lang: str,
    text_vocab: List[Dict],
) -> None:
    """Update global frequency after text vocabulary is added.

    Called from text_state_builder after word extraction.

    Args:
        db: Database session
        lang: Language code
        text_vocab: List of word dicts with keys: lemma, pos, occurrences
    """
    try:
        for word_info in text_vocab:
            lemma = word_info.get('lemma')
            pos = word_info.get('pos', 'NOUN')

            if not lemma:
                continue

            # Update or create frequency record using merge
            # merge will insert if not exists, or update if exists
            gwf = db.query(GlobalWordFrequency).filter(
                GlobalWordFrequency.lang == lang,
                GlobalWordFrequency.lemma == lemma,
                GlobalWordFrequency.pos == pos
            ).first()

            if gwf:
                gwf.text_count += 1
                gwf.total_occurrences += word_info.get('occurrences', 1)
                gwf.updated_at = datetime.now(timezone.utc)
            else:
                # Create new record - handle race conditions with try/except
                new_gwf = GlobalWordFrequency(
                    lang=lang,
                    lemma=lemma,
                    pos=pos,
                    text_count=1,
                    total_occurrences=word_info.get('occurrences', 1),
                    frequency_percentile=0.0
                )
                try:
                    db.add(new_gwf)
                    db.flush()  # Try to insert immediately
                except Exception:
                    # Race condition: another thread inserted it first
                    db.rollback()
                    # Query the existing record and update it
                    gwf = db.query(GlobalWordFrequency).filter(
                        GlobalWordFrequency.lang == lang,
                        GlobalWordFrequency.lemma == lemma,
                        GlobalWordFrequency.pos == pos
                    ).first()
                    if gwf:
                        gwf.text_count += 1
                        gwf.total_occurrences += word_info.get('occurrences', 1)
                        gwf.updated_at = datetime.now(timezone.utc)

        db.commit()

        # Recalculate percentiles periodically
        _recalculate_percentiles(db, lang)

    except Exception as e:
        logger.error(f"Error updating frequency: {e}")
        db.rollback()


def _recalculate_percentiles(
    db: Session,
    lang: str,
) -> None:
    """Recalculate frequency percentiles for a language.

    Percentiles are based on text_count (number of texts containing the word).
    Higher text_count = higher percentile (more common).
    """
    try:
        all_words = db.query(GlobalWordFrequency).filter(
            GlobalWordFrequency.lang == lang
        ).order_by(GlobalWordFrequency.text_count.desc()).all()

        total = len(all_words)
        if total == 0:
            return

        for i, word in enumerate(all_words):
            word.frequency_percentile = (total - i) / total

        db.commit()
        logger.debug(f"Recalculated percentiles for {lang}: {total} words")
    except Exception as e:
        logger.error(f"Error recalculating percentiles: {e}")
        db.rollback()
