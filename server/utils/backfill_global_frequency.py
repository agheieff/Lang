"""Backfill global word frequency from existing TextVocabulary data.

Run this script to populate the GlobalWordFrequency table with aggregated
data from existing TextVocabulary entries.
"""
from server.db import SessionLocal
from server.models import TextVocabulary, GlobalWordFrequency, ReadingText
from sqlalchemy import func


def backfill_global_frequency():
    """Populate GlobalWordFrequency from existing text vocabulary."""
    with SessionLocal() as db:
        # Get all words from TextVocabulary, joining with ReadingText for lang
        word_stats = db.query(
            ReadingText.lang,
            TextVocabulary.lemma,
            TextVocabulary.pos,
            func.count(TextVocabulary.text_id).label('text_count'),
            func.sum(TextVocabulary.occurrence_count).label('total_occurrences')
        ).join(
            ReadingText, TextVocabulary.text_id == ReadingText.id
        ).group_by(
            ReadingText.lang,
            TextVocabulary.lemma,
            TextVocabulary.pos
        ).all()

        print(f"Found {len(word_stats)} unique words across all texts")

        # Group by language for percentile calculation
        lang_words = {}
        for word in word_stats:
            if word.lang not in lang_words:
                lang_words[word.lang] = []
            lang_words[word.lang].append(word)

        # Process each language
        total_added = 0
        for lang, words in lang_words.items():
            # Sort by text_count (descending) - most common first
            words.sort(key=lambda x: x.text_count or 0, reverse=True)
            total = len(words)

            print(f"\nProcessing {lang}: {total} words")

            # Create percentile mapping and insert records
            for i, word in enumerate(words):
                # Percentile: 1.0 = most common, 0.0 = least common
                percentile = (total - i) / total if total > 0 else 0.0

                # Check if already exists
                existing = db.query(GlobalWordFrequency).filter(
                    GlobalWordFrequency.lang == word.lang,
                    GlobalWordFrequency.lemma == word.lemma,
                    GlobalWordFrequency.pos == word.pos
                ).first()

                if existing:
                    # Update existing record
                    existing.text_count = word.text_count or 0
                    existing.total_occurrences = word.total_occurrences or 0
                    existing.frequency_percentile = percentile
                else:
                    # Create new record
                    gwf = GlobalWordFrequency(
                        lang=word.lang,
                        lemma=word.lemma,
                        pos=word.pos,
                        text_count=word.text_count or 0,
                        total_occurrences=word.total_occurrences or 0,
                        frequency_percentile=percentile
                    )
                    db.add(gwf)
                    total_added += 1

        db.commit()
        print(f"\n✓ Backfill complete: {total_added} words added/updated")


if __name__ == "__main__":
    backfill_global_frequency()
