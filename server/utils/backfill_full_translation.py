"""Backfill full_translation for existing text states.

Run this script to add the full_translation field to existing text states
that were created before this feature was added.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm.attributes import flag_modified
from server.db import SessionLocal
from server.models import TextState, ReadingTextTranslation


def backfill_full_translation():
    """Add full_translation to all text states that don't have it."""
    db = SessionLocal()

    try:
        # Get all text states
        all_states = db.query(TextState).all()

        print(f"Found {len(all_states)} text states")

        updated_count = 0
        for state in all_states:
            state_data = state.state_data or {}

            # Check if full_translation already exists
            if 'full_translation' in state_data:
                continue

            # Get translations for this text
            translations = db.query(ReadingTextTranslation).filter(
                ReadingTextTranslation.text_id == state.text_id
            ).order_by(ReadingTextTranslation.segment_index).all()

            if translations:
                # Build full translation by joining all sentence translations
                full_translation = " ".join([
                    t.translated_text for t in translations if t.translated_text
                ])

                # Update state_data
                state_data['full_translation'] = full_translation
                state.state_data = state_data

                # Flag the JSON field as modified
                flag_modified(state, "state_data")

                updated_count += 1
                print(f"✓ Updated text_id {state.text_id} with {len(translations)} translations")
            else:
                print(f"⚠ No translations found for text_id {state.text_id}")

        db.commit()
        print(f"\n✓ Backfilled {updated_count} text states")

    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    backfill_full_translation()
