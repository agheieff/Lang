"""
Add paragraphs field to existing text states that are missing it.
"""

import sys
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

from server.db import SessionLocal
from server.models import TextState, ReadingTextTranslation
from sqlalchemy.orm.attributes import flag_modified


def _build_paragraph_structure(content: str, translations: list) -> list[dict]:
    """Build hierarchical paragraph structure from content and translations."""

    # Split content into paragraphs
    paragraphs_raw = [p.strip() for p in content.split('\n') if p.strip()]

    # Assign translations to paragraphs based on position
    paragraph_data = []

    for para_idx, para_text in enumerate(paragraphs_raw):
        # Find all sentences that belong to this paragraph
        sentences = []
        para_start_pos = content.find(para_text)

        for t in translations:
            # Check if this sentence appears in this paragraph
            source_text = t.source_text
            sentence_pos = content.find(source_text)

            # Check if sentence position is within paragraph bounds
            if para_start_pos >= 0 and sentence_pos >= 0:
                para_end_pos = para_start_pos + len(para_text)
                if para_start_pos <= sentence_pos < para_end_pos:
                    sentences.append({
                        "sentence_index": t.segment_index,
                        "source": t.source_text,
                        "translation": t.translated_text,
                    })

        paragraph_data.append({
            "paragraph_index": para_idx,
            "content": para_text,
            "sentences": sentences,
        })

    return paragraph_data


def main():
    db = SessionLocal()

    try:
        # Find all states, then filter in Python
        all_states = db.query(TextState).all()
        states = [s for s in all_states if not s.state_data.get('paragraphs')]

        print(f"Found {len(states)} states missing paragraphs")

        if not states:
            print("No states need updating - all have paragraphs!")
            return

        for i, state in enumerate(states, 1):
            print(f"\n[{i}/{len(states)}] Updating text {state.text_id}...")

            content = state.state_data.get('content', '')
            if not content:
                print(f"  Text {state.text_id}: No content, skipping")
                continue

            # Get translations for this text
            translations = (
                db.query(ReadingTextTranslation)
                .filter(
                    ReadingTextTranslation.text_id == state.text_id,
                    ReadingTextTranslation.unit == "sentence",
                )
                .order_by(ReadingTextTranslation.segment_index)
                .all()
            )

            # Build paragraphs
            paragraphs = _build_paragraph_structure(content, translations)

            # Update state
            state.state_data['paragraphs'] = paragraphs
            state.state_data['paragraph_count'] = len(paragraphs)
            flag_modified(state, 'state_data')

            db.commit()

            print(f"  Text {state.text_id}: Added {len(paragraphs)} paragraphs with {len(translations)} sentences")

        print(f"\n✅ Update complete! Processed {len(states)} states")

    except Exception as e:
        print(f"\n❌ Update failed: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
