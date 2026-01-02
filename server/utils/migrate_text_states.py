"""
Migrate older texts to use the new text state system.

This script finds all texts that have word glosses but no text state,
and generates the text states retroactively.
"""

import sys
from pathlib import Path

# Add server directory to path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))

from server.db import SessionLocal
from server.models import (
    ReadingText,
    ReadingWordGloss,
    ReadingTextTranslation,
    TextState,
)
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


def migrate_text_to_state(db, text: ReadingText) -> TextState:
    """Create text state for an existing text."""

    # Check if state already exists
    existing = db.query(TextState).filter(TextState.text_id == text.id).first()
    if existing:
        print(f"  Text {text.id}: State already exists, skipping")
        return existing

    # Build initial state
    state_data = {
        "text_id": text.id,
        "lang": text.lang,
        "target_lang": text.target_lang,
        "title": text.title,
        "content": text.content,
        "topic": text.topic,
        "difficulty_estimate": text.difficulty_estimate,
        "ci_target": text.ci_target,
        "generated_at": text.generated_at.isoformat() if text.generated_at else None,
        "words": [],
        "sentence_translations": [],
    }

    # Add word glosses
    word_glosses = (
        db.query(ReadingWordGloss)
        .filter(ReadingWordGloss.text_id == text.id)
        .order_by(ReadingWordGloss.span_start)
        .all()
    )

    for g in word_glosses:
        state_data["words"].append(
            {
                "surface": g.surface,
                "lemma": g.lemma,
                "pos": g.pos,
                "pinyin": g.pinyin,
                "translation": g.translation,
                "lemma_translation": g.lemma_translation,
                "grammar": g.grammar or {},
                "span_start": g.span_start,
                "span_end": g.span_end,
            }
        )

    # Add sentence translations
    translations = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text.id,
            ReadingTextTranslation.unit == "sentence",
        )
        .order_by(ReadingTextTranslation.segment_index)
        .all()
    )

    for t in translations:
        state_data["sentence_translations"].append(
            {
                "index": t.segment_index,
                "source": t.source_text,
                "translation": t.translated_text,
            }
        )

    # Build paragraphs structure from content and translations
    paragraphs = _build_paragraph_structure(text.content, translations)
    state_data["paragraphs"] = paragraphs
    state_data["paragraph_count"] = len(paragraphs)

    # Determine status
    has_words = len(word_glosses) > 0
    has_translations = len(translations) > 0
    status = "ready" if (has_words and has_translations) else "building"

    state_data["word_count"] = len(state_data["words"])
    state_data["sentence_count"] = len(state_data["sentence_translations"])

    # Create state record
    state = TextState(
        text_id=text.id,
        status=status,
        has_content=True,
        has_words=has_words,
        has_translations=has_translations,
        state_data=state_data,
    )

    db.add(state)
    db.commit()

    print(
        f"  Text {text.id}: Created state (words={len(word_glosses)}, translations={len(translations)}, status={status})"
    )

    return state


def main():
    db = SessionLocal()

    try:
        # Find all texts with word glosses but no text state
        texts_with_glosses = (
            db.query(ReadingText)
            .join(ReadingWordGloss, ReadingText.id == ReadingWordGloss.text_id)
            .outerjoin(TextState, ReadingText.id == TextState.text_id)
            .filter(TextState.id.is_(None))
            .distinct()
            .all()
        )

        print(f"Found {len(texts_with_glosses)} texts needing migration")

        if not texts_with_glosses:
            print("No texts need migration - all up to date!")
            return

        # Migrate each text
        for i, text in enumerate(texts_with_glosses, 1):
            print(f"\n[{i}/{len(texts_with_glosses)}] Migrating text {text.id}...")
            migrate_text_to_state(db, text)

        print(f"\n✅ Migration complete! Processed {len(texts_with_glosses)} texts")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
