"""
Text State Builder - builds complete text state JSON incrementally during text generation.

The state is built in stages:
1. Initial creation (when ReadingText is created)
2. Words added (when word glosses are generated)
3. Translations added (when sentence translations are generated)
4. Marked ready (when all components are complete)
"""

from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from server.models import (
    ReadingText,
    ReadingWordGloss,
    ReadingTextTranslation,
    TextState,
)


def create_text_state(db: Session, text: ReadingText) -> TextState:
    """Create initial text state when text is first generated."""
    state = TextState(
        text_id=text.id,
        status="building",
        has_content=True,
        has_words=False,
        has_translations=False,
        state_data={
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
        },
    )
    db.add(state)
    db.commit()
    return state


def add_words_to_state(db: Session, text_id: int) -> TextState:
    """Add word glosses to text state."""
    state = db.query(TextState).filter(TextState.text_id == text_id).first()
    if not state:
        return None

    # Get all word glosses for this text
    word_glosses = (
        db.query(ReadingWordGloss)
        .filter(ReadingWordGloss.text_id == text_id)
        .all()
    )

    words = [
        {
            "surface": g.surface,
            "lemma": g.lemma,
            "pos": g.pos,
            "pinyin": g.pinyin,
            "translation": g.translation,
            "lemma_translation": g.lemma_translation,
            "grammar": g.grammar,
            "span_start": g.span_start,
            "span_end": g.span_end,
        }
        for g in word_glosses
    ]

    state.state_data["words"] = words
    state.has_words = True
    state.state_data["word_count"] = len(words)

    # Flag the JSON field as modified so SQLAlchemy persists it
    flag_modified(state, "state_data")

    # Check if complete
    _check_and_mark_ready(db, state)

    db.commit()
    return state


def add_translations_to_state(db: Session, text_id: int) -> TextState:
    """Add sentence translations to text state, grouped by paragraphs."""
    state = db.query(TextState).filter(TextState.text_id == text_id).first()
    if not state:
        return None

    # Get all sentence translations for this text
    translations = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.unit == "sentence",
        )
        .order_by(ReadingTextTranslation.segment_index)
        .all()
    )

    # Group sentences into paragraphs
    content = state.state_data.get("content", "")
    paragraphs = _split_into_paragraphs(content)

    # Assign each sentence to its paragraph
    paragraph_data = []
    current_para_idx = 0

    for t in translations:
        # Find which paragraph this sentence belongs to
        source_text = t.source_text

        # Move to next paragraph if needed
        while current_para_idx < len(paragraphs) - 1:
            # Check if this sentence appears after the current paragraph
            current_para_end = content.find(paragraphs[current_para_idx]) + len(paragraphs[current_para_idx])
            next_para_start = content.find(paragraphs[current_para_idx + 1])

            # If sentence appears in next paragraph, move to it
            if source_text in content and content.find(source_text) >= next_para_start:
                current_para_idx += 1
            else:
                break

        # Add sentence to current paragraph
        para_text = paragraphs[current_para_idx] if current_para_idx < len(paragraphs) else ""

        paragraph_data.append({
            "paragraph_index": current_para_idx,
            "paragraph_text": para_text,
            "sentence_index": t.segment_index,
            "source": t.source_text,
            "translation": t.translated_text,
        })

    # Now group by paragraph
    paragraphs_with_sentences = []
    for para_idx, para_text in enumerate(paragraphs):
        sentences = [
            {
                "sentence_index": s["sentence_index"],
                "source": s["source"],
                "translation": s["translation"],
            }
            for s in paragraph_data
            if s["paragraph_index"] == para_idx
        ]

        paragraphs_with_sentences.append({
            "paragraph_index": para_idx,
            "content": para_text,
            "sentences": sentences,
        })

    state.state_data["paragraphs"] = paragraphs_with_sentences
    state.has_translations = True
    state.state_data["sentence_count"] = len(translations)
    state.state_data["paragraph_count"] = len(paragraphs_with_sentences)

    # Flag the JSON field as modified so SQLAlchemy persists it
    flag_modified(state, "state_data")

    # Check if complete
    _check_and_mark_ready(db, state)

    db.commit()
    return state


def _split_into_paragraphs(content: str) -> list[str]:
    """Split text content into paragraphs."""
    # Split by newline and filter empty paragraphs
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    return paragraphs


def _check_and_mark_ready(db: Session, state: TextState):
    """Mark state as ready if all components are present."""
    if state.has_content and state.has_words and state.has_translations:
        state.status = "ready"
        state.completed_at = datetime.now(timezone.utc)
        state.state_data["status"] = "ready"
        state.state_data["completed_at"] = state.completed_at.isoformat()
        # Flag the JSON field as modified
        flag_modified(state, "state_data")


def get_text_state(db: Session, text_id: int) -> dict:
    """Get the complete text state for serving to clients."""
    state = db.query(TextState).filter(TextState.text_id == text_id).first()
    if not state:
        return None

    return state.state_data


def is_text_state_ready(db: Session, text_id: int) -> bool:
    """Check if text state is ready to be served."""
    state = db.query(TextState).filter(TextState.text_id == text_id).first()
    return state and state.status == "ready"
