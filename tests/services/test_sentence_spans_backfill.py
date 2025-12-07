from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.db import Base
from server.services.translation_service import TranslationService
from server.models import ReadingText, ReadingTextTranslation


def test_backfill_sentence_spans_updates_missing_rows():
    """Test that backfill_sentence_spans updates translations without spans (global model)."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = SessionLocal()
    try:
        # Global model - no account_id
        text = ReadingText(lang="en", target_lang="es", content="Hello world.")
        db.add(text)
        db.flush()

        # Insert a sentence translation without spans (global model)
        rtt = ReadingTextTranslation(
            text_id=text.id,
            target_lang="es",
            unit="sentence",
            segment_index=0,
            span_start=None,
            span_end=None,
            source_text="Hello world.",
            translated_text="Hola mundo.",
            provider="test",
            model="test-model",
        )
        db.add(rtt)
        db.commit()

        # Backfill spans
        svc = TranslationService()
        ok = svc.backfill_sentence_spans(db, text.id, target_lang="es")
        assert ok is True

        # Verify row now has spans
        row = (
            db.query(ReadingTextTranslation)
            .filter(
                ReadingTextTranslation.text_id == text.id,
                ReadingTextTranslation.target_lang == "es",
                ReadingTextTranslation.unit == "sentence"
            )
            .first()
        )
        assert row is not None
        assert isinstance(row.span_start, int)
        assert isinstance(row.span_end, int)
        assert 0 <= row.span_start <= row.span_end <= len("Hello world.")
    finally:
        db.close()
