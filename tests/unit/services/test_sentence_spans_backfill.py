from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.db import Base
from server.services.translation_service import TranslationService
from server.models import ReadingText, ReadingTextTranslation


def test_backfill_sentence_spans_updates_missing_rows():
    # Local in-memory DB for this test
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db = SessionLocal()
    try:
        account_id = 123
        lang = "en"
        text = ReadingText(account_id=account_id, lang=lang, content="Hello world.")
        db.add(text)
        db.flush()

        # Insert a sentence translation without spans
        rtt = ReadingTextTranslation(
            account_id=account_id,
            text_id=text.id,
            unit="sentence",
            target_lang="en",
            segment_index=0,
            span_start=None,
            span_end=None,
            source_text="Hello world.",
            translated_text="Hello world (tr).",
            provider="test",
            model="test-model",
        )
        db.add(rtt)
        db.commit()

        # Backfill spans
        svc = TranslationService()
        ok = svc.backfill_sentence_spans(db, account_id, text.id)
        assert ok is True

        # Verify row now has spans
        row = (
            db.query(ReadingTextTranslation)
            .filter(ReadingTextTranslation.account_id == account_id,
                    ReadingTextTranslation.text_id == text.id,
                    ReadingTextTranslation.unit == "sentence")
            .first()
        )
        assert row is not None
        assert isinstance(row.span_start, int)
        assert isinstance(row.span_end, int)
        # Spans should be within the bounds of the source text
        assert 0 <= row.span_start <= row.span_end <= len("Hello world.")
    finally:
        db.close()
