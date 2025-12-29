from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session
from server.models import Language


def _has_column(inspector, table: str, col: str) -> bool:
    try:
        return any(c.get("name") == col for c in inspector.get_columns(table))
    except Exception:
        return False


def ensure_reading_text_lifecycle_columns(db: Session) -> None:
    """Best-effort, idempotent addition of lifecycle columns on reading_texts.

    Safe to call on SQLite; ignores errors if columns already exist.
    """
    try:
        eng = db.get_bind()
        insp = inspect(eng)
        cols = {
            "request_sent_at": "ALTER TABLE reading_texts ADD COLUMN request_sent_at TIMESTAMP",
            "generated_at": "ALTER TABLE reading_texts ADD COLUMN generated_at TIMESTAMP",
            "is_read": "ALTER TABLE reading_texts ADD COLUMN is_read BOOLEAN DEFAULT 0",
            "read_at": "ALTER TABLE reading_texts ADD COLUMN read_at TIMESTAMP",
            "opened_at": "ALTER TABLE reading_texts ADD COLUMN opened_at TIMESTAMP",
            "words_complete": "ALTER TABLE reading_texts ADD COLUMN words_complete BOOLEAN DEFAULT 0",
            "sentences_complete": "ALTER TABLE reading_texts ADD COLUMN sentences_complete BOOLEAN DEFAULT 0",
            "translation_attempts": "ALTER TABLE reading_texts ADD COLUMN translation_attempts INTEGER DEFAULT 0",
            "last_translation_attempt": "ALTER TABLE reading_texts ADD COLUMN last_translation_attempt TIMESTAMP",
        }
        for name, ddl in cols.items():
            if not _has_column(insp, "reading_texts", name):
                try:
                    db.execute(text(ddl))
                except Exception:
                    pass
        try:
            db.commit()
        except Exception:
            db.rollback()
    except Exception:
        # Non-fatal
        try:
            db.rollback()
        except Exception:
            pass


def seed_languages(db: Session) -> None:
    """Seed the languages table with supported languages."""
    languages = [
        {
            "code": "es",
            "name": "Spanish",
            "display_name": "Spanish",
            "script": "Latin",
            "is_enabled": True,
        },
        {
            "code": "zh",
            "name": "Chinese",
            "display_name": "Chinese (Simplified)",
            "script": "Hans",
            "is_enabled": True,
        },
        {
            "code": "zh-TW",
            "name": "Chinese",
            "display_name": "Chinese (Traditional)",
            "script": "Hant",
            "is_enabled": True,
        },
        {
            "code": "en",
            "name": "English",
            "display_name": "English",
            "script": "Latin",
            "is_enabled": True,
        },
    ]

    existing_codes = {lang.code for lang in db.query(Language).all()}

    for lang_data in languages:
        if lang_data["code"] not in existing_codes:
            lang = Language(**lang_data)
            db.add(lang)

    db.commit()
