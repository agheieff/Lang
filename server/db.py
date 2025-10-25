from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

# SQLite tuning for concurrent web usage: allow cross-thread access, wait for locks,
# and prefer WAL journaling to reduce writer contention.
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0,  # seconds to wait on database locks
    },
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore[no-redef]
    try:
        cur = dbapi_connection.cursor()
        # Enable WAL for better concurrency (one writer, many readers)
        cur.execute("PRAGMA journal_mode=WAL")
        # Reasonable durability vs performance
        cur.execute("PRAGMA synchronous=NORMAL")
        # Enforce FK constraints
        cur.execute("PRAGMA foreign_keys=ON")
        # Additional busy timeout at connection level (ms)
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()
    except Exception:
        # Best-effort; ignore if driver doesn't support pragmas
        pass
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401 - ensure models are imported
    _run_migrations()
    _ensure_tables()


def _run_migrations() -> None:
    """Lightweight, idempotent migrations for SQLite.

    Adds new columns when missing. For complex changes, prefer Alembic in the future.
    """
    try:
        with engine.begin() as conn:
            def has_column(table: str, name: str) -> bool:
                rows = conn.exec_driver_sql(f"PRAGMA table_info('{table}')").all()
                return any(r[1] == name for r in rows)

            # profiles: level_value, level_var, level_code
            if not has_column("profiles", "level_value"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN level_value REAL DEFAULT 0.0")
            if not has_column("profiles", "level_var"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN level_var REAL DEFAULT 1.0")
            if not has_column("profiles", "level_code"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN level_code VARCHAR(32)")
            # profiles: preferred_script (for Chinese)
            if not has_column("profiles", "preferred_script"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN preferred_script VARCHAR(8)")

            # user_lexemes: importance, importance_var
            if not has_column("user_lexemes", "importance"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN importance REAL DEFAULT 0.5")
            if not has_column("user_lexemes", "importance_var"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN importance_var REAL DEFAULT 0.3")

            # word_events: text_id
            if not has_column("word_events", "text_id"):
                conn.exec_driver_sql("ALTER TABLE word_events ADD COLUMN text_id INTEGER")

            # user_lexemes: alpha, beta, difficulty, last_decay_at
            if not has_column("user_lexemes", "alpha"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN alpha REAL DEFAULT 1.0")
            if not has_column("user_lexemes", "beta"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN beta REAL DEFAULT 9.0")
            if not has_column("user_lexemes", "difficulty"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN difficulty REAL DEFAULT 1.0")
            if not has_column("user_lexemes", "last_decay_at"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN last_decay_at DATETIME")

            # indexes for performance
            try:
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_word_events_user_prof_lex_ts ON word_events(user_id, profile_id, lexeme_id, ts)")
            except Exception:
                pass
            try:
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_lexeme_info_freq_rank ON lexeme_info(freq_rank)")
            except Exception:
                pass
            try:
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_user_lexemes_due ON user_lexemes(profile_id, next_due_at)")
            except Exception:
                pass
            # translation_logs: response text (to support conversation continuation)
            if not has_column("translation_logs", "response"):
                conn.exec_driver_sql("ALTER TABLE translation_logs ADD COLUMN response TEXT")
            # reading_texts: is_read, read_at
            if not has_column("reading_texts", "is_read"):
                conn.exec_driver_sql("ALTER TABLE reading_texts ADD COLUMN is_read BOOLEAN DEFAULT 0")
            if not has_column("reading_texts", "read_at"):
                conn.exec_driver_sql("ALTER TABLE reading_texts ADD COLUMN read_at DATETIME")
            
            # Tables mapped by SQLAlchemy will be created in _ensure_tables
    except Exception:
        # Best-effort; avoid crashing app startup
        pass


def _ensure_tables() -> None:
    try:
        insp = inspect(engine)
        with engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    table.create(bind=conn)
    except Exception:
        pass
