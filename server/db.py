from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
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

            # user_lexemes: importance, importance_var
            if not has_column("user_lexemes", "importance"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN importance REAL DEFAULT 0.5")
            if not has_column("user_lexemes", "importance_var"):
                conn.exec_driver_sql("ALTER TABLE user_lexemes ADD COLUMN importance_var REAL DEFAULT 0.3")

            # word_events: text_id
            if not has_column("word_events", "text_id"):
                conn.exec_driver_sql("ALTER TABLE word_events ADD COLUMN text_id INTEGER")
            
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
