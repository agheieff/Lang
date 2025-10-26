from __future__ import annotations

from pathlib import Path
import os
from typing import Generator, Optional, Dict

from fastapi import Request
from sqlalchemy import create_engine, inspect, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"  # Global/app DB (auth, shared catalogs)

# ---- Engine helpers ----
def _configure_sqlite_engine(e: Engine) -> None:
    @event.listens_for(e, "connect")
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


# Global engine (shared/auth data)
global_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0,
    },
)
_configure_sqlite_engine(global_engine)


GlobalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=global_engine)
# Backwards-compat name used in a few places for global lookups (e.g., rate limit)
SessionLocal = GlobalSessionLocal

Base = declarative_base()

# ---- Per-account DB routing ----
_ACCOUNT_ENGINES: Dict[int, Engine] = {}
_ACCOUNTS_DIR = DATA_DIR / "accounts"
_ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)

# Tables to provision in per-account DBs (user-scoped data)
PER_ACCOUNT_TABLES = {
    # learning/profile + reading
    "profiles",
    "profile_prefs",
    "reading_texts",
    "cards",
    # lexicon + user stats
    "lexemes",
    "lexeme_variants",
    "lexeme_info",
    "user_lexemes",
    "user_lexeme_contexts",
    "word_events",
    # activity/logs per reading
    "generation_logs",
    "reading_text_translations",
    "reading_word_glosses",
    "translation_logs",
    "reading_lookups",
    # request logs
    "llm_request_logs",
    # curated/user-specific lists
    "language_word_lists",
}


def _account_db_path(account_id: int) -> Path:
    return _ACCOUNTS_DIR / f"{int(account_id)}.db"


def _get_or_create_account_engine(account_id: int) -> Engine:
    aid = int(account_id)
    e = _ACCOUNT_ENGINES.get(aid)
    if e is not None:
        return e
    path = _account_db_path(aid)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={
            "check_same_thread": False,
            "timeout": 30.0,
        },
    )
    _configure_sqlite_engine(eng)
    _ACCOUNT_ENGINES[aid] = eng
    return eng


def _ensure_account_tables(account_engine: Engine) -> None:
    try:
        insp = inspect(account_engine)
        with account_engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name not in PER_ACCOUNT_TABLES:
                    continue
                if not insp.has_table(table.name):
                    table.create(bind=conn)
    except Exception:
        # don't fail request on best-effort ensure
        pass


def _resolve_account_id(request: Optional[Request]) -> Optional[int]:
    if request is None:
        return None
    # Prefer middleware-populated user
    try:
        user = getattr(request.state, "user", None)
        if user is not None and hasattr(user, "id"):
            return int(getattr(user, "id"))
    except Exception:
        pass
    # Fallback to Authorization bearer token (decode only)
    try:
        from arcadia_auth import decode_token  # type: ignore
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
            secret = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
            data = decode_token(token, secret, ["HS256"])  # type: ignore
            sub = (data or {}).get("sub")
            if sub is not None:
                return int(sub)
    except Exception:
        pass
    return None


def get_db(request: Request) -> Generator[Session, None, None]:
    """FastAPI dependency: yields a per-account Session when authenticated,
    otherwise falls back to the global DB session.
    """
    aid = _resolve_account_id(request)
    if aid is None:
        db = GlobalSessionLocal()
        try:
            yield db
        finally:
            db.close()
        return
    eng = _get_or_create_account_engine(aid)
    _ensure_account_tables(eng)
    Local = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    db = Local()
    try:
        yield db
    finally:
        db.close()


def get_global_db() -> Generator[Session, None, None]:
    db = GlobalSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from . import models  # noqa: F401 - ensure models are imported
    _run_migrations()
    _ensure_tables()
    _ensure_auth_tables()


def _run_migrations() -> None:
    """Lightweight, idempotent migrations for SQLite.

    Adds new columns when missing. For complex changes, prefer Alembic in the future.
    """
    try:
        with global_engine.begin() as conn:
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
            # profiles: target_lang
            if not has_column("profiles", "target_lang"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN target_lang VARCHAR(16) DEFAULT 'en'")
            # profiles: settings (JSON for flexible preferences)
            if not has_column("profiles", "settings"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN settings TEXT DEFAULT '{}'")  # SQLite doesn't support JSON type natively
            # profiles: text_length, text_preferences
            if not has_column("profiles", "text_length"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN text_length INTEGER")
            if not has_column("profiles", "text_preferences"):
                conn.exec_driver_sql("ALTER TABLE profiles ADD COLUMN text_preferences TEXT")

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
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_word_events_acct_prof_lex_ts ON word_events(account_id, profile_id, lexeme_id, ts)")
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
            try:
                conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_ul_acct_profile ON user_lexemes(account_id, profile_id)")
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
        insp = inspect(global_engine)
        with global_engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    table.create(bind=conn)
    except Exception:
        pass


def _ensure_auth_tables() -> None:
    """Ensure auth tables from arcadia_auth are created"""
    try:
        from arcadia_auth import create_sqlite_engine, create_tables

        # Create auth tables in the same global database
        auth_engine = create_sqlite_engine(f"sqlite:///{DB_PATH}")
        create_tables(auth_engine)
        auth_engine.dispose()
    except Exception as e:
        # Best-effort; log but don't crash
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to create auth tables: {e}")
