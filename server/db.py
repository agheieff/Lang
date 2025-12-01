from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# ---- Configuration ----
DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

# Development vs Production settings
IS_DEV = os.getenv("ENV", "development") == "development"

# ---- SQLite Optimization ----
def _configure_sqlite_engine(engine: Engine) -> None:
    """Configure SQLite for optimal performance and safety"""
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")      # Better concurrency
            cursor.execute("PRAGMA synchronous=NORMAL")    # Balance durability/performance
            cursor.execute("PRAGMA foreign_keys=ON")       # Enforce relationships
            cursor.execute("PRAGMA busy_timeout=30000")    # 30s timeout
        except Exception as e:
            logger.warning(f"Failed to set SQLite pragmas: {e}")
        finally:
            cursor.close()

# ---- Engine & Session Setup ----
global_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0,
    },
    pool_pre_ping=True,  # Verify connections before use
    echo=False,  # Silence raw SQL logs in stdout
)
_configure_sqlite_engine(global_engine)

GlobalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=global_engine)
SessionLocal = GlobalSessionLocal  # Backwards compatibility
Base = declarative_base()

# Tables for global DB (shared reference data + profile settings)
# Per-account DBs have: lexemes, word_events, reading history, etc.
GLOBAL_TABLES = {
    "accounts",
    "profiles",
    "profile_prefs",  # Preferences are part of profile settings
    "languages",
    "reading_texts",
    "reading_word_glosses",
    "reading_text_translations",
    "text_vocabulary",
    "llm_models",
    "subscription_tiers",
    "generation_logs",
    "translation_logs",
    "llm_request_logs",
    "reading_lookups",
}

# ---- Dependency Injection ----
def get_global_db() -> Generator[Session, None, None]:
    """Provide database session with automatic cleanup"""
    db = GlobalSessionLocal()
    try:
        yield db
    finally:
        db.close()


def open_global_session() -> Session:
    """Create and return a global DB Session (caller must close).
    
    Use this for background threads that need to access global DB tables
    (profiles, texts, translations, etc.).
    """
    return GlobalSessionLocal()

# ---- Database Initialization ----
def init_db() -> None:
    """Initialize database: create only global tables from models"""
    try:
        # Import models to register them with Base
        from . import models  # noqa: F401

        # Create only global tables (per-account tables go in per-account DBs)
        with global_engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name in GLOBAL_TABLES:
                    table.create(bind=conn, checkfirst=True)
        logger.info("Global database tables created successfully")

        # Create auth tables
        _ensure_auth_tables()

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def _ensure_auth_tables() -> None:
    """Create authentication tables from local auth package"""
    try:
        from server.auth import create_sqlite_engine, create_tables

        auth_engine = create_sqlite_engine(f"sqlite:///{DB_PATH}")
        create_tables(auth_engine)
        auth_engine.dispose()
        logger.info("Auth tables created successfully")

    except Exception as e:
        logger.warning(f"Failed to create auth tables: {e}")
        # Don't raise - auth might be optional


# ---- Utilities ----
def drop_all_tables() -> None:
    """Drop global tables - useful for development resets"""
    logger.warning("Dropping global database tables!")
    from . import models  # noqa: F401
    with global_engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in GLOBAL_TABLES:
                table.drop(bind=conn, checkfirst=True)

def recreate_db() -> None:
    """Drop and recreate all tables - development only"""
    drop_all_tables()
    init_db()

def get_db_info() -> dict:
    """Get database information for debugging"""
    from sqlalchemy import inspect

    inspector = inspect(global_engine)
    return {
        "path": str(DB_PATH),
        "exists": DB_PATH.exists(),
        "size_mb": DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0,
        "tables": inspector.get_table_names(),
    }

def check_db_health() -> bool:
    """Verify database connectivity"""
    try:
        with global_engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
