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
    echo=IS_DEV,  # Auto-enable SQL logging in dev
)
_configure_sqlite_engine(global_engine)

GlobalSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=global_engine)
SessionLocal = GlobalSessionLocal  # Backwards compatibility
Base = declarative_base()

# ---- Dependency Injection ----
def get_global_db() -> Generator[Session, None, None]:
    """Provide database session with automatic cleanup"""
    db = GlobalSessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- Database Initialization ----
def init_db() -> None:
    """Initialize database: create all tables from models"""
    try:
        # Import models to register them with Base
        from . import models  # noqa: F401

        # Create all tables
        Base.metadata.create_all(bind=global_engine)
        logger.info("Database tables created successfully")

        # Create auth tables
        _ensure_auth_tables()

        # Run any pending migrations
        _run_migrations()

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


def _run_migrations() -> None:
    """Run any pending database migrations"""
    try:
        # Check if reading_texts table has opened_at column
        from sqlalchemy import inspect

        inspector = inspect(global_engine)
        if "reading_texts" in inspector.get_table_names():
            columns = [col["name"] for col in inspector.get_columns("reading_texts")]
            if "opened_at" not in columns:
                logger.info("Adding opened_at column to reading_texts table")
                with global_engine.connect() as conn:
                    conn.execute("ALTER TABLE reading_texts ADD COLUMN opened_at DATETIME")
                    conn.commit()
                logger.info("Successfully added opened_at column")
    except Exception as e:
        logger.warning(f"Migration failed: {e}")
        # Don't raise - migration failures shouldn't break the app

# ---- Utilities ----
def drop_all_tables() -> None:
    """Drop all tables - useful for development resets"""
    logger.warning("Dropping all database tables!")
    Base.metadata.drop_all(bind=global_engine)

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
