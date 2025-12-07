from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# ---- Configuration ----
DATA_DIR = Path.cwd() / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "app.db"

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

# ---- Single Engine & Session Setup ----
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0,
    },
    pool_pre_ping=True,
    echo=False,
)
_configure_sqlite_engine(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---- Dependency Injection ----
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- Database Initialization ----
def init_db() -> None:
    """Initialize database: create all tables"""
    try:
        from . import models  # noqa: F401

        with engine.begin() as conn:
            Base.metadata.create_all(bind=conn, checkfirst=True)
        logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


# ---- Utilities ----
def drop_all_tables() -> None:
    """Drop all tables - useful for development resets"""
    logger.warning("Dropping all database tables!")
    from . import models  # noqa: F401
    with engine.begin() as conn:
        Base.metadata.drop_all(bind=conn, checkfirst=True)

def recreate_db() -> None:
    """Drop and recreate all tables - development only"""
    drop_all_tables()
    init_db()

def get_db_info() -> dict:
    """Get database information for debugging"""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    return {
        "path": str(DB_PATH),
        "exists": DB_PATH.exists(),
        "size_mb": DB_PATH.stat().st_size / (1024 * 1024) if DB_PATH.exists() else 0,
        "tables": inspector.get_table_names(),
    }

def check_db_health() -> bool:
    """Verify database connectivity"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
