from __future__ import annotations

from pathlib import Path
import logging
import os
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
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
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# ---- Consolidated Session Management ----
# Merged from utils/session_helpers.py and utils/session_manager.py

@contextmanager
def global_session() -> Generator[Session, None, None]:
    """
    Context manager for global DB sessions.
    
    Usage:
        with global_session() as db:
            # use db
            result = db.query(...).first()
        # session automatically closed
    """
    db = GlobalSessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Error in global session: {e}")
        db.rollback()
        raise
    finally:
        try:
            db.close()
        except Exception as e:
            logger.warning(f"Error closing global session: {e}")


class DatabaseSessionManager:
    """
    Centralized session management with automatic transaction handling.
    
    Merged from utils/session_manager.py
    """
    
    @contextmanager
    def transaction(self, account_id: Optional[int] = None) -> Generator[Session, None, None]:
        """
        Context manager for read/write database operations.
        
        Automatically handles:
        - Session creation (account-specific or global)
        - Transaction commit on success
        - Transaction rollback on exception  
        - Session cleanup
        
        Args:
            account_id: User account ID for per-account database, None for global
            
        Yields:
            Session: Database session for operations
        """
        db = None
        try:
            # Create appropriate session
            if account_id is not None:
                from .account_db import open_account_session
                db = open_account_session(account_id)
                logger.debug(f"Created account session for account_id={account_id}")
            else:
                # Fall back to global session
                db = GlobalSessionLocal()
                logger.debug("Created global database session")
            
            yield db
            
            # Commit successful transaction
            db.commit()
            logger.debug("Transaction committed successfully")
            
        except Exception as e:
            # Rollback on any exception
            if db:
                try:
                    db.rollback()
                    logger.error(f"Transaction rolled back due to error: {e}")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback transaction: {rollback_error}")
            raise
            
        finally:
            # Always close session
            if db:
                try:
                    db.close()
                    logger.debug("Database session closed")
                except Exception as close_error:
                    logger.error(f"Error closing database session: {close_error}")
    
    @contextmanager
    def read_only(self, account_id: Optional[int] = None) -> Generator[Session, None, None]:
        """
        Context manager for read-only database operations.
        
        Provides read consistency guarantees and automatic cleanup.
        Best for queries that don't modify data.
        
        Args:
            account_id: User account ID for per-account database, None for global
            
        Yields:
            Session: Read-only database session
        """
        db = None
        try:
            # Create appropriate session
            if account_id is not None:
                from .account_db import open_account_session
                db = open_account_session(account_id)
                logger.debug(f"Created read-only account session for account_id={account_id}")
            else:
                # Fall back to global session
                db = GlobalSessionLocal()
                logger.debug("Created read-only global database session")
            
            # Ensure we're in read-only mode
            db.execute(text("BEGIN IMMEDIATE"))  # SQLite read lock
            
            yield db
            
            # No commit needed for read-only
            logger.debug("Read-only session completed")
            
        except Exception as e:
            logger.error(f"Read-only session failed: {e}")
            raise
            
        finally:
            # Always close session
            if db:
                try:
                    db.close()
                    logger.debug("Read-only database session closed")
                except Exception as close_error:
                    logger.error(f"Error closing read-only database session: {close_error}")
    
    def get_session(self, account_id: Optional[int] = None) -> Session:
        """
        Get a database session without automatic transaction management.
        
        Use with caution - caller is responsible for manual transaction handling,
        commit/rollback, and session cleanup.
        
        Args:
            account_id: User account ID for per-account database, None for global
            
        Returns:
            Session: Database session (caller must manage)
            
        Warning:
            This bypasses automatic transaction management. Use only when
            you need custom transaction control, otherwise prefer .transaction()
        """
        if account_id is not None:
            from .account_db import open_account_session
            db = open_account_session(account_id)
            logger.debug(f"Manual account session created for account_id={account_id}")
        else:
            db = GlobalSessionLocal()
            logger.debug("Manual global session created")
            
        return db


@contextmanager
def managed_session(account_id: int, read_only: bool = False) -> Generator[Session, None, None]:
    """
    Context manager for account-specific DB sessions.
    
    Args:
        account_id: Account ID for the session
        read_only: Whether to use read-only session
    
    Usage:
        with managed_session(account_id) as db:
            # use db
            result = db.query(...).first()
        # session automatically closed
    """
    if read_only:
        with db_manager.read_only(account_id) as db:
            yield db
    else:
        with db_manager.transaction(account_id) as db:
            yield db


def execute_with_session(
    func, 
    use_global: bool = True,
    account_id: Optional[int] = None,
    read_only: bool = False,
    *args, **kwargs
):
    """
    Execute a function with proper session management.
    
    Args:
        func: Function to execute (receives db as first argument)
        use_global: Use global session if True, account session if False
        account_id: Account ID for account sessions
        read_only: Use read-only session
        *args, **kwargs: Additional arguments to pass to func
    
    Returns:
        Result of func(db, *args, **kwargs)
    """
    if use_global:
        with global_session() as db:
            return func(db, *args, **kwargs)
    else:
        if account_id is None:
            raise ValueError("account_id is required for account sessions")
        with managed_session(account_id, read_only) as db:
            return func(db, *args, **kwargs)


class SessionHelper:
    """
    Helper class for common session operations.
    
    Merged from utils/session_helpers.py
    """
    
    @staticmethod
    def get_or_create_global(
        model_class,
        filter_kwargs: dict,
        create_kwargs: dict = None
    ):
        """Get or create a model instance using a global session."""
        if create_kwargs is None:
            create_kwargs = {}
        
        def _get_or_create(db: Session):
            instance = db.query(model_class).filter_by(**filter_kwargs).first()
            if not instance:
                instance = model_class(**{**filter_kwargs, **create_kwargs})
                db.add(instance)
                db.commit()
                db.refresh(instance)
            return instance
        
        return execute_with_session(_get_or_create, use_global=True)
    
    @staticmethod
    def get_or_create_account(
        model_class,
        account_id: int,
        filter_kwargs: dict,
        create_kwargs: dict = None
    ):
        """Get or create a model instance using an account session."""
        if create_kwargs is None:
            create_kwargs = {}
        
        def _get_or_create(db: Session):
            instance = db.query(model_class).filter_by(**filter_kwargs).first()
            if not instance:
                instance = model_class(**{**filter_kwargs, **create_kwargs})
                db.add(instance)
                db.commit()
                db.refresh(instance)
            return instance
        
        return execute_with_session(
            _get_or_create, 
            use_global=False, 
            account_id=account_id
        )
    
    @staticmethod
    def safe_commit(db: Session, error_message: str = "Failed to commit transaction"):
        """Safely commit a transaction with error handling."""
        try:
            db.commit()
            return True
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            db.rollback()
            return False
    
    @staticmethod
    def safe_execute(db: Session, query_func, error_message: str = "Query failed"):
        """Safely execute a query with error handling."""
        try:
            return query_func(db)
        except Exception as e:
            logger.error(f"{error_message}: {e}")
            return None


# Global instance for use across all services
db_manager = DatabaseSessionManager()
