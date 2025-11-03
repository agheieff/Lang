"""
Centralized database session management utility.

Provides consistent transaction handling, automatic cleanup,
and resource management for database operations across services.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy.orm import Session

from ..account_db import open_account_session

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    """
    Centralized session management with automatic transaction handling.
    
    Provides context managers for common database operations:
    - transaction(): Read/write operations with automatic commit/rollback
    - read_only(): Read-only operations with connection consistency
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
            
        Example:
            with db_manager.transaction(account_id) as db:
                user = db.query(User).first()
                user.name = "New Name"
                # Auto-commit on exit, rollback on exception
        """
        db = None
        try:
            # Create appropriate session
            if account_id is not None:
                db = open_account_session(account_id)
                logger.debug(f"Created account session for account_id={account_id}")
            else:
                # Fall back to global session
                from ..db import GlobalSessionLocal
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
            
        Example:
            with db_manager.read_only(account_id) as db:
                users = db.query(User).all()
                # No transaction, auto-cleanup on exit
        """
        db = None
        try:
            # Create appropriate session
            if account_id is not None:
                db = open_account_session(account_id)
                logger.debug(f"Created read-only account session for account_id={account_id}")
            else:
                # Fall back to global session
                from ..db import GlobalSessionLocal
                db = GlobalSessionLocal()
                logger.debug("Created read-only global database session")
            
            # Ensure we're in read-only mode
            db.execute("BEGIN IMMEDIATE")  # SQLite read lock
            
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
            db = open_account_session(account_id)
            logger.debug(f"Manual account session created for account_id={account_id}")
        else:
            from ..db import GlobalSessionLocal
            db = GlobalSessionLocal()
            logger.debug("Manual global session created")
            
        return db


# Global instance for use across all services
db_manager = DatabaseSessionManager()
