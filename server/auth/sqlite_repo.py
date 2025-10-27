from __future__ import annotations

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session, sessionmaker
from .repo import MutableAuthRepository
from .models import Account, create_sqlite_engine, create_tables


class SQLiteRepository(MutableAuthRepository):
    """SQLite implementation of AuthRepository with extensible schema support."""
    
    def __init__(self, database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False):
        """Initialize SQLite repository.
        
        Args:
            database_url: SQLAlchemy database URL
            echo: Whether to echo SQL statements (useful for debugging)
        """
        self.engine = create_sqlite_engine(database_url, echo)
        create_tables(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def find_account_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Find account by email (case-insensitive)"""
        with self._get_session() as session:
            account = session.query(Account).filter(
                Account.email.ilike(email.strip().lower())
            ).first()
            return account.to_dict() if account else None

    def get_account_credentials(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials-only view for login path"""
        with self._get_session() as session:
            account = session.query(Account).filter(
                Account.email.ilike(email.strip().lower())
            ).first()
            if not account:
                return None
            return {
                "id": account.id,
                "password_hash": account.password_hash,
                "is_active": bool(account.is_active),
                "is_verified": bool(account.is_verified),
            }
    
    def create_account(self, email: str, password_hash: str, **extra_fields) -> Dict[str, Any]:
        """Create a new account with optional extended fields"""
        with self._get_session() as session:
            # Check if email already exists
            existing = session.query(Account).filter(
                Account.email.ilike(email.strip().lower())
            ).first()
            if existing:
                raise ValueError("email already registered")

            # Create account with extended fields
            account = Account(
                email=email.strip().lower(),
                password_hash=password_hash,
                **extra_fields  # Allow arbitrary extra fields from apps
            )

            session.add(account)
            session.commit()
            session.refresh(account)

            return account.to_dict()
    
    def get_account_by_id(self, account_id: str | int) -> Optional[Dict[str, Any]]:
        """Get account by ID"""
        with self._get_session() as session:
            account = session.query(Account).filter(Account.id == int(account_id)).first()
            return account.to_dict() if account else None
    
    def update_account(self, account_id: str | int, **updates) -> Optional[Dict[str, Any]]:
        """Update account with extended field support"""
        with self._get_session() as session:
            account = session.query(Account).filter(Account.id == int(account_id)).first()
            if not account:
                return None
            
            # Update fields that exist on the model
            for field, value in updates.items():
                if hasattr(account, field):
                    setattr(account, field, value)
            
            session.commit()
            session.refresh(account)
            return account.to_dict()


# Convenience function for easy instantiation
def create_sqlite_repo(database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False) -> SQLiteRepository:
    """Create a SQLite repository instance"""
    return SQLiteRepository(database_url, echo)

