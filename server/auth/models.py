from __future__ import annotations

from sqlalchemy import Column, Integer, String, Boolean, Text, JSON, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from typing import Optional, Dict, Any

Base = declarative_base()


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # Core fields from original schema
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False) 
    role = Column(String(50), nullable=True)  # user|admin|...
    subscription_tier = Column(String(50), nullable=True)  # free|pro|...
    
    # Extended fields - apps can add their own here
    # Note: Profile fields (name, timezone, avatar_url) have been removed
    # Applications should implement their own profile systems
    
    # Metadata with JSON fallback for truly flexible data
    extras = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by AuthRepository interface"""
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "role": self.role,
            "subscription_tier": self.subscription_tier,
            "extras": self.extras,
        }


def create_sqlite_engine(database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False):
    """Create SQLite engine with proper configuration"""
    engine = create_engine(database_url, echo=echo, connect_args={"check_same_thread": False})
    return engine


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)

