"""Authentication and account related models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from server.db import Base


class SubscriptionTier(Base):
    """User subscription tiers."""
    __tablename__ = "subscription_tiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), default=None)


class UsageTracking(Base):
    """Track monthly usage for Free tier quota enforcement."""
    __tablename__ = "usage_tracking"
    __table_args__ = (
        UniqueConstraint("account_id", "period_start", name="uq_usage_account_period"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # First of month
    texts_generated: Mapped[int] = mapped_column(Integer, default=0)
    chars_generated: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserModelConfig(Base):
    """
    User's configured LLM models.
    
    Models can come from:
    - "user": User-defined custom models with their own API keys
    - "system": Server-injected models (e.g., free tier defaults)
    - "subscription": Models granted by paid subscription
    
    System/subscription models have is_editable=False and is_key_visible=False.
    """
    __tablename__ = "user_model_configs"
    __table_args__ = (
        UniqueConstraint("account_id", "model_id", "source", name="uq_umc_account_model_source"),
        Index("ix_umc_account_active", "account_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    
    # Model identification
    display_name: Mapped[str] = mapped_column(String(128))  # User-editable friendly name
    provider: Mapped[str] = mapped_column(String(64))  # "openrouter", "openai", "anthropic", "local"
    model_id: Mapped[str] = mapped_column(String(256))  # e.g. "anthropic/claude-3-sonnet"
    base_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # Custom endpoint
    api_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # User's own key
    
    # Source tracking
    source: Mapped[str] = mapped_column(String(32), default="user")  # "user" | "system" | "subscription"
    is_editable: Mapped[bool] = mapped_column(Boolean, default=True)  # False for system-injected
    is_key_visible: Mapped[bool] = mapped_column(Boolean, default=True)  # False for system models
    
    # Model capabilities/settings
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    capabilities: Mapped[dict] = mapped_column(JSON, default=list)  # ["text", "translation"]
    extra_params: Mapped[dict] = mapped_column(JSON, default=dict)  # Temperature, etc.
    
    # Usage assignment - which tasks this model is used for
    use_for_generation: Mapped[bool] = mapped_column(Boolean, default=False)
    use_for_word_translation: Mapped[bool] = mapped_column(Boolean, default=False)
    use_for_sentence_translation: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Ordering/priority
    priority: Mapped[int] = mapped_column(Integer, default=100)  # Lower = higher priority
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)  # User can disable models
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class UserProviderConfig(Base):
    """DEPRECATED: Use UserModelConfig instead. Kept for migration compatibility."""
    __tablename__ = "user_provider_configs"
    __table_args__ = (
        UniqueConstraint("account_id", "provider_name", name="uq_upc_account_provider"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    
    # e.g., "OpenRouter", "Local", "My Custom OpenAI"
    provider_name: Mapped[str] = mapped_column(String(64))
    # e.g., "openrouter", "openai_compatible"
    provider_type: Mapped[str] = mapped_column(String(32))
    
    base_url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    
    # Optional: Default model ID to use with this provider
    default_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    
    # Extra metadata matching ProviderConfig
    app_title: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    referer: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NextReadyOverride(Base):
    """Override next text ready status for testing/debugging."""
    __tablename__ = "next_ready_overrides"
    __table_args__ = (
        UniqueConstraint("account_id", "lang", name="uq_nro_account_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    lang: Mapped[str] = mapped_column(String(16), index=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
