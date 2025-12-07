"""LLM and generation related models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column
from server.db import Base


class LLMModel(Base):
    """Available LLM models and their capabilities."""
    __tablename__ = "llm_models"
    __table_args__ = (
        UniqueConstraint("mid", name="uq_llm_models_mid"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mid: Mapped[str] = mapped_column(String(128), index=True)  # provider/model-id
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    label: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    family: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    context_window: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    max_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    modalities: Mapped[dict] = mapped_column(JSON, default=dict)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    tiers: Mapped[dict] = mapped_column(JSON, default=dict)
    pricing: Mapped[dict] = mapped_column(JSON, default=dict)
    limits: Mapped[dict] = mapped_column(JSON, default=dict)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class GenerationLog(Base):
    """Log of text generation requests and responses."""
    __tablename__ = "generation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[Optional[int]] = mapped_column(ForeignKey("profiles.id", ondelete="SET NULL"), index=True, default=None)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    base_url: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    prompt: Mapped[dict] = mapped_column(JSON, default=dict)
    words: Mapped[dict] = mapped_column(JSON, default=dict)
    level_hint: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    approx_len: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    unit: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TranslationLog(Base):
    """Log of translation requests and responses."""
    __tablename__ = "translation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[Optional[int]] = mapped_column(ForeignKey("reading_texts.id", ondelete="SET NULL"), index=True, default=None)
    unit: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    target_lang: Mapped[Optional[str]] = mapped_column(String(8), default=None)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    prompt: Mapped[dict] = mapped_column(JSON, default=dict)
    segments: Mapped[dict] = mapped_column(JSON, default=dict)
    response: Mapped[Optional[str]] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LLMRequestLog(Base):
    """Comprehensive log of all LLM requests."""
    __tablename__ = "llm_request_logs"
    __table_args__ = (
        Index("ix_llmrl_account_created", "account_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, default=None)
    text_id: Mapped[Optional[int]] = mapped_column(ForeignKey("reading_texts.id", ondelete="SET NULL"), index=True, default=None)
    kind: Mapped[str] = mapped_column(String(32))  # reading|translation|other
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    base_url: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    status: Mapped[str] = mapped_column(String(16), default="error")  # ok|error
    request: Mapped[dict] = mapped_column(JSON, default=dict)  # typically {messages: [...]} and params
    response: Mapped[Optional[str]] = mapped_column(String, default=None)
    error: Mapped[Optional[str]] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class GenerationRetryAttempt(Base):
    """Track retry attempts for failed generation components."""
    __tablename__ = "generation_retry_attempts"
    __table_args__ = (
        Index("ix_genretry_account_text", "account_id", "text_id"),
        Index("ix_genretry_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    # Which components failed: bitmap: 1=words, 2=sentences, 4=structured (combinations allowed)
    failed_components: Mapped[int] = mapped_column(Integer, default=0)  # bitmask: bits 0-1=words, 2-3=sentences, etc.
    # Retry attempt number (starts at 1)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    # Overall status of this retry attempt
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|completed|failed
    # What was actually completed in this attempt
    completed_components: Mapped[int] = mapped_column(Integer, default=0)  # bitmask of what succeeded
    # Error details if failed
    error_details: Mapped[Optional[str]] = mapped_column(String, default=None)
    # When this retry was attempted
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
