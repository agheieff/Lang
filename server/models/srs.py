"""Spaced Repetition System (SRS) and vocabulary related models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column
from server.db import Base


class Card(Base):
    """Placeholder for future SRS implementation."""
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    head: Mapped[str] = mapped_column(String(256))
    lang: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Lexeme(Base):
    """User-specific vocabulary entries with SRS tracking."""
    __tablename__ = "lexemes"
    __table_args__ = (
        UniqueConstraint("account_id", "profile_id", "lang", "lemma", "pos", name="uq_lexeme_account_profile_lang_lemma_pos"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # User-specific lexeme tracking
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)

    # Lexeme data
    lang: Mapped[str] = mapped_column(String(16), index=True)
    lemma: Mapped[str] = mapped_column(String(256), index=True)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)

    # Global word properties
    frequency_rank: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # Global frequency rank
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)  # e.g., 'HSK1', 'A2'

    # SRS tracking (from UserLexeme)
    a_click: Mapped[int] = mapped_column(Integer, default=1)
    b_nonclick: Mapped[int] = mapped_column(Integer, default=4)
    stability: Mapped[float] = mapped_column(Float, default=0.2)  # 0..1
    # Bayesian posterior for click propensity (non-lookup vs click)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=9.0)
    difficulty: Mapped[float] = mapped_column(Float, default=1.0)
    # Word importance (e.g., based on frequency or curriculum); user-specific so it can evolve
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    importance_var: Mapped[float] = mapped_column(Float, default=0.3)

    exposures: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    distinct_texts: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    last_clicked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    next_due_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    last_decay_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    # Familiarity tracking for word selection
    familiarity: Mapped[Optional[float]] = mapped_column(Float, default=None)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class LexemeVariant(Base):
    """Lexeme form variants (e.g., simplified/traditional Chinese)."""
    __tablename__ = "lexeme_variants"
    __table_args__ = (
        UniqueConstraint("script", "form", name="uq_variant_script_form"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    script: Mapped[str] = mapped_column(String(8), index=True)  # Hans | Hant
    form: Mapped[str] = mapped_column(String(256), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class WordEvent(Base):
    """Tracking of word interaction events."""
    __tablename__ = "word_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    # Note: lexeme_id removed since lexemes are now user-specific and contain account/profile info
    event_type: Mapped[str] = mapped_column(String(16))  # exposure|click|assign|hover
    count: Mapped[int] = mapped_column(Integer, default=1)
    surface: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    source: Mapped[Optional[str]] = mapped_column(String(16), default=None)  # llm|manual|unknown
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    text_id: Mapped[Optional[int]] = mapped_column(Integer, default=None, index=True)


class UserLexemeContext(Base):
    """Context in which a lexeme was encountered."""
    __tablename__ = "user_lexeme_contexts"
    __table_args__ = (
        UniqueConstraint("lexeme_id", "context_hash", name="uq_lexeme_ctx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    context_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
