"""
Consolidated SQLAlchemy models for reading texts, SRS, LLM, and authentication.
All database models in one module for better organization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    ForeignKey,
    Boolean,
    UniqueConstraint,
    Index,
    JSON,
    Float,
    Column,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


# =============================================================================
# Authentication Models
# =============================================================================


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)

    # Core fields
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False)
    # Unified tier for both access control and features: Free|Standard|Pro|Pro+|BYOK|admin|system
    subscription_tier = Column(String(50), default="Free", nullable=False)

    # OpenRouter per-user key management (for paid tiers)
    openrouter_key_encrypted = Column(Text, nullable=True)
    openrouter_key_id = Column(String(128), nullable=True)

    # Extended fields
    extras = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def to_dict(self):
        """Convert to dict format expected by AuthRepository interface"""
        return {
            "id": self.id,
            "email": self.email,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "subscription_tier": self.subscription_tier,
            "extras": self.extras,
            "has_openrouter_key": self.openrouter_key_id is not None,
        }


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
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    texts_generated: Mapped[int] = mapped_column(Integer, default=0)
    chars_generated: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


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
        UniqueConstraint(
            "account_id", "model_id", "source", name="uq_umc_account_model_source"
        ),
        Index("ix_umc_account_active", "account_id", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)

    # Model identification
    display_name: Mapped[str] = mapped_column(String(128))
    provider: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(256))
    base_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    api_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Source tracking
    source: Mapped[str] = mapped_column(String(32), default="user")
    is_editable: Mapped[bool] = mapped_column(Boolean, default=True)
    is_key_visible: Mapped[bool] = mapped_column(Boolean, default=True)

    # Model capabilities/settings
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    capabilities: Mapped[dict] = mapped_column(JSON, default=list)
    extra_params: Mapped[dict] = mapped_column(JSON, default=dict)

    # Usage assignment
    use_for_generation: Mapped[bool] = mapped_column(Boolean, default=False)
    use_for_word_translation: Mapped[bool] = mapped_column(Boolean, default=False)
    use_for_sentence_translation: Mapped[bool] = mapped_column(Boolean, default=False)

    # Ordering/priority
    priority: Mapped[int] = mapped_column(Integer, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# =============================================================================
# Reading Models
# =============================================================================


class TextUnit(str, Enum):
    """Text unit types for translations."""

    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TEXT = "text"


class Language(Base):
    """Supported languages."""

    __tablename__ = "languages"
    __table_args__ = (UniqueConstraint("code", name="uq_language_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str] = mapped_column(String(128))
    script: Mapped[str] = mapped_column(String(16))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class Profile(Base):
    """User language learning profiles - one profile per (account, lang, target_lang) combination."""

    __tablename__ = "profiles"
    __table_args__ = (
        UniqueConstraint(
            "account_id", "lang", "target_lang", name="uq_profile_account_lang_target"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)

    # Language pair (immutable after creation)
    lang: Mapped[str] = mapped_column(String(16), index=True)
    target_lang: Mapped[str] = mapped_column(String(16), index=True)

    # User's level in this language
    level_value: Mapped[float] = mapped_column(Float, default=0.0)
    level_var: Mapped[float] = mapped_column(Float, default=1.0)
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)

    # Preferences
    preferred_script: Mapped[Optional[str]] = mapped_column(String(8), default=None)
    text_length: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    ci_preference: Mapped[float] = mapped_column(Float, default=0.92)

    # Current reading state
    current_text_id: Mapped[Optional[int]] = mapped_column(
        Integer, index=True, default=None
    )

    # Async preferences update tracking
    preferences_updating: Mapped[bool] = mapped_column(Boolean, default=False)

    # Re-read settings: None = never show again, 0 = always allow, N = cooldown in days
    reread_cooldown_days: Mapped[Optional[int]] = mapped_column(
        Integer, default=None, nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ProfileTopicPref(Base):
    """User's topic preferences with weights for text generation.

    Topics are shared across profiles but each profile can customize weights.
    Example: {"fiction": 2.0, "news": 0.5} means prefer fiction 4x more than news.
    """

    __tablename__ = "profile_topic_prefs"
    __table_args__ = (
        UniqueConstraint("profile_id", "topic", name="uq_ptp_profile_topic"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    topic: Mapped[str] = mapped_column(String(32), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ProfileSetting(Base):
    """General profile settings stored as key-value pairs."""

    __tablename__ = "profile_settings"
    __table_args__ = (UniqueConstraint("profile_id", "key", name="uq_ps_profile_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    key: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ReadingText(Base):
    """Global text storage - texts are shared across users with matching lang/target_lang."""

    __tablename__ = "reading_texts"
    __table_args__ = (
        Index("ix_rt_lang_target", "lang", "target_lang"),
        Index(
            "ix_rt_ready", "lang", "target_lang", "words_complete", "sentences_complete"
        ),
        Index("ix_rt_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Who requested generation (nullable for imported texts)
    generated_for_account_id: Mapped[Optional[int]] = mapped_column(
        Integer, index=True, nullable=True
    )

    # Language pair
    lang: Mapped[str] = mapped_column(String(16), index=True)
    target_lang: Mapped[str] = mapped_column(String(16), index=True)

    # Content
    content: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    source: Mapped[Optional[str]] = mapped_column(String(16), default="llm")

    # Generation lifecycle
    request_sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # Pool-based selection fields
    ci_target: Mapped[Optional[float]] = mapped_column(Float, default=None)
    topic: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    difficulty_estimate: Mapped[Optional[float]] = mapped_column(Float, default=None)

    # Completion flags
    words_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    sentences_complete: Mapped[bool] = mapped_column(Boolean, default=False)

    # Retry tracking
    translation_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_translation_attempt: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # Vocabulary stats
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_lemma_count: Mapped[int] = mapped_column(Integer, default=0)

    # User ratings and feedback
    rating_avg: Mapped[Optional[float]] = mapped_column(Float, default=None)
    rating_count: Mapped[int] = mapped_column(Integer, default=0)
    report_count: Mapped[int] = mapped_column(Integer, default=0)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)

    # Generation prompt data (for reproducibility)
    prompt_words: Mapped[dict] = mapped_column(JSON, default=dict)
    prompt_level_hint: Mapped[Optional[str]] = mapped_column(String(128), default=None)

    @property
    def is_ready(self) -> bool:
        """Text is ready for reading when it has content AND complete translations."""
        return bool(self.content and self.words_complete and self.sentences_complete)

    @classmethod
    def ready_filter(cls):
        """Filter conditions for ready texts - use with query.filter()."""
        return (
            cls.content.is_not(None),
            cls.content != "",
            cls.words_complete == True,
            cls.sentences_complete == True,
        )


class ReadingTextTranslation(Base):
    """Global translations storage - shared across users."""

    __tablename__ = "reading_text_translations"
    __table_args__ = (
        UniqueConstraint(
            "text_id",
            "target_lang",
            "unit",
            "segment_index",
            name="uq_rtt_unique",
        ),
        Index("ix_rtt_text_target", "text_id", "target_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    target_lang: Mapped[str] = mapped_column(String(8), index=True)
    unit: Mapped[str] = mapped_column(String(16))
    segment_index: Mapped[Optional[int]] = mapped_column(
        Integer, default=None, nullable=True
    )
    source_text: Mapped[str] = mapped_column(String)
    translated_text: Mapped[str] = mapped_column(String)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ReadingWordGloss(Base):
    """Global word glosses - per-word translations for texts."""

    __tablename__ = "reading_word_glosses"
    __table_args__ = (
        UniqueConstraint(
            "text_id",
            "target_lang",
            "span_start",
            "span_end",
            name="uq_rwg_text_span",
        ),
        Index("ix_rwg_text_id", "text_id"),
        Index("ix_rwg_text_target", "text_id", "target_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    target_lang: Mapped[str] = mapped_column(String(16), index=True)
    lang: Mapped[str] = mapped_column(String(16))
    surface: Mapped[str] = mapped_column(String(256))
    lemma: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    pinyin: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    translation: Mapped[Optional[str]] = mapped_column(String, default=None)
    lemma_translation: Mapped[Optional[str]] = mapped_column(String, default=None)
    grammar: Mapped[dict] = mapped_column(JSON, default=dict)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ReadingLookup(Base):
    """User word lookups during reading - tracks what users click on."""

    __tablename__ = "reading_lookups"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "text_id",
            "span_start",
            "span_end",
            name="uq_reading_lookup_span",
        ),
        Index("ix_rl_text_id", "text_id"),
        Index("ix_rl_account_text", "account_id", "text_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    lang: Mapped[str] = mapped_column(String(16))
    surface: Mapped[str] = mapped_column(String(256))
    lemma: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    translations: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TextVocabulary(Base):
    """Vocabulary index for texts - enables efficient word overlap matching."""

    __tablename__ = "text_vocabulary"
    __table_args__ = (
        UniqueConstraint("text_id", "lemma", "pos", name="uq_tv_text_lemma_pos"),
        Index("ix_tv_text_id", "text_id"),
        Index("ix_tv_lemma", "lemma"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    lemma: Mapped[str] = mapped_column(String(256), index=True)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)


class ProfileTextRead(Base):
    """Tracks which texts a profile has read."""

    __tablename__ = "profile_text_reads"
    __table_args__ = (
        UniqueConstraint("profile_id", "text_id", name="uq_ptr_profile_text"),
        Index("ix_ptr_profile_id", "profile_id"),
        Index("ix_ptr_last_read", "profile_id", "last_read_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    text_id: Mapped[int] = mapped_column(Integer, index=True)
    read_count: Mapped[int] = mapped_column(Integer, default=1)
    first_read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_read_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ProfileTextQueue(Base):
    """Cached text queue for a profile - maintains ordered list of recommended texts."""

    __tablename__ = "profile_text_queue"
    __table_args__ = (
        UniqueConstraint("profile_id", "text_id", name="uq_ptq_profile_text"),
        Index("ix_ptq_profile_rank", "profile_id", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    text_id: Mapped[int] = mapped_column(Integer, index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# SRS Models
# =============================================================================


class Lexeme(Base):
    """User-specific vocabulary entries with SRS tracking."""

    __tablename__ = "lexemes"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "profile_id",
            "lang",
            "lemma",
            "pos",
            name="uq_lexeme_account_profile_lang_lemma_pos",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # User-specific tracking
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )

    # Lexeme data
    lang: Mapped[str] = mapped_column(String(16), index=True)
    lemma: Mapped[str] = mapped_column(String(256), index=True)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)

    # Global word properties
    frequency_rank: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)

    # SRS tracking
    a_click: Mapped[int] = mapped_column(Integer, default=1)
    b_nonclick: Mapped[int] = mapped_column(Integer, default=4)
    stability: Mapped[float] = mapped_column(Float, default=0.2)
    alpha: Mapped[float] = mapped_column(Float, default=1.0)
    beta: Mapped[float] = mapped_column(Float, default=9.0)
    difficulty: Mapped[float] = mapped_column(Float, default=1.0)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    importance_var: Mapped[float] = mapped_column(Float, default=0.3)

    # Interaction tracking
    exposures: Mapped[int] = mapped_column(Integer, default=0)
    clicks: Mapped[int] = mapped_column(Integer, default=0)
    distinct_texts: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_clicked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    next_due_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )
    last_decay_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # Familiarity (computed)
    familiarity: Mapped[Optional[float]] = mapped_column(Float, default=None)
    last_seen: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class LexemeVariant(Base):
    """Lexeme form variants (e.g., simplified/traditional Chinese)."""

    __tablename__ = "lexeme_variants"
    __table_args__ = (
        UniqueConstraint(
            "lexeme_id", "script", "form", name="uq_variant_lexeme_script_form"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("lexemes.id", ondelete="CASCADE"), index=True
    )
    script: Mapped[str] = mapped_column(String(8), index=True)
    form: Mapped[str] = mapped_column(String(256), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class WordEvent(Base):
    """Tracking of word interaction events for analytics and SRS."""

    __tablename__ = "word_events"
    __table_args__ = (
        Index("ix_we_account_ts", "account_id", "ts"),
        Index("ix_we_profile_ts", "profile_id", "ts"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(16))
    count: Mapped[int] = mapped_column(Integer, default=1)
    surface: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    source: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    text_id: Mapped[Optional[int]] = mapped_column(Integer, default=None, index=True)


class LexemeContext(Base):
    """Context in which a lexeme was encountered."""

    __tablename__ = "lexeme_contexts"
    __table_args__ = (
        UniqueConstraint("lexeme_id", "context_hash", name="uq_lexeme_ctx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(
        ForeignKey("lexemes.id", ondelete="CASCADE"), index=True
    )
    context_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# LLM Models
# =============================================================================


class LLMModel(Base):
    """Available LLM models and their capabilities."""

    __tablename__ = "llm_models"
    __table_args__ = (UniqueConstraint("mid", name="uq_llm_models_mid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mid: Mapped[str] = mapped_column(String(128), index=True)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    label: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    family: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    context_window: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    max_output_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    modalities: Mapped[dict] = mapped_column(JSON, default=dict)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    tiers: Mapped[dict] = mapped_column(JSON, default=dict)
    pricing: Mapped[dict] = mapped_column(JSON, default=dict)
    limits: Mapped[dict] = mapped_column(JSON, default=dict)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class GenerationLog(Base):
    """Log of text generation requests and responses."""

    __tablename__ = "generation_logs"
    __table_args__ = (Index("ix_gl_account_created", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("profiles.id", ondelete="SET NULL"), index=True, default=None
    )
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    base_url: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    prompt: Mapped[dict] = mapped_column(JSON, default=dict)
    words: Mapped[dict] = mapped_column(JSON, default=dict)
    level_hint: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    approx_len: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    unit: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class TranslationLog(Base):
    """Log of translation requests and responses."""

    __tablename__ = "translation_logs"
    __table_args__ = (Index("ix_tl_account_created", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="SET NULL"), index=True, default=None
    )
    unit: Mapped[Optional[str]] = mapped_column(String(16), default=None)
    target_lang: Mapped[Optional[str]] = mapped_column(String(8), default=None)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    prompt: Mapped[dict] = mapped_column(JSON, default=dict)
    segments: Mapped[dict] = mapped_column(JSON, default=dict)
    response: Mapped[Optional[str]] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class LLMRequestLog(Base):
    """Comprehensive log of all LLM requests."""

    __tablename__ = "llm_request_logs"
    __table_args__ = (Index("ix_llmrl_account_created", "account_id", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, default=None)
    text_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="SET NULL"), index=True, default=None
    )
    kind: Mapped[str] = mapped_column(String(32))
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    base_url: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    status: Mapped[str] = mapped_column(String(16), default="error")
    request: Mapped[dict] = mapped_column(JSON, default=dict)
    response: Mapped[Optional[str]] = mapped_column(String, default=None)
    error: Mapped[Optional[str]] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class GenerationRetryAttempt(Base):
    """Track retry attempts for failed generation components."""

    __tablename__ = "generation_retry_attempts"
    __table_args__ = (
        Index("ix_genretry_account_text", "account_id", "text_id"),
        Index("ix_genretry_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    failed_components: Mapped[int] = mapped_column(Integer, default=0)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    completed_components: Mapped[int] = mapped_column(Integer, default=0)
    error_details: Mapped[Optional[str]] = mapped_column(String, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )


class TextState(Base):
    """Complete text state JSON - generated during text creation and served to users."""

    __tablename__ = "text_states"
    __table_args__ = (
        UniqueConstraint("text_id", name="uq_text_state_text"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), unique=True, index=True
    )

    # Build status - can be updated incrementally
    status: Mapped[str] = mapped_column(
        String(32), default="building"
    )  # building | ready | failed

    # Components flags
    has_content: Mapped[bool] = mapped_column(Boolean, default=False)
    has_words: Mapped[bool] = mapped_column(Boolean, default=False)
    has_translations: Mapped[bool] = mapped_column(Boolean, default=False)

    # The complete state JSON
    state_data: Mapped[dict] = mapped_column(JSON, default=dict)

    # Metadata
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=None
    )


class ProfileTextState(Base):
    """Text state returned by user after reading - keyed by (text_id, profile_id)."""

    __tablename__ = "profile_text_states"
    __table_args__ = (
        UniqueConstraint("text_id", "profile_id", name="uq_profile_text_state"),
        Index("ix_pts_profile_saved", "profile_id", "saved_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(
        ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True
    )
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("profiles.id", ondelete="CASCADE"), index=True
    )
    account_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)

    # The state data as returned by client (includes interactions)
    state_data: Mapped[dict] = mapped_column(JSON, default=dict)

    # Timestamps
    saved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# Helper Functions
# =============================================================================


def create_sqlite_engine(
    database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False
):
    """Create SQLite engine with proper configuration"""
    from sqlalchemy import create_engine

    engine = create_engine(
        database_url, echo=echo, connect_args={"check_same_thread": False}
    )
    return engine


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
