"""
Consolidated SQLAlchemy models for reading texts, SRS, LLM, and authentication.
All database models in one module for better organization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, JSON, Float, Column, Text
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
    
    # Core fields from original schema
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=True, nullable=False) 
    # Unified tier for both access control and features: Free|Standard|Pro|Pro+|BYOK|admin|system
    subscription_tier = Column(String(50), default="Free", nullable=False)
    
    # OpenRouter per-user key management (for paid tiers)
    openrouter_key_encrypted = Column(Text, nullable=True)  # Fernet-encrypted API key
    openrouter_key_id = Column(String(128), nullable=True)  # OpenRouter key hash/identifier for management
    
    # Extended fields - apps can add their own here
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
    __table_args__ = (
        UniqueConstraint("code", name="uq_language_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # e.g., 'es', 'zh', 'en'
    name: Mapped[str] = mapped_column(String(64))  # e.g., 'Spanish', 'Chinese', 'English'
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Profile(Base):
    """User language learning profiles."""
    __tablename__ = "profiles"
    __table_args__ = (
        UniqueConstraint("account_id", "lang", "target_lang", name="uq_profile_account_lang_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Per-account DB: store numeric account id; no cross-DB FK to accounts
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    # Store language codes directly (not modifiable by user after creation)
    lang: Mapped[str] = mapped_column(String(16), index=True)  # language being learned (e.g., 'es', 'zh')
    target_lang: Mapped[str] = mapped_column(String(16), index=True)  # user's native/reference language (e.g., 'en')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # User's level in this language
    level_value: Mapped[float] = mapped_column(Float, default=0.0)  # continuous estimate (e.g., 0..10)
    level_var: Mapped[float] = mapped_column(Float, default=1.0)    # uncertainty / learning-rate proxy
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)  # e.g., HSK3, A2, etc.
    # For Chinese, user's preferred script: Hans or Hant
    preferred_script: Mapped[Optional[str]] = mapped_column(String(8), default=None)
    # Profile metadata (stored in JSON for flexibility)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)  # learning preferences, topics, etc.
    # User-configurable reading length hint (words or chars per language)
    text_length: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    # Free-form preferences/topics for reading generation
    text_preferences: Mapped[Optional[str]] = mapped_column(String, default=None)
    # Current text being read (references global DB - no FK constraint)
    current_text_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, default=None)
    # Pool-based selection preferences
    ci_preference: Mapped[float] = mapped_column(Float, default=0.92)  # Target comprehension (0.85-0.98)
    topic_weights: Mapped[dict] = mapped_column(JSON, default=dict)  # Populated from config.DEFAULT_TOPIC_WEIGHTS
    # Async preferences update tracking
    preferences_updating: Mapped[bool] = mapped_column(Boolean, default=False)
    # Re-read settings: None = never show again, 0 = always allow, N = cooldown in days
    reread_cooldown_days: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)


class ReadingText(Base):
    """Global text storage - texts are shared across profiles with matching lang/target_lang."""
    __tablename__ = "reading_texts"
    __table_args__ = (
        Index("ix_rt_lang_target", "lang", "target_lang"),
        Index("ix_rt_ready", "lang", "target_lang", "words_complete", "sentences_complete"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Who requested generation (nullable for imported texts)
    generated_for_account_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    # Source and target languages
    lang: Mapped[str] = mapped_column(String(16), index=True)  # Source language (e.g., 'zh')
    target_lang: Mapped[str] = mapped_column(String(16), index=True)  # Translation target (e.g., 'en')
    # Content
    content: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    source: Mapped[Optional[str]] = mapped_column(String(16), default="llm")  # llm|manual|import
    # Generation lifecycle timestamps
    request_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    # Pool-based selection fields
    ci_target: Mapped[Optional[float]] = mapped_column(Float, default=None)  # Target comprehension (0.85-0.98)
    topic: Mapped[Optional[str]] = mapped_column(String(64), default=None)  # Topic category (can be comma-separated)
    difficulty_estimate: Mapped[Optional[float]] = mapped_column(Float, default=None)  # Estimated difficulty (0-1)
    # Generation completion flags - text is ready when both are True
    words_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    sentences_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    # Retry tracking for failed translations
    translation_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_translation_attempt: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), default=None)
    # Vocabulary stats for quick filtering
    word_count: Mapped[int] = mapped_column(Integer, default=0)  # Total words
    unique_lemma_count: Mapped[int] = mapped_column(Integer, default=0)  # Distinct lemmas
    # Generation prompt data (for reproducibility and analysis)
    prompt_words: Mapped[dict] = mapped_column(JSON, default=dict)  # Words used in generation prompt
    prompt_level_hint: Mapped[Optional[str]] = mapped_column(String(128), default=None)  # Level hint used

    @property
    def is_ready(self) -> bool:
        """Check if text is ready for reading (has content and complete translations)."""
        return bool(self.content and self.words_complete and self.sentences_complete)
    
    @classmethod
    def ready_filter(cls):
        """Get filter conditions for ready texts - use with query.filter()."""
        return (
            cls.content.isnot(None),
            cls.content != "",
            cls.words_complete == True,
            cls.sentences_complete == True,
        )


class ReadingTextTranslation(Base):
    """Global translations storage - translations are shared across users."""
    __tablename__ = "reading_text_translations"
    __table_args__ = (
        UniqueConstraint(
            "text_id",
            "target_lang",
            "unit",
            "segment_index",
            "span_start",
            "span_end",
            name="uq_rtt_unique",
        ),
        Index("ix_rtt_text_target", "text_id", "target_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    target_lang: Mapped[str] = mapped_column(String(8), index=True)
    unit: Mapped[str] = mapped_column(String(16))  # sentence|paragraph|text
    segment_index: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    span_start: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    span_end: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    source_text: Mapped[str] = mapped_column(String)
    translated_text: Mapped[str] = mapped_column(String)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


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
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    target_lang: Mapped[str] = mapped_column(String(16), index=True)  # Translation target language
    lang: Mapped[str] = mapped_column(String(16))  # Source language
    surface: Mapped[str] = mapped_column(String(256))
    lemma: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    pinyin: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    translation: Mapped[Optional[str]] = mapped_column(String, default=None)
    lemma_translation: Mapped[Optional[str]] = mapped_column(String, default=None)
    grammar: Mapped[dict] = mapped_column(JSON, default=dict)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ReadingLookup(Base):
    """User word lookups while reading."""
    __tablename__ = "reading_lookups"
    __table_args__ = (
        UniqueConstraint("account_id", "text_id", "target_lang", "span_start", "span_end", name="uq_reading_lookup_span"),
        Index("ix_rl_text_id", "text_id"),
        Index("ix_rl_account_text", "account_id", "text_id"),
        Index("ix_rl_text_target", "text_id", "target_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    lang: Mapped[str] = mapped_column(String(16))
    target_lang: Mapped[str] = mapped_column(String(8))
    surface: Mapped[str] = mapped_column(String(256))
    lemma: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    span_start: Mapped[int] = mapped_column(Integer)
    span_end: Mapped[int] = mapped_column(Integer)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    translations: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TextVocabulary(Base):
    """Vocabulary index for texts - enables efficient word overlap matching."""
    __tablename__ = "text_vocabulary"
    __table_args__ = (
        UniqueConstraint("text_id", "lemma", "pos", name="uq_tv_text_lemma_pos"),
        Index("ix_tv_text_id", "text_id"),
        Index("ix_tv_lemma", "lemma"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    lemma: Mapped[str] = mapped_column(String(256), index=True)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)


class ProfileTextRead(Base):
    """Tracks which texts a profile has read (per-account DB)."""
    __tablename__ = "profile_text_reads"
    __table_args__ = (
        UniqueConstraint("profile_id", "text_id", name="uq_ptr_profile_text"),
        Index("ix_ptr_profile_id", "profile_id"),
        Index("ix_ptr_last_read", "profile_id", "last_read_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    text_id: Mapped[int] = mapped_column(Integer, index=True)  # References global DB (no FK)
    read_count: Mapped[int] = mapped_column(Integer, default=1)
    first_read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProfileTextQueue(Base):
    """Cached text queue for a profile (per-account DB)."""
    __tablename__ = "profile_text_queue"
    __table_args__ = (
        UniqueConstraint("profile_id", "text_id", name="uq_ptq_profile_text"),
        Index("ix_ptq_profile_rank", "profile_id", "rank"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    text_id: Mapped[int] = mapped_column(Integer, index=True)  # References global DB (no FK)
    rank: Mapped[int] = mapped_column(Integer)  # 1-10, lower = better match
    score: Mapped[float] = mapped_column(Float)  # Matching score, lower = better
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProfilePref(Base):
    """Profile preferences storage."""
    __tablename__ = "profile_prefs"
    __table_args__ = (
        UniqueConstraint("profile_id", name="uq_profile_pref_profile_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


# =============================================================================
# SRS Models
# =============================================================================

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


# =============================================================================
# LLM Models
# =============================================================================

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


# Helper functions
def create_sqlite_engine(database_url: str = "sqlite:///arcadia_auth.db", echo: bool = False):
    """Create SQLite engine with proper configuration"""
    from sqlalchemy import create_engine
    engine = create_engine(database_url, echo=echo, connect_args={"check_same_thread": False})
    return engine


def create_tables(engine):
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
