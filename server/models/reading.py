"""Reading text and vocabulary related models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, Boolean, UniqueConstraint, Index, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column
from server.db import Base


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
