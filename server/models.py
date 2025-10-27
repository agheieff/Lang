from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, UniqueConstraint, JSON, Float, Boolean, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

# Import Account model from arcadia_auth to avoid duplication
from arcadia_auth import Account


class Language(Base):
    __tablename__ = "languages"
    __table_args__ = (
        UniqueConstraint("code", name="uq_language_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(16), unique=True, index=True)  # e.g., 'es', 'zh', 'en'
    name: Mapped[str] = mapped_column(String(64))  # e.g., 'Spanish', 'Chinese', 'English'
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Profile(Base):
    __tablename__ = "profiles"  # Using existing table
    __table_args__ = (
        UniqueConstraint("account_id", "lang", "target_lang", name="uq_profile_account_lang_target"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Per-account DB: store numeric account id; no cross-DB FK to accounts
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    # Store language codes directly (not modifiable by user after creation)
    lang: Mapped[str] = mapped_column(String(16), index=True)  # language being learned (e.g., 'es', 'zh')
    target_lang: Mapped[str] = mapped_column(String(16), index=True)  # user's native/reference language (e.g., 'en')
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
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

    # Relationship to Account (global DB) is not declared to avoid cross-DB FK
    # account: Mapped["Account"] = relationship("Account", foreign_keys=[account_id])


class ReadingText(Base):
    __tablename__ = "reading_texts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    lang: Mapped[str] = mapped_column(String(16), index=True)
    content: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source: Mapped[Optional[str]] = mapped_column(String(16), default="llm")  # llm|manual
    # Read tracking
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


# Placeholder for future SRS tables (not implemented yet)
class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    head: Mapped[str] = mapped_column(String(256))
    lang: Mapped[str] = mapped_column(String(16), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SubscriptionTier(Base):
    __tablename__ = "subscription_tiers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(255), default=None)


class ProfilePref(Base):
    __tablename__ = "profile_prefs"
    __table_args__ = (
        UniqueConstraint("profile_id", name="uq_profile_pref_profile_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    data: Mapped[dict] = mapped_column(JSON, default=dict)


class Lexeme(Base):
    __tablename__ = "lexemes"
    __table_args__ = (
        UniqueConstraint("lang", "lemma", "pos", name="uq_lexeme_lang_lemma_pos"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lang: Mapped[str] = mapped_column(String(16), index=True)
    lemma: Mapped[str] = mapped_column(String(256), index=True)
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LexemeVariant(Base):
    __tablename__ = "lexeme_variants"
    __table_args__ = (
        UniqueConstraint("script", "form", name="uq_variant_script_form"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    script: Mapped[str] = mapped_column(String(8), index=True)  # Hans | Hant
    form: Mapped[str] = mapped_column(String(256), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserLexeme(Base):
    __tablename__ = "user_lexemes"
    __table_args__ = (
        UniqueConstraint("account_id", "profile_id", "lexeme_id", name="uq_account_profile_lexeme"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)

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
    first_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    last_clicked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    next_due_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    last_decay_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WordEvent(Base):
    __tablename__ = "word_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(16))  # exposure|click|assign|hover
    count: Mapped[int] = mapped_column(Integer, default=1)
    surface: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    source: Mapped[Optional[str]] = mapped_column(String(16), default=None)  # llm|manual|unknown
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    text_id: Mapped[Optional[int]] = mapped_column(Integer, default=None, index=True)


class UserLexemeContext(Base):
    __tablename__ = "user_lexeme_contexts"
    __table_args__ = (
        UniqueConstraint("user_lexeme_id", "context_hash", name="uq_ul_ctx"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_lexeme_id: Mapped[int] = mapped_column(ForeignKey("user_lexemes.id", ondelete="CASCADE"), index=True)
    context_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LexemeInfo(Base):
    __tablename__ = "lexeme_info"
    __table_args__ = (
        UniqueConstraint("lexeme_id", name="uq_lexeme_info_lexeme_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    freq_rank: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    freq_score: Mapped[Optional[float]] = mapped_column(Float, default=None)
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)  # e.g., HSK1..HSK6
    source: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    tags: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GenerationLog(Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReadingTextTranslation(Base):
    __tablename__ = "reading_text_translations"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "text_id",
            "target_lang",
            "unit",
            "segment_index",
            "span_start",
            "span_end",
            name="uq_rtt_unique",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
    unit: Mapped[str] = mapped_column(String(16))  # sentence|paragraph|text
    target_lang: Mapped[str] = mapped_column(String(8))
    segment_index: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    span_start: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    span_end: Mapped[Optional[int]] = mapped_column(Integer, default=None, nullable=True)
    source_text: Mapped[str] = mapped_column(String)
    translated_text: Mapped[str] = mapped_column(String)
    provider: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    model: Mapped[Optional[str]] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReadingWordGloss(Base):
    __tablename__ = "reading_word_glosses"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "text_id",
            "span_start",
            "span_end",
            name="uq_rwg_account_text_span",
        ),
        Index("ix_rwg_text_id", "text_id"),
        Index("ix_rwg_account_text", "account_id", "text_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, index=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("reading_texts.id", ondelete="CASCADE"), index=True)
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TranslationLog(Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReadingLookup(Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LLMModel(Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LanguageWordList(Base):
    __tablename__ = "language_word_lists"
    __table_args__ = (
        UniqueConstraint("lang", "list_name", "word", name="uq_lang_list_word"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    lang: Mapped[str] = mapped_column(String(16), index=True)  # e.g., 'es', 'zh'
    list_name: Mapped[str] = mapped_column(String(64), index=True)  # e.g., 'core_1000', 'hsk1', 'a1'
    word: Mapped[str] = mapped_column(String(256), index=True)  # the word/lemma
    pos: Mapped[Optional[str]] = mapped_column(String(32), default=None)
    frequency_rank: Mapped[Optional[int]] = mapped_column(Integer, default=None)  # global frequency rank
    frequency_score: Mapped[Optional[float]] = mapped_column(Float, default=None)  # frequency score (0-1)
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)  # e.g., 'HSK1', 'A2'
    category: Mapped[Optional[str]] = mapped_column(String(64), default=None)  # semantic category
    tags: Mapped[dict] = mapped_column(JSON, default=dict)  # additional metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LLMRequestLog(Base):
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
