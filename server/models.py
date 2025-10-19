from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, Integer, DateTime, ForeignKey, UniqueConstraint, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    subscription_tier: Mapped[str] = mapped_column(String(32), default="free")  # free|pro|enterprise

    profiles: Mapped[list[Profile]] = relationship("Profile", back_populates="user", cascade="all, delete-orphan")


class Profile(Base):
    __tablename__ = "profiles"
    __table_args__ = (
        UniqueConstraint("user_id", "lang", name="uq_profile_user_lang"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    lang: Mapped[str] = mapped_column(String(16), index=True)  # e.g., 'es', 'zh-Hans'
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # User's level in this language
    level_value: Mapped[float] = mapped_column(Float, default=0.0)  # continuous estimate (e.g., 0..10)
    level_var: Mapped[float] = mapped_column(Float, default=1.0)    # uncertainty / learning-rate proxy
    level_code: Mapped[Optional[str]] = mapped_column(String(32), default=None)  # e.g., HSK3, A2, etc.

    user: Mapped[User] = relationship("User", back_populates="profiles")


# Placeholder for future SRS tables (not implemented yet)
class Card(Base):
    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
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
        UniqueConstraint("user_id", "profile_id", "lexeme_id", name="uq_user_profile_lexeme"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)

    a_click: Mapped[int] = mapped_column(Integer, default=1)
    b_nonclick: Mapped[int] = mapped_column(Integer, default=4)
    stability: Mapped[float] = mapped_column(Float, default=0.2)  # 0..1
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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class WordEvent(Base):
    __tablename__ = "word_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("profiles.id", ondelete="CASCADE"), index=True)
    lexeme_id: Mapped[int] = mapped_column(ForeignKey("lexemes.id", ondelete="CASCADE"), index=True)
    event_type: Mapped[str] = mapped_column(String(16))  # exposure|click|assign|hover
    count: Mapped[int] = mapped_column(Integer, default=1)
    surface: Mapped[Optional[str]] = mapped_column(String(256), default=None)
    context_hash: Mapped[Optional[str]] = mapped_column(String(64), default=None)
    source: Mapped[Optional[str]] = mapped_column(String(16), default=None)  # llm|manual|unknown
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


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
