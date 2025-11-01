from __future__ import annotations

from pathlib import Path
import os
from typing import Generator, Optional, Dict

from fastapi import Request
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from .db import Base, DATA_DIR, _configure_sqlite_engine

# Per-account DB routing
_ACCOUNT_ENGINES: Dict[int, Engine] = {}
_ACCOUNTS_DIR = DATA_DIR / "accounts"
_ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)

# Tables to provision in per-account DBs (user-scoped data)
PER_ACCOUNT_TABLES = {
    # learning/profile + reading
    "profiles",
    "profile_prefs",
    "reading_texts",
    "cards",
    # lexicon + user stats
    "lexemes",
    "lexeme_variants",
    "lexeme_info",
    "user_lexemes",
    "user_lexeme_contexts",
    "word_events",
    # activity/logs per reading
    "generation_logs",
    "reading_text_translations",
    "reading_word_glosses",
    "translation_logs",
    "reading_lookups",
    # readiness override
    "next_ready_overrides",
    # request logs
    "llm_request_logs",
    # curated/user-specific lists
    "language_word_lists",
}


def _account_db_path(account_id: int) -> Path:
    return _ACCOUNTS_DIR / f"{int(account_id)}.db"


def _get_or_create_account_engine(account_id: int) -> Engine:
    aid = int(account_id)
    e = _ACCOUNT_ENGINES.get(aid)
    if e is not None:
        return e
    path = _account_db_path(aid)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(
        f"sqlite:///{path}",
        connect_args={
            "check_same_thread": False,
            "timeout": 30.0,
        },
    )
    _configure_sqlite_engine(eng)
    _ACCOUNT_ENGINES[aid] = eng
    return eng


def _ensure_account_tables(account_engine: Engine) -> None:
    """Create all account-specific tables from scratch"""
    try:
        # Import models to register them with Base
        from . import models  # noqa: F401

        # Create only the tables needed for account databases
        with account_engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if table.name in PER_ACCOUNT_TABLES:
                    table.create(bind=conn, checkfirst=True)
    except Exception as e:
        # Log but don't fail request
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to ensure account tables: {e}")


def _resolve_account_id(request: Optional[Request]) -> Optional[int]:
    if request is None:
        return None
    # Prefer middleware-populated user
    try:
        user = getattr(request.state, "user", None)
        if user is not None:
            # Support object with .id or dict with ['id']
            if hasattr(user, "id"):
                return int(getattr(user, "id"))
            if isinstance(user, dict) and ("id" in user):
                return int(user["id"])  # type: ignore[arg-type]
    except Exception:
        pass
    # Fallback to Authorization bearer token (decode only)
    try:
        from server.auth import decode_token  # type: ignore
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1]
            secret = os.getenv("ARC_LANG_JWT_SECRET", "dev-secret-change")
            data = decode_token(token, secret, ["HS256"])  # type: ignore
            sub = (data or {}).get("sub")
            if sub is not None:
                return int(sub)
    except Exception:
        pass
    return None


def get_db(request: Request) -> Generator[Session, None, None]:
    """FastAPI dependency: yields a per-account Session when authenticated,
    otherwise falls back to the global DB session.
    """
    from .db import GlobalSessionLocal

    aid = _resolve_account_id(request)
    if aid is None:
        db = GlobalSessionLocal()
        try:
            yield db
        finally:
            db.close()
        return
    eng = _get_or_create_account_engine(aid)
    _ensure_account_tables(eng)
    Local = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    db = Local()
    try:
        yield db
    finally:
        db.close()


def open_account_session(account_id: int) -> Session:
    """Create and return a dedicated per-account Session (caller must close)."""
    eng = _get_or_create_account_engine(int(account_id))
    _ensure_account_tables(eng)
    Local = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return Local()
