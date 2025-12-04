"""
Startup Service - Handles system initialization on server start.

Responsibilities:
- Create/ensure system account exists
- Configure system account with API key
- Pre-generate texts for configured languages
- Block until minimum texts are ready
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Optional, List

from sqlalchemy.orm import Session

from ..db import open_global_session
from ..account_db import open_account_session
from ..models import Profile, ReadingText, UserModelConfig
from ..config import (
    STARTUP_TEXTS_PER_LANG,
    STARTUP_LANGS,
    STARTUP_TARGET_LANG,
    STARTUP_TIMEOUT_SEC,
)

logger = logging.getLogger(__name__)

# System account config (not in config.py as it's internal)
SYSTEM_ACCOUNT_EMAIL = os.getenv("ARC_SYSTEM_ACCOUNT_EMAIL", "system@arcadia.local")
SYSTEM_API_KEY = os.getenv("ARC_SYSTEM_API_KEY", "")


def _get_or_create_system_account(db: Session) -> int:
    """Get or create the system account. Returns account ID."""
    from server.auth import Account
    from server.auth.security import hash_password
    
    account = db.query(Account).filter(Account.email == SYSTEM_ACCOUNT_EMAIL).first()
    if account:
        logger.info(f"[STARTUP] System account exists: id={account.id}")
        return account.id
    
    # Create system account with "system" tier (unlimited access, not for human use)
    account = Account(
        email=SYSTEM_ACCOUNT_EMAIL,
        password_hash=hash_password("system-no-login"),  # Not used for login
        is_active=True,
        is_verified=True,
        subscription_tier="system",
    )
    db.add(account)
    db.commit()
    db.refresh(account)
    logger.info(f"[STARTUP] Created system account: id={account.id}")
    return account.id


def _ensure_system_profile(db: Session, account_id: int, lang: str, target_lang: str) -> Profile:
    """Ensure system account has a profile for the given language."""
    profile = db.query(Profile).filter(
        Profile.account_id == account_id,
        Profile.lang == lang,
        Profile.target_lang == target_lang
    ).first()
    
    if profile:
        return profile
    
    profile = Profile(
        account_id=account_id,
        lang=lang,
        target_lang=target_lang,
        level_value=5.0,  # Mid-level for diverse texts
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    logger.info(f"[STARTUP] Created system profile: lang={lang}, target={target_lang}")
    return profile


def _configure_system_model(account_db: Session, account_id: int, api_key: str) -> None:
    """Configure the system account with the API key for text generation."""
    if not api_key:
        logger.warning("[STARTUP] No ARC_SYSTEM_API_KEY set - using fallback model resolution")
        return
    
    # Check if model config already exists
    existing = account_db.query(UserModelConfig).filter(
        UserModelConfig.account_id == account_id,
        UserModelConfig.source == "system",
        UserModelConfig.use_for_generation == True
    ).first()
    
    if existing:
        # Update API key if changed
        if existing.api_key != api_key:
            existing.api_key = api_key
            account_db.commit()
            logger.info("[STARTUP] Updated system model API key")
        return
    
    # Create system model config
    model_id = os.getenv("ARC_SYSTEM_MODEL_ID", "anthropic/claude-3-5-sonnet")
    config = UserModelConfig(
        account_id=account_id,
        display_name="System Generation Model",
        provider="openrouter",
        model_id=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        source="system",
        is_editable=False,
        is_key_visible=False,
        use_for_generation=True,
        use_for_word_translation=True,
        use_for_sentence_translation=True,
        priority=1,
        is_active=True,
    )
    account_db.add(config)
    account_db.commit()
    logger.info(f"[STARTUP] Created system model config with model_id={model_id}")


def _count_ready_texts(db: Session, lang: str, target_lang: str) -> int:
    """Count fully ready texts for a language pair."""
    return db.query(ReadingText).filter(
        ReadingText.lang == lang,
        ReadingText.target_lang == target_lang,
        ReadingText.content.isnot(None),
        ReadingText.words_complete == True,
        ReadingText.sentences_complete == True,
    ).count()


def _trigger_generation(account_id: int, lang: str) -> None:
    """Trigger text generation for the system account."""
    from .generation_orchestrator import GenerationOrchestrator
    
    orchestrator = GenerationOrchestrator()
    global_db = open_global_session()
    try:
        orchestrator._start_generation_job(global_db, account_id, lang)
        logger.info(f"[STARTUP] Triggered generation for lang={lang}")
    finally:
        global_db.close()


async def ensure_startup_texts() -> None:
    """
    Main startup function - ensure system account exists and minimum texts are ready.
    
    This function:
    1. Creates/gets system account (always)
    2. Configures API key (if provided)
    3. Triggers generation for each configured language (if API key provided)
    4. Waits until minimum texts are ready (if API key provided)
    """
    # Step 1: Always create system account for admin visibility
    global_db = open_global_session()
    try:
        account_id = _get_or_create_system_account(global_db)
        
        # Create profiles for each configured language
        langs = [l.strip() for l in STARTUP_LANGS.split(",") if l.strip()]
        for lang in langs:
            _ensure_system_profile(global_db, account_id, lang, STARTUP_TARGET_LANG)
    finally:
        global_db.close()
    
    logger.info(f"[STARTUP] System account ready (id={account_id})")
    
    # Step 2: Configure API key if provided
    if SYSTEM_API_KEY:
        account_db = open_account_session(account_id)
        try:
            _configure_system_model(account_db, account_id, SYSTEM_API_KEY)
        finally:
            account_db.close()
    else:
        logger.info("[STARTUP] No ARC_SYSTEM_API_KEY configured - system account created but text generation disabled")
    
    # Step 3: Text pre-generation (only if API key and texts enabled)
    if STARTUP_TEXTS_PER_LANG <= 0:
        logger.info("[STARTUP] Text pre-generation disabled (ARC_STARTUP_TEXTS_PER_LANG=0)")
        return
    
    if not SYSTEM_API_KEY:
        logger.info("[STARTUP] No ARC_SYSTEM_API_KEY set - skipping text pre-generation")
        return
    
    langs = [l.strip() for l in STARTUP_LANGS.split(",") if l.strip()]
    if not langs:
        logger.info("[STARTUP] No languages configured for pre-generation")
        return
    
    logger.info(f"[STARTUP] Ensuring {STARTUP_TEXTS_PER_LANG} text(s) ready for: {langs}")
    
    # Check current state and trigger generation if needed
    needs_generation: List[str] = []
    global_db = open_global_session()
    try:
        for lang in langs:
            count = _count_ready_texts(global_db, lang, STARTUP_TARGET_LANG)
            if count < STARTUP_TEXTS_PER_LANG:
                needs_generation.append(lang)
                logger.info(f"[STARTUP] {lang}: {count}/{STARTUP_TEXTS_PER_LANG} texts ready - triggering generation")
            else:
                logger.info(f"[STARTUP] {lang}: {count}/{STARTUP_TEXTS_PER_LANG} texts ready - OK")
    finally:
        global_db.close()
    
    # Trigger generation for languages that need it
    for lang in needs_generation:
        _trigger_generation(account_id, lang)
    
    # Wait for texts to be ready
    if needs_generation:
        start_time = time.time()
        while True:
            elapsed = time.time() - start_time
            if elapsed > STARTUP_TIMEOUT_SEC:
                logger.warning(f"[STARTUP] Timeout waiting for texts after {elapsed:.0f}s")
                break
            
            all_ready = True
            global_db = open_global_session()
            try:
                for lang in needs_generation:
                    count = _count_ready_texts(global_db, lang, STARTUP_TARGET_LANG)
                    if count < STARTUP_TEXTS_PER_LANG:
                        all_ready = False
                        logger.debug(f"[STARTUP] Waiting for {lang}: {count}/{STARTUP_TEXTS_PER_LANG}")
            finally:
                global_db.close()
            
            if all_ready:
                logger.info(f"[STARTUP] All texts ready after {elapsed:.1f}s")
                break
            
            await asyncio.sleep(2.0)  # Check every 2 seconds
    
    logger.info("[STARTUP] Startup complete")


def get_system_account_id() -> Optional[int]:
    """Get the system account ID if it exists."""
    global_db = open_global_session()
    try:
        from server.auth import Account
        account = global_db.query(Account).filter(Account.email == SYSTEM_ACCOUNT_EMAIL).first()
        return account.id if account else None
    finally:
        global_db.close()
