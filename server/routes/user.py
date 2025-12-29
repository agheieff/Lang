from __future__ import annotations

from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session
import re
import json
import logging

from fastapi import APIRouter, Depends, Request, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.auth import Account  # type: ignore
from server.llm.client import chat_complete

from ..deps import get_current_account as _get_current_account
from ..db import get_db
from ..models import Profile, ProfileTopicPref

logger = logging.getLogger(__name__)

router = APIRouter(tags=["user"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _templates() -> Jinja2Templates:
    try:
        env = Environment(
            loader=FileSystemLoader([str(TEMPLATES_DIR)]),
            autoescape=select_autoescape(["html", "xml"]),
        )
        return Jinja2Templates(env=env)
    except Exception:
        return Jinja2Templates(directory=str(TEMPLATES_DIR))


# HTML Page Endpoints
@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse(request, "pages/profile.html", {"title": "Profile"})


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    context = {
        "title": "Settings",
        "current_profile": profile or {},
    }

    return t.TemplateResponse(request, "pages/settings.html", context)


@router.get("/stats", response_class=HTMLResponse)
def stats_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    from ..models import Language

    default_lang = db.query(Language).filter(Language.code == "es").first()
    lang_code = (
        profile.lang if profile else (default_lang.code if default_lang else "en")
    )

    return t.TemplateResponse(
        request,
        "pages/stats.html",
        {"title": "Statistics", "lang": lang_code},
    )


@router.get("/words", response_class=HTMLResponse)
def words_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    from ..models import Language

    default_lang = db.query(Language).filter(Language.code == "es").first()
    lang_code = (
        profile.lang if profile else (default_lang.code if default_lang else "en")
    )

    return t.TemplateResponse(
        request,
        "pages/words.html",
        {"title": "My Words", "lang": lang_code},
    )


# API Endpoints (merged from profile.py and settings.py)


@router.get("/me/profile")
def get_me_profile(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get current user profile (alias for compatibility with templates)."""
    return _profile_data_to_dict(account, db)


@router.post("/me/profile")
def post_me_profile(
    profile_data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Create or update user profile (alias for compatibility with templates)."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    if profile:
        for key, value in profile_data.items():
            if hasattr(profile, key) and key not in ["id", "account_id", "created_at"]:
                setattr(profile, key, value)
        db.commit()
        db.refresh(profile)
    else:
        profile = Profile(account_id=account.id, **profile_data)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return _profile_data_to_dict(account, db)


@router.get("/profile/api")
def get_profile_endpoint(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get current user profile."""
    return _profile_data_to_dict(account, db)


@router.get("/languages")
def get_languages(db: Session = Depends(get_db)):
    """Get supported languages for profile creation."""
    from ..models import Language

    languages = db.query(Language).filter(Language.is_enabled == True).all()

    return {
        "languages": [
            {"code": lang.code, "name": lang.display_name} for lang in languages
        ]
    }


def _profile_data_to_dict(account: Account, db: Session) -> dict:
    """Shared function to get profile data."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")

    return {
        "id": profile.id,
        "account_id": profile.account_id,
        "lang": profile.lang,
        "target_lang": profile.target_lang,
        "level_value": profile.level_value,
        "text_length": profile.text_length,
        "preferred_script": getattr(profile, "preferred_script", None),
        "created_at": profile.created_at,
    }


@router.post("/profile/api")
def create_or_update_profile(
    profile_data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Create or update user profile."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    if profile:
        # Update existing
        for key, value in profile_data.items():
            if hasattr(profile, key) and key not in ["id", "account_id", "created_at"]:
                setattr(profile, key, value)
        db.commit()
        db.refresh(profile)
    else:
        # Create new
        profile = Profile(account_id=account.id, **profile_data)
        db.add(profile)
        db.commit()
        db.refresh(profile)

    return _profile_data_to_dict(account, db)


@router.put("/profile/api")
def update_profile(
    profile_data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Update existing profile."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")

    for key, value in profile_data.items():
        if hasattr(profile, key) and key not in ["id", "account_id", "created_at"]:
            setattr(profile, key, value)

    db.commit()
    db.refresh(profile)

    return _profile_data_to_dict(account, db)


# Preferences Management (LLM-based updates)
@router.post("/settings/preferences/update")
async def update_preferences_via_llm(
    data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Update preferences using LLM to parse natural language request."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return {"success": False, "error": "Profile not found"}

    message = data.get("message", "").strip()
    if not message:
        return {"success": False, "error": "Message is required"}

    try:
        setattr(profile, "preferences_updating", True)
        db.commit()

        from ..utils.nlp import extract_text_from_llm_response

        prompt = f"""The user wants to update their reading preferences. Their message is:

"{message}"

Current settings:
- Language: {profile.lang}
- Target language: {profile.target_lang}
- Current text length: {profile.text_length or "default"}
- Level: {profile.level_value:.1f}

Parse their request and extract:
1. Text length preference (number, or null if not mentioned)
2. Topic preferences as a JSON object mapping topics to weights (1.0 = default, >1.0 = prefer, <1.0 = avoid)
   Topics: {["fiction", "news", "science", "technology", "history", "daily_life", "culture", "sports", "business"]}
3. Any other specific settings

Return ONLY a JSON object in this exact format:
{{
  "text_length": <number or null>,
  "topics": {{
    "fiction": <number>,
    "news": <number>,
    "science": <number>,
    "technology": <number>,
    "history": <number>,
    "daily_life": <number>,
    "culture": <number>,
    "sports": <number>,
    "business": <number>
  }}
}}"""

        response = chat_complete(messages=[{"role": "user", "content": prompt}])

        cleaned = extract_text_from_llm_response(response)
        match = re.search(r"\{[^{}]*\}", cleaned)

        if not match:
            raise ValueError("No JSON found in LLM response")

        preferences = json.loads(match.group())

        text_length = preferences.get("text_length")
        if text_length and isinstance(text_length, (int, float)):
            profile.text_length = max(50, min(2000, int(text_length)))

        topic_weights = preferences.get("topics", {})
        if topic_weights and isinstance(topic_weights, dict):
            for topic, weight in topic_weights.items():
                if topic in [
                    "fiction",
                    "news",
                    "science",
                    "technology",
                    "history",
                    "daily_life",
                    "culture",
                    "sports",
                    "business",
                ]:
                    if isinstance(weight, (int, float)):
                        existing = (
                            db.query(ProfileTopicPref)
                            .filter(
                                ProfileTopicPref.profile_id == profile.id,
                                ProfileTopicPref.topic == topic,
                            )
                            .first()
                        )
                        if existing:
                            existing.weight = float(weight)
                        else:
                            new_pref = ProfileTopicPref(
                                profile_id=profile.id, topic=topic, weight=float(weight)
                            )
                            db.add(new_pref)

        setattr(profile, "preferences_updating", False)
        db.commit()

        return {
            "success": True,
            "status": "done",
            "message": "Preferences updated successfully",
            "text_length": profile.text_length,
        }

    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        if "profile" in locals():
            setattr(profile, "preferences_updating", False)
            db.commit()
        return {
            "success": False,
            "error": f"Failed to update preferences: {str(e)}",
        }


@router.get("/settings/preferences/status")
def get_preferences_status(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Check if preferences update is in progress."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return {"updating": False}

    return {
        "updating": bool(profile.preferences_updating),
        "text_length": profile.text_length,
    }


@router.get("/settings/topics")
def get_current_topics(
    lang: Optional[str] = None,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get current topic weights for profile."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return {"topics": []}

    topic_prefs = (
        db.query(ProfileTopicPref)
        .filter(ProfileTopicPref.profile_id == profile.id)
        .all()
    )

    topic_weights = {tp.topic: tp.weight for tp in topic_prefs}

    all_topics = [
        {"topic": t, "weight": topic_weights.get(t, 1.0)}
        for t in [
            "fiction",
            "news",
            "science",
            "technology",
            "history",
            "daily_life",
            "culture",
            "sports",
            "business",
        ]
    ]

    return {"topics": all_topics}


@router.get("/settings/tier")
def get_tier_info(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get subscription tier and usage info."""
    from ..models import UsageTracking
    from datetime import datetime, timezone
    from ..config import (
        FREE_TIER_CHAR_LIMIT,
        FREE_TIER_TEXT_LIMIT,
        TIER_SPENDING_LIMITS,
        SubscriptionTier,
    )

    tier_str = str(account.subscription_tier or "Free")

    # Convert string to enum for dictionary lookup
    try:
        tier = SubscriptionTier(tier_str)
    except ValueError:
        tier = SubscriptionTier.FREE

    month_start = datetime.now(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )

    usage = (
        db.query(UsageTracking)
        .filter(
            UsageTracking.account_id == account.id,
            UsageTracking.period_start >= month_start,
        )
        .first()
    )

    if usage:
        chars_used = usage.chars_generated or 0
        texts_used = usage.texts_generated or 0
    else:
        chars_used = 0
        texts_used = 0

    if tier == SubscriptionTier.FREE:
        char_limit = FREE_TIER_CHAR_LIMIT
        text_limit = FREE_TIER_TEXT_LIMIT
    else:
        char_limit = TIER_SPENDING_LIMITS.get(tier, 0)
        text_limit = None

    return {
        "tier": tier_str,
        "chars_used": chars_used,
        "chars_limit": char_limit,
        "texts_used": texts_used,
        "text_limit": text_limit,
        "chars_percentage": (chars_used / char_limit * 100) if char_limit else 0,
    }
