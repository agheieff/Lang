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
from server.config import LANG_INFO

from ..deps import get_current_account as _get_current_account
from ..db import get_db, db_transaction
from ..models import Profile, ProfileTopicPref

logger = logging.getLogger(__name__)

router = APIRouter(tags=["user"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _get_lang_info(lang: str) -> dict:
    """Get language info with fallback for unknown codes."""
    info = LANG_INFO.get(lang)
    if not info:
        if lang.startswith("zh-"):
            info = {"flag": "🌏", "name": f"Chinese ({lang.split('-')[1].upper()})"}
        elif lang == "zh":
            info = {"flag": "🇨🇳", "name": "Chinese"}
        else:
            info = {"flag": "🌐", "name": lang.upper()}
    return info


def _get_active_profile(request: Request, account: Account, db: Session) -> Optional[Profile]:
    """Get active profile from cookie, falling back to first profile."""
    active_profile_id = request.cookies.get("active_profile_id")
    if active_profile_id:
        try:
            return db.query(Profile).filter(
                Profile.id == int(active_profile_id),
                Profile.account_id == account.id
            ).first()
        except ValueError:
            pass
    return db.query(Profile).filter(Profile.account_id == account.id).first()


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
def profile_page(request: Request, account: Account = Depends(_get_current_account), db: Session = Depends(get_db)):
    t = _templates()
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    return t.TemplateResponse(request, "pages/profile.html", {
        "title": "Profile",
        "current_profile": profile
    })


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = _get_active_profile(request, account, db)
    return t.TemplateResponse(request, "pages/settings.html", {
        "title": "Settings",
        "current_profile": profile
    })


@router.get("/stats", response_class=HTMLResponse)
def stats_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = _get_active_profile(request, account, db)

    from ..models import Language
    default_lang = db.query(Language).filter(Language.code == "es").first()
    lang_code = profile.lang if profile else (default_lang.code if default_lang else "en")

    return t.TemplateResponse(request, "pages/stats.html", {"title": "Statistics", "lang": lang_code})


@router.get("/words", response_class=HTMLResponse)
def words_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    t = _templates()
    profile = _get_active_profile(request, account, db)

    from ..models import Language
    default_lang = db.query(Language).filter(Language.code == "es").first()
    lang_code = profile.lang if profile else (default_lang.code if default_lang else "en")

    return t.TemplateResponse(request, "pages/words.html", {"title": "My Words", "lang": lang_code})


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

    with db_transaction(db):
        if profile:
            for key, value in profile_data.items():
                if hasattr(profile, key) and key not in ["id", "account_id", "created_at"]:
                    setattr(profile, key, value)
            db.refresh(profile)
        else:
            profile = Profile(account_id=account.id, **profile_data)
            db.add(profile)
            db.refresh(profile)

    return _profile_data_to_dict(account, db)


@router.get("/profile/api")
def get_profile_endpoint(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get current user profile."""
    profile = _get_active_profile(request, account, db)
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
    # Check for existing profile with same lang and target_lang
    profile = db.query(Profile).filter(
        Profile.account_id == account.id,
        Profile.lang == profile_data.get("lang"),
        Profile.target_lang == profile_data.get("target_lang")
    ).first()

    if profile:
        # Update existing profile for this language pair
        for key, value in profile_data.items():
            if hasattr(profile, key) and key not in ["id", "account_id", "created_at", "lang", "target_lang"]:
                setattr(profile, key, value)
        db.commit()
        db.refresh(profile)
    else:
        # Create new profile for this language pair
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


@router.get("/settings/topics", response_class=HTMLResponse)
def get_current_topics(
    lang: Optional[str] = None,
    request: Request = None,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get current topic weights for profile as HTML fragment."""
    profile = _get_active_profile(request, account, db) if request else db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        return '<p class="text-gray-500 text-sm">No profile found.</p>'

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

    # Helper function to get color classes for weight
    def get_weight_color(weight: float) -> str:
        """Get Tailwind color class for weight."""
        if weight <= 0.5:
            return "bg-gray-100 text-gray-700 border-gray-300 hover:bg-gray-200"
        elif weight == 1.0:
            return "bg-blue-50 text-blue-700 border-blue-300 hover:bg-blue-100"
        elif weight == 2.0:
            return "bg-purple-50 text-purple-700 border-purple-300 hover:bg-purple-100"
        else:  # 3.0
            return "bg-yellow-50 text-yellow-700 border-yellow-300 hover:bg-yellow-100"

    topics_html = " ".join([
        f'<button data-topic="{t["topic"]}" '
        f'class="{get_weight_color(t["weight"])} inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border cursor-pointer hover:shadow-md transition-all duration-200 select-none" '
        f'title="Left-click to increase, right-click to decrease">'
        f'{t["topic"].replace("_", " ").title()} {t["weight"]}x</button>'
        for t in all_topics
    ])

    return f'<div class="flex flex-wrap gap-2">{topics_html}</div>'


@router.post("/settings/topics/update", response_class=HTMLResponse)
async def update_topic_weight(
    request: Request,
    topic: str = Body(..., embed=True),
    direction: str = Body(..., embed=True),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Update a single topic weight for the current profile.

    Args:
        topic: Topic name (e.g., "science", "fiction")
        direction: "increase" or "decrease"

    Returns:
        Updated topic weight as HTML fragment
    """
    profile = _get_active_profile(request, account, db)
    if not profile:
        return '<p class="text-red-500 text-sm">No profile found.</p>'

    # Validate topic
    from ..services.recommendation import AVAILABLE_TOPICS
    if topic not in AVAILABLE_TOPICS:
        return '<p class="text-red-500 text-sm">Invalid topic.</p>'

    # Get or create preference
    pref = db.query(ProfileTopicPref).filter(
        ProfileTopicPref.profile_id == profile.id,
        ProfileTopicPref.topic == topic
    ).first()

    # Weight cycle: 0.5 → 1.0 → 2.0 → 3.0 → 0.5
    WEIGHT_CYCLE = [0.5, 1.0, 2.0, 3.0]

    if pref:
        current_idx = WEIGHT_CYCLE.index(pref.weight) if pref.weight in WEIGHT_CYCLE else 1
    else:
        current_idx = 1  # Default to 1.0
        pref = ProfileTopicPref(
            profile_id=profile.id,
            topic=topic,
            weight=1.0
        )
        db.add(pref)

    # Calculate new weight
    if direction == "increase":
        new_idx = (current_idx + 1) % len(WEIGHT_CYCLE)
    else:  # decrease
        new_idx = (current_idx - 1) % len(WEIGHT_CYCLE)

    pref.weight = WEIGHT_CYCLE[new_idx]
    db.commit()

    # Return updated topics HTML fragment
    # Reuse the existing get_current_topics logic
    return get_current_topics(
        lang=profile.lang,
        request=request,
        account=account,
        db=db
    )


@router.get("/settings/tier", response_class=HTMLResponse)
def get_tier_info(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get subscription tier and usage info as HTML fragment."""
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

    chars_percentage = (chars_used / char_limit * 100) if char_limit else 0

    # Format values
    chars_limit_str = f"{char_limit:,}" if char_limit else "∞"
    texts_limit_str = str(text_limit) if text_limit else "∞"

    # Return HTML fragment
    return f"""
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <span class="text-sm font-medium text-gray-700">Subscription Tier</span>
        <span class="px-3 py-1 rounded-full text-sm font-medium
              {'bg-purple-100 text-purple-800' if tier != SubscriptionTier.FREE else 'bg-gray-100 text-gray-800'}">
          {tier_str}
        </span>
      </div>

      <div>
        <div class="flex justify-between text-sm mb-1">
          <span class="text-gray-600">Characters This Month</span>
          <span class="font-medium">{chars_used:,} / {chars_limit_str}</span>
        </div>
        {f'''<div class="w-full bg-gray-200 rounded-full h-2">
          <div class="bg-blue-600 h-2 rounded-full transition-all duration-300"
               style="width: {min(chars_percentage, 100)}%"></div>
        </div>''' if char_limit else ''}
      </div>

      <div class="flex justify-between text-sm">
        <span class="text-gray-600">Texts Generated</span>
        <span class="font-medium">{texts_used} / {texts_limit_str}</span>
      </div>
    </div>
    """


# Profile Management
@router.get("/profile/list")
def list_profiles(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Get all profiles for the current user."""
    profiles = db.query(Profile).filter(Profile.account_id == account.id).all()

    result = []
    for p in profiles:
        info = _get_lang_info(p.lang)
        result.append({
            "id": p.id,
            "lang": p.lang,
            "target_lang": p.target_lang,
            "level_value": p.level_value,
            "flag": info["flag"],
            "name": info["name"],
        })

    return {"profiles": result}


@router.post("/profile/switch")
def switch_profile(
    profile_id: int = Body(..., embed=True),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_db),
):
    """Switch to a different profile."""
    profile = db.query(Profile).filter(
        Profile.id == profile_id,
        Profile.account_id == account.id
    ).first()

    if not profile:
        raise HTTPException(404, "Profile not found")

    from fastapi.responses import JSONResponse
    response = JSONResponse({"success": True, "profile_id": profile.id})

    # Set cookie to remember active profile (7 days)
    response.set_cookie(
        "active_profile_id",
        str(profile.id),
        max_age=604800,
        httponly=True,
        secure=False,  # Set True in production with HTTPS
        samesite="lax",
    )

    return response
