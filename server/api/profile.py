from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langs.parsing import ENGINES

from ..db import get_global_db
from ..account_db import get_db
from ..models import Profile, ProfilePref, SubscriptionTier, Language
from arcadia_auth import Account
from ..deps import get_current_account
from ..repos.tiers import ensure_default_tiers
from ..repos.profiles import get_or_create_profile, get_pref_row, get_user_profile

# Server-side guardrails for free-form profile fields
_MIN_TEXT_LEN = 50
_MAX_TEXT_LEN = 2000


router = APIRouter()


def _is_supported_lang(code: str) -> bool:
    return code in ENGINES


def _get_available_language(db: Session, code: str) -> Optional[Language]:
    """Get language from global catalog if it exists and is supported."""
    lang = db.query(Language).filter(Language.code == code).first()
    if lang and _is_supported_lang(code):
        return lang
    return None


class ProfileRequest(BaseModel):
    lang: str
    target_lang: Optional[str] = "en"  # User's native/reference language
    settings: Optional[Dict[str, Any]] = None
    level_value: Optional[float] = None
    level_var: Optional[float] = None
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None  # 'Hans' | 'Hant' for zh
    text_length: Optional[int] = None
    text_preferences: Optional[str] = None


@router.get("/languages")
def get_available_languages(
    tiers_db: Session = Depends(get_global_db),
):
    """Get all available languages from the global catalog."""
    ensure_default_tiers(tiers_db)
    languages = tiers_db.query(Language).all()
    available = []
    for lang in languages:
        if _is_supported_lang(lang.code):
            available.append({
                "code": lang.code,
                "name": lang.name,
                "created_at": lang.created_at
            })
    return {"languages": available}


@router.post("/me/profile")
def create_profile(
    req: ProfileRequest,
    db: Session = Depends(get_db),
    tiers_db: Session = Depends(get_global_db),
    account: Account = Depends(get_current_account),
):
    """Create a new profile with languages from global catalog."""
    ensure_default_tiers(tiers_db)

    # Validate that languages exist in global catalog
    lang_obj = _get_available_language(tiers_db, req.lang)
    target_lang_obj = _get_available_language(tiers_db, req.target_lang or "en")

    if not lang_obj:
        raise HTTPException(400, f"Unsupported language: {req.lang}")
    if not target_lang_obj:
        raise HTTPException(400, f"Unsupported target language: {req.target_lang}")

    # Check if profile already exists
    existing_profile = get_user_profile(db, account.id, req.lang, req.target_lang or "en")
    if existing_profile:
        raise HTTPException(409, "Profile already exists for this language combination")

    # Create profile with language codes directly (not modifiable after creation)
    prof = Profile(
        account_id=account.id,
        lang=req.lang,
        target_lang=req.target_lang or "en"
    )
    if req.settings is not None:
        pref = get_pref_row(db, prof.id)
        pref.data = req.settings
    if req.preferred_script is not None and req.lang.startswith("zh"):
        ps = req.preferred_script
        if ps not in ("Hans", "Hant"):
            raise HTTPException(400, "preferred_script must be 'Hans' or 'Hant'")
        prof.preferred_script = ps
    if req.level_value is not None:
        prof.level_value = float(req.level_value)
    if req.level_var is not None:
        prof.level_var = float(req.level_var)
    if req.level_code is not None:
        prof.level_code = req.level_code
    if req.text_length is not None:
        try:
            n = int(req.text_length)
            if n < _MIN_TEXT_LEN:
                n = _MIN_TEXT_LEN
            if n > _MAX_TEXT_LEN:
                n = _MAX_TEXT_LEN
            prof.text_length = n
        except Exception:
            prof.text_length = None
    if req.text_preferences is not None:
        prof.text_preferences = (req.text_preferences or '').strip() or None

    # Add profile to database
    db.add(prof)
    db.flush()

    # Create preference row if settings were provided
    if req.settings is not None:
        pref = ProfilePref(profile_id=prof.id, data=req.settings)
        db.add(pref)

    db.commit()

    return {
        "ok": True,
        "account_id": account.id,
        "lang": req.lang,
        "target_lang": req.target_lang or "en",
        "level_value": prof.level_value,
        "level_var": prof.level_var,
        "level_code": prof.level_code,
    }


class ProfileOut(BaseModel):
    lang: str
    target_lang: str
    created_at: datetime
    settings: Optional[Dict[str, Any]] = None
    level_value: float
    level_var: float
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None
    text_length: Optional[int] = None
    text_preferences: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    """Request model for updating existing profile (languages cannot be changed)."""
    settings: Optional[Dict[str, Any]] = None
    level_value: Optional[float] = None
    level_var: Optional[float] = None
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None  # 'Hans' | 'Hant' for zh
    text_length: Optional[int] = None
    text_preferences: Optional[str] = None


@router.put("/me/profile")
def update_profile(
    lang: str,
    target_lang: str = "en",
    req: ProfileUpdateRequest = None,
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    """Update an existing profile (languages cannot be changed)."""
    prof = get_user_profile(db, account.id, lang, target_lang)
    if not prof:
        raise HTTPException(404, "Profile not found")

    # Update allowed fields
    if req.settings is not None:
        pref = get_pref_row(db, prof.id)
        pref.data = req.settings
    if req.preferred_script is not None and lang.startswith("zh"):
        ps = req.preferred_script
        if ps not in ("Hans", "Hant"):
            raise HTTPException(400, "preferred_script must be 'Hans' or 'Hant'")
        prof.preferred_script = ps
    if req.level_value is not None:
        prof.level_value = float(req.level_value)
    if req.level_var is not None:
        prof.level_var = float(req.level_var)
    if req.level_code is not None:
        prof.level_code = req.level_code
    if req.text_length is not None:
        try:
            n = int(req.text_length)
            if n < _MIN_TEXT_LEN:
                n = _MIN_TEXT_LEN
            if n > _MAX_TEXT_LEN:
                n = _MAX_TEXT_LEN
            prof.text_length = n
        except Exception:
            prof.text_length = None
    if req.text_preferences is not None:
        prof.text_preferences = (req.text_preferences or '').strip() or None

    db.commit()
    return {"ok": True}


@router.get("/me/profiles", response_model=List[ProfileOut])
def list_profiles(
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    profiles = db.query(Profile).filter(Profile.account_id == account.id).all()
    out: List[ProfileOut] = []
    for p in profiles:
        pref = db.query(ProfilePref).filter(ProfilePref.profile_id == p.id).first()
        out.append(
            ProfileOut(
                lang=p.lang,
                target_lang=p.target_lang,
                created_at=p.created_at,
                settings=(pref.data if pref else None) or p.settings,
                level_value=p.level_value,
                level_var=p.level_var,
                level_code=p.level_code,
                preferred_script=p.preferred_script,
                text_length=p.text_length,
                text_preferences=p.text_preferences,
            )
        )
    return out


@router.delete("/me/profile")
def delete_profile(
    lang: str,
    target_lang: str = "en",
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    prof = get_user_profile(db, account.id, lang, target_lang)
    if not prof:
        raise HTTPException(404, "Profile not found")
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    if pref:
        db.delete(pref)
    db.delete(prof)
    db.commit()
    return {"ok": True}

class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@router.get("/me", response_model=MeOut)
def get_me(account: Account = Depends(get_current_account)):
    # Align with /me/tier behavior: default to "Free" if unset
    return MeOut(
        id=account.id,
        email=account.email,
        subscription_tier=(account.subscription_tier or "Free"),
    )


class UIPrefsIn(BaseModel):
    motion: Optional[bool] = None
    density: Optional[str] = None
    scale: Optional[float] = None
    clear: Optional[bool] = None


## Theme endpoints removed — themes come from UI libs entirely.


@router.get("/prefs")
def get_ui_prefs(
    lang: str,
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    prof = get_or_create_profile(db, account.id, lang)
    pref = get_pref_row(db, prof.id)
    data = dict((pref.data or {}) if pref else {})
    return data.get("ui_prefs") or {}


@router.post("/prefs")
def set_ui_prefs(
    payload: UIPrefsIn,
    lang: str,
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    prof = get_or_create_profile(db, account.id, lang)
    pref = get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    if payload.clear:
        data.pop("ui_prefs", None)
    else:
        cur = data.get("ui_prefs") or {}
        if payload.motion is not None:
            cur["motion"] = bool(payload.motion)
        if payload.density:
            cur["density"] = payload.density
        if payload.scale is not None:
            cur["scale"] = float(payload.scale)
        data["ui_prefs"] = cur
    pref.data = data
    db.commit()
    return {"ok": True}
