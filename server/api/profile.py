from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from Lang.parsing.registry import ENGINES

from ..db import get_db
from ..models import Profile, ProfilePref, SubscriptionTier
from arcadia_auth import Account
from ..deps import get_current_account
from ..repos.tiers import ensure_default_tiers
from ..repos.profiles import get_or_create_profile, get_pref_row


router = APIRouter()


def _is_supported_lang(code: str) -> bool:
    return code in ENGINES


class ProfileRequest(BaseModel):
    lang: str
    target_lang: Optional[str] = "en"  # User's native/reference language
    settings: Optional[Dict[str, Any]] = None
    level_value: Optional[float] = None
    level_var: Optional[float] = None
    level_code: Optional[str] = None
    preferred_script: Optional[str] = None  # 'Hans' | 'Hant' for zh


@router.post("/me/profile")
def upsert_profile(
    req: ProfileRequest,
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    ensure_default_tiers(db)
    if not _is_supported_lang(req.lang):
        raise HTTPException(400, "Unsupported language code")
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == req.lang).first()
    if not prof:
        prof = Profile(account_id=account.id, lang=req.lang, target_lang=req.target_lang or "en")
        db.add(prof)
        db.flush()
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
    db.commit()
    return {
        "ok": True,
        "account_id": account.id,
        "lang": prof.lang,
        "target_lang": prof.target_lang,
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
            )
        )
    return out


@router.delete("/me/profile")
def delete_profile(
    lang: str,
    db: Session = Depends(get_db),
    account: Account = Depends(get_current_account),
):
    prof = db.query(Profile).filter(Profile.account_id == account.id, Profile.lang == lang).first()
    if not prof:
        raise HTTPException(404, "Profile not found")
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == prof.id).first()
    if pref:
        db.delete(pref)
    db.delete(prof)
    db.commit()
    return {"ok": True}


class TierOut(BaseModel):
    name: str
    description: Optional[str] = None


# (Tier management functions removed - not needed for this iteration)


class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@router.get("/me", response_model=MeOut)
def get_me(user: User = Depends(get_current_user)):
    return MeOut(id=user.id, email=user.email, subscription_tier=user.subscription_tier)


class ThemeIn(BaseModel):
    name: Optional[str] = None
    vars: Optional[Dict[str, Any]] = None
    clear: Optional[bool] = None


class UIPrefsIn(BaseModel):
    motion: Optional[bool] = None
    density: Optional[str] = None
    scale: Optional[float] = None
    clear: Optional[bool] = None


@router.get("/theme")
def get_theme(
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = get_or_create_profile(db, user.id, lang)
    pref = get_pref_row(db, prof.id)
    data = dict((pref.data or {}) if pref else {})
    theme = data.get("theme") or {}
    return {"name": theme.get("name"), "vars": theme.get("vars") or {}}


@router.post("/theme")
def set_theme(
    payload: ThemeIn,
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = get_or_create_profile(db, user.id, lang)
    pref = get_pref_row(db, prof.id)
    data = dict(pref.data or {})
    if payload.clear:
        data.pop("theme", None)
    else:
        cur = data.get("theme") or {}
        if payload.name is not None:
            cur["name"] = payload.name
        if payload.vars is not None:
            cur["vars"] = payload.vars
        data["theme"] = cur
    pref.data = data
    db.commit()
    return {"ok": True}


@router.get("/prefs")
def get_ui_prefs(
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = get_or_create_profile(db, user.id, lang)
    pref = get_pref_row(db, prof.id)
    data = dict((pref.data or {}) if pref else {})
    return data.get("ui_prefs") or {}


@router.post("/prefs")
def set_ui_prefs(
    payload: UIPrefsIn,
    lang: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = get_or_create_profile(db, user.id, lang)
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
