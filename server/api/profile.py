from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from langs.parsing import ENGINES

from ..db import get_db
from ..models import Profile, ProfilePref, SubscriptionTier
from arcadia_auth import Account
from ..deps import get_current_account
from ..repos.tiers import ensure_default_tiers
from profiles.service import get_or_create as get_or_create_profile, pref_row as get_pref_row


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
    text_length: Optional[int] = None
    text_preferences: Optional[str] = None


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
    if req.text_length is not None:
        try:
            prof.text_length = int(req.text_length)
        except Exception:
            prof.text_length = None
    if req.text_preferences is not None:
        prof.text_preferences = (req.text_preferences or '').strip() or None
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
    text_length: Optional[int] = None
    text_preferences: Optional[str] = None


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

class MeOut(BaseModel):
    id: int
    email: str
    subscription_tier: str


@router.get("/me", response_model=MeOut)
def get_me(account: Account = Depends(get_current_account)):
    return MeOut(id=account.id, email=account.email, subscription_tier=account.subscription_tier)


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
