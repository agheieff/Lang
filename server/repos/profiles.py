from __future__ import annotations

from sqlalchemy.orm import Session
from typing import Optional

from ..models import Profile, ProfilePref


def get_or_create_profile(db: Session, user_id: int, lang: str, target_lang: str = "en") -> Profile:
    """Get or create a profile for learning 'lang' with 'target_lang' as reference."""
    prof = db.query(Profile).filter(Profile.account_id == user_id, Profile.lang == lang).first()
    if prof:
        # Update target_lang if provided and different
        if target_lang != "en" and prof.target_lang != target_lang:
            prof.target_lang = target_lang
            db.flush()
        return prof
    prof = Profile(account_id=user_id, lang=lang, target_lang=target_lang)
    db.add(prof)
    db.flush()
    return prof


def get_user_profile(db: Session, user_id: int, lang: str) -> Optional[Profile]:
    """Get user's profile for a specific language."""
    return db.query(Profile).filter(Profile.account_id == user_id, Profile.lang == lang).first()


def get_pref_row(db: Session, profile_id: int) -> ProfilePref:
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == profile_id).first()
    if not pref:
        pref = ProfilePref(profile_id=profile_id, data={})
        db.add(pref)
        db.flush()
    return pref
