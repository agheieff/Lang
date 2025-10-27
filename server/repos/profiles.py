from __future__ import annotations

from sqlalchemy.orm import Session
from typing import Optional

from ..models import Profile, ProfilePref, Language


def get_or_create_language(db: Session, code: str, name: Optional[str] = None) -> Language:
    """Get or create a language by code."""
    language = db.query(Language).filter(Language.code == code).first()
    if language:
        return language

    # Default names for common languages
    if not name:
        name_map = {
            "es": "Spanish",
            "zh": "Chinese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean"
        }
        name = name_map.get(code, code.capitalize())

    language = Language(code=code, name=name)
    db.add(language)
    db.flush()
    return language


def get_or_create_profile(db: Session, user_id: int, lang: str, target_lang: str = "en") -> Profile:
    """Get or create a profile for learning 'lang' with 'target_lang' as reference.

    Note: This function is deprecated in favor of explicit profile creation via API.
    Languages must exist in global catalog and cannot be created automatically.
    """
    prof = db.query(Profile).filter(
        Profile.account_id == user_id,
        Profile.lang == lang,
        Profile.target_lang == target_lang
    ).first()

    if prof:
        return prof

    prof = Profile(
        account_id=user_id,
        lang=lang,
        target_lang=target_lang
    )
    db.add(prof)
    db.flush()
    return prof


def get_user_profile(db: Session, user_id: int, lang: str, target_lang: str = "en") -> Optional[Profile]:
    """Get user's profile for a specific language and target language."""
    return db.query(Profile).filter(
        Profile.account_id == user_id,
        Profile.lang == lang,
        Profile.target_lang == target_lang
    ).first()


def get_pref_row(db: Session, profile_id: int) -> ProfilePref:
    pref = db.query(ProfilePref).filter(ProfilePref.profile_id == profile_id).first()
    if not pref:
        pref = ProfilePref(profile_id=profile_id, data={})
        db.add(pref)
        db.flush()
    return pref
