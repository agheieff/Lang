from __future__ import annotations

from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, Request, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.auth import Account  # type: ignore

from ..deps import get_current_account as _get_current_account
from ..db import get_global_db
from ..models import Profile

router = APIRouter(tags=["user"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_templates_env: Optional[Jinja2Templates] = None

def _templates() -> Jinja2Templates:
    global _templates_env
    if _templates_env is None:
        try:
            env = Environment(
                loader=FileSystemLoader([str(TEMPLATES_DIR)]),
                autoescape=select_autoescape(["html", "xml"]),
            )
            _templates_env = Jinja2Templates(env=env)
        except Exception:
            _templates_env = Jinja2Templates(directory=str(TEMPLATES_DIR))
    return _templates_env


# HTML Page Endpoints
@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse(request, "pages/profile.html", {"title": "Profile"})


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_global_db),
):
    t = _templates()
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    context = {
        "title": "Settings",
        "current_profile": profile or {},
    }

    return t.TemplateResponse(request, "pages/settings.html", context)


@router.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse(request, "pages/stats.html", {"title": "Statistics"})


# API Endpoints (merged from profile.py and settings.py)

@router.get("/profile/api")
def get_profile(
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_global_db)
):
    """Get current user profile."""
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
        "preferred_script": getattr(profile, 'preferred_script', None),
        "created_at": profile.created_at,
        "updated_at": profile.updated_at
    }


@router.post("/profile/api")
def create_or_update_profile(
    profile_data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_global_db)
):
    """Create or update user profile."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    
    if profile:
        # Update existing
        for key, value in profile_data.items():
            if hasattr(profile, key) and key not in ['id', 'account_id', 'created_at']:
                setattr(profile, key, value)
        db.commit()
        db.refresh(profile)
    else:
        # Create new
        profile = Profile(account_id=account.id, **profile_data)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    
    return {
        "id": profile.id,
        "account_id": profile.account_id,
        "lang": profile.lang,
        "target_lang": profile.target_lang,
        "level_value": profile.level_value,
        "text_length": profile.text_length,
        "preferred_script": getattr(profile, 'preferred_script', None)
    }


@router.put("/profile/api")
def update_profile(
    profile_data: dict = Body(...),
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_global_db)
):
    """Update existing profile."""
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()
    if not profile:
        raise HTTPException(404, "Profile not found")
    
    for key, value in profile_data.items():
        if hasattr(profile, key) and key not in ['id', 'account_id', 'created_at']:
            setattr(profile, key, value)
    
    db.commit()
    db.refresh(profile)
    
    return {
        "id": profile.id,
        "account_id": profile.account_id,
        "lang": profile.lang,
        "target_lang": profile.target_lang,
        "level_value": profile.level_value,
        "text_length": profile.text_length,
        "preferred_script": getattr(profile, 'preferred_script', None)
    }
