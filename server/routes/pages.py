from __future__ import annotations

from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..db import get_db

router = APIRouter(tags=["ui"])

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


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    t = _templates()
    return t.TemplateResponse(request, "pages/login.html", {"title": "Log in"})


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    t = _templates()
    return t.TemplateResponse(request, "pages/signup.html", {"title": "Sign up"})


@router.get("/", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    no_texts: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Dashboard/home page - simple landing with links to reading and other features."""
    t = _templates()
    from ..models import Profile

    account_id = getattr(request.state, "account_id", None)

    profile = None
    if account_id is not None:
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()

    # Map language codes to display names
    lang_names = {
        "es": "Spanish",
        "zh": "Chinese",
        "en": "English",
        "fr": "French",
        "de": "German",
    }

    context = {
        "title": "Arcadia Lang",
        "has_profile": profile is not None,
        "profile_lang": (profile.lang if profile is not None else None),
        "profile_lang_name": lang_names.get(profile.lang, profile.lang)
        if profile
        else None,
        "is_authenticated": account_id is not None,
        "no_texts": no_texts == "1",
    }

    return t.TemplateResponse(request, "pages/dashboard.html", context)
