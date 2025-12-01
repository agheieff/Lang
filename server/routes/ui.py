from __future__ import annotations

from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.auth import Account  # type: ignore

from ..deps import get_current_account as _get_current_account
from ..account_db import get_db as get_account_db
from ..db import get_global_db


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


@router.get("/words", response_class=HTMLResponse)
def words_page(
    request: Request,
    lang: Optional[str] = None,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_global_db),
):
    t = _templates()
    # Default to the user's first profile language if not provided
    eff_lang: Optional[str] = None
    if lang:
        eff_lang = lang
    else:
        try:
            from ..models import Profile
            prof = db.query(Profile).filter(Profile.account_id == account.id).first()
            eff_lang = prof.lang if prof else None
        except Exception:
            eff_lang = None
    return t.TemplateResponse(
        request,
        "pages/words.html",
        {"title": "My Words", "lang": eff_lang or "es"},
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    t = _templates()
    return t.TemplateResponse(request, "pages/login.html", {"title": "Log in"})


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    t = _templates()
    return t.TemplateResponse(request, "pages/signup.html", {"title": "Sign up"})


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
    from ..models import Profile
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


@router.get("/", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    no_texts: Optional[str] = None,
    db: Session = Depends(get_global_db),
):
    """Dashboard/home page - simple landing with links to reading and other features."""
    t = _templates()
    from ..models import Profile

    account_id: Optional[int] = None
    try:
        u = getattr(request.state, "user", None)
        if u is not None:
            if isinstance(u, dict) and "id" in u:
                account_id = int(u["id"])
            elif hasattr(u, "id"):
                account_id = int(getattr(u, "id"))
    except Exception:
        account_id = None

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
        "profile_lang_name": lang_names.get(profile.lang, profile.lang) if profile else None,
        "is_authenticated": account_id is not None,
        "no_texts": no_texts == "1",
    }

    return t.TemplateResponse(request, "pages/dashboard.html", context)


@router.get("/reading", response_class=HTMLResponse)
def reading_page(
    request: Request,
    db: Session = Depends(get_global_db),
):
    """Reading practice page with text display."""
    t = _templates()
    from ..models import Profile, ReadingText

    account_id: Optional[int] = None
    try:
        u = getattr(request.state, "user", None)
        if u is not None:
            if isinstance(u, dict) and "id" in u:
                account_id = int(u["id"])
            elif hasattr(u, "id"):
                account_id = int(getattr(u, "id"))
    except Exception:
        account_id = None

    # Redirect to login if not authenticated
    if account_id is None:
        return RedirectResponse(url="/login", status_code=302)

    profile = None
    if account_id is not None:
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()

    # Redirect to profile creation if no profile
    if profile is None:
        return RedirectResponse(url="/profile", status_code=302)

    # Check if there are any ready texts for this language
    ready_count = db.query(ReadingText).filter(
        ReadingText.lang == profile.lang,
        ReadingText.target_lang == profile.target_lang,
        ReadingText.content.isnot(None),
        ReadingText.words_complete == True,
        ReadingText.sentences_complete == True,
    ).count()

    # If no texts ready, redirect to dashboard with message
    if ready_count == 0:
        return RedirectResponse(url="/?no_texts=1", status_code=302)

    context = {
        "title": "Reading Practice",
        "has_profile": profile is not None,
        "profile_lang": (profile.lang if profile is not None else None),
        "is_authenticated": account_id is not None,
    }

    return t.TemplateResponse(request, "pages/home.html", context)


## Logout handled by /auth/logout in the auth router

