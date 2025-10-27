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
):
    t = _templates()
    return t.TemplateResponse(
        "pages/words.html", {"request": request, "title": "My Words", "lang": lang or "es"}
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    t = _templates()
    return t.TemplateResponse("pages/login.html", {"request": request, "title": "Log in"})


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    t = _templates()
    return t.TemplateResponse("pages/signup.html", {"request": request, "title": "Sign up"})


@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse("pages/profile.html", {"request": request, "title": "Profile"})


@router.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    account: Account = Depends(_get_current_account),
    db: Session = Depends(get_account_db),
):
    t = _templates()
    from ..models import Profile
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    context = {
        "request": request,
        "title": "Settings",
        "current_profile": profile or {}
    }

    return t.TemplateResponse("pages/settings.html", context)


@router.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse("pages/stats.html", {"request": request, "title": "Statistics"})


@router.get("/", response_class=HTMLResponse)
def home_page(
    request: Request,
    db: Session = Depends(get_account_db),
):
    t = _templates()

    # Get the user's default language (first profile)
    from ..models import Profile

    account_id: Optional[int] = None
    try:
        u = getattr(request.state, "user", None)
        if u is not None:
            if isinstance(u, dict) and "id" in u:
                account_id = int(u["id"])  # type: ignore[arg-type]
            elif hasattr(u, "id"):
                account_id = int(getattr(u, "id"))
    except Exception:
        account_id = None

    profile = None
    if account_id is not None:
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()

    context = {
        "request": request,
        "title": "Arcadia Lang",
        "has_profile": profile is not None,
        "profile_lang": (profile.lang if profile is not None else None),
        "is_authenticated": account_id is not None,
    }

    # Do not generate or fetch reading synchronously here; HTMX will fetch it after load.

    return t.TemplateResponse("pages/home.html", context)


@router.get("/logout")
def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/", status_code=302)
    try:
        resp.delete_cookie("access_token", path="/")
    except Exception:
        pass
    return resp

