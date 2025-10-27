from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.auth import Account  # type: ignore

from ..deps import get_current_account as _get_current_account


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
def settings_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse("pages/settings.html", {"request": request, "title": "Settings"})


@router.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request, account: Account = Depends(_get_current_account)):
    t = _templates()
    return t.TemplateResponse("pages/stats.html", {"request": request, "title": "Statistics"})


@router.get("/", response_class=HTMLResponse)
def home_page(
    request: Request,
    account: Account = Depends(_get_current_account),
):
    t = _templates()

    # Get the user's default language (first profile)
    from ..account_db import get_db
    from ..models import Profile, ReadingText

    db = next(get_db(request))
    profile = db.query(Profile).filter(Profile.account_id == account.id).first()

    context = {
        "request": request,
        "title": "Arcadia Lang",
        "has_profile": profile is not None
    }

    if profile:
        # Get the most recent unopened text
        unopened_text = (
            db.query(ReadingText)
            .filter(
                ReadingText.account_id == account.id,
                ReadingText.lang == profile.lang,
                ReadingText.opened_at.is_(None)
            )
            .order_by(ReadingText.created_at.desc())
            .first()
        )

        if unopened_text:
            context["current_text"] = unopened_text
            context["current_text_id"] = unopened_text.id
        else:
            # No unopened texts - check if we should generate one
            from ..services.llm_service import should_generate_new_text
            if should_generate_new_text(db, account.id, profile.lang):
                # Generate a new text
                try:
                    from ..services.llm_service import generate_reading
                    result = generate_reading(
                        db,
                        account_id=account.id,
                        lang=profile.lang,
                        length=None,
                        include_words=None,
                        model=None,
                        provider="openrouter",
                        base_url="http://localhost:1234/v1"
                    )
                    context["current_text"] = {
                        "id": result.get("text_id"),
                        "content": result.get("text", "")
                    }
                    context["current_text_id"] = result.get("text_id")
                except Exception as e:
                    context["generation_error"] = str(e)

    return t.TemplateResponse("pages/home.html", context)


@router.get("/logout")
def logout() -> RedirectResponse:
    resp = RedirectResponse(url="/", status_code=302)
    try:
        resp.delete_cookie("access_token", path="/")
    except Exception:
        pass
    return resp

