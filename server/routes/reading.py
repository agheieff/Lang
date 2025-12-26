from __future__ import annotations

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.orm import Session

from server.db import get_db
from server.models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from server.deps import get_current_account

router = APIRouter(tags=["reading"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_templates_env = None


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


# Demo text for when no texts are available
DEMO_TEXT = """Hola! Welcome to your reading practice.
This is a simple demo text to get you started.
The full text generation system will be available soon."""


@router.get("/reading", response_class=HTMLResponse)
def reading_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Reading practice page."""
    t = _templates()

    # Get current user
    account_id = None
    try:
        u = getattr(request.state, "user", None)
        if u is not None:
            account_id = getattr(u, "id", None)
    except Exception:
        pass

    if account_id is None:
        return HTMLResponse(
            "<div><h1>Please log in</h1><p><a href='/login'>Login</a></p></div>"
        )

    profile = db.query(Profile).filter(Profile.account_id == account_id).first()

    if profile is None:
        return HTMLResponse(
            "<div><h1>No profile</h1><p><a href='/profile'>Create a profile</a></p></div>"
        )

    # Check if there are any ready texts
    ready_text = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == profile.lang,
            ReadingText.target_lang == profile.target_lang,
            ReadingText.status == "ready",
        )
        .first()
    )

    if not ready_text:
        # Use demo text
        context = {
            "title": "Reading Practice",
            "profile": profile,
            "text_content": DEMO_TEXT,
            "is_demo": True,
            "text_id": None,
        }
    else:
        # Load the ready text
        word_glosses = (
            db.query(ReadingWordGloss)
            .filter(ReadingWordGloss.text_id == ready_text.id)
            .all()
        )

        word_data = [
            {
                "surface": g.surface,
                "lemma": g.lemma,
                "pos": g.pos,
                "translation": g.translation,
                "span_start": g.span_start,
                "span_end": g.span_end,
            }
            for g in word_glosses
        ]

        context = {
            "title": "Reading Practice",
            "profile": profile,
            "text_content": ready_text.content,
            "is_demo": False,
            "text_id": ready_text.id,
            "word_data": word_data,
        }

    return t.TemplateResponse(request, "pages/reading.html", context)


@router.get("/reading/current", response_class=HTMLResponse)
def reading_current(
    request: Request,
    db: Session = Depends(get_db),
):
    """Get current text as HTML fragment (for HTMX refresh)."""
    return reading_page(request, db)


@router.post("/reading/next")
def reading_next(
    request: Request,
    db: Session = Depends(get_db),
):
    """Mark current text as read and move to next."""
    # For now, just return a success
    return JSONResponse({"status": "ok", "message": "Moving to next text"})


@router.get("/reading/{text_id}/translations")
def get_text_translations(
    text_id: int,
    unit: str = "sentence",
    db: Session = Depends(get_db),
):
    """Get translations for a text."""
    translations = (
        db.query(ReadingTextTranslation)
        .filter(
            ReadingTextTranslation.text_id == text_id,
            ReadingTextTranslation.unit == unit,
        )
        .all()
    )

    items = [
        {
            "index": t.index,
            "unit": t.unit,
            "source": t.source,
            "translation": t.translation,
        }
        for t in translations
    ]

    return {"items": items}


@router.get("/reading/{text_id}/status")
def get_text_status(
    text_id: int,
    db: Session = Depends(get_db),
):
    """Get status of a text."""
    text = db.query(ReadingText).filter(ReadingText.id == text_id).first()
    if not text:
        return {"status": "not_found", "next_ready": False}

    return {
        "status": text.status,
        "next_ready": text.status == "ready",
    }


@router.post("/reading/word-click")
def word_click(
    data: dict,
    db: Session = Depends(get_db),
):
    """Track word clicks (for now just log)."""
    return {"status": "ok"}


@router.post("/reading/save-session")
def save_session(
    data: dict,
    db: Session = Depends(get_db),
):
    """Save reading session (for now just log)."""
    return {"status": "ok"}
