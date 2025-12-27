from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.orm import Session

from server.db import get_db
from server.models import (
    Profile,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    ProfileTextRead,
)
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

    # Use recommendation engine to select best text
    from server.services.recommendation import select_best_text

    ready_text = select_best_text(db, profile)

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
        # Mark text as current
        profile.current_text_id = ready_text.id
        db.commit()

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
            "title": ready_text.title or "Reading Practice",
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
    from server.services.recommendation import select_best_text
    from server.models import ProfileTextRead

    try:
        # Get current user
        account_id = None
        try:
            u = getattr(request.state, "user", None)
            if u is not None:
                account_id = getattr(u, "id", None)
        except Exception:
            pass

        if not account_id:
            return JSONResponse({"status": "error", "message": "Not authenticated"})

        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile or not profile.current_text_id:
            return JSONResponse({"status": "error", "message": "No current text"})

        # Mark current text as read
        existing = (
            db.query(ProfileTextRead)
            .filter(
                ProfileTextRead.profile_id == profile.id,
                ProfileTextRead.text_id == profile.current_text_id,
            )
            .first()
        )

        if existing:
            existing.read_count += 1
            existing.last_read_at = datetime.now(timezone.utc)
        else:
            read_entry = ProfileTextRead(
                profile_id=profile.id,
                text_id=profile.current_text_id,
                read_count=1,
            )
            db.add(read_entry)

        # Select next text
        next_text = select_best_text(db, profile)

        if next_text:
            profile.current_text_id = next_text.id

        db.commit()

        return JSONResponse(
            {
                "status": "ok",
                "message": "Moving to next text",
                "next_text_id": next_text.id if next_text else None,
            }
        )

    except Exception as e:
        db.rollback()
        return JSONResponse({"status": "error", "message": str(e)})


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
            "index": t.segment_index,
            "unit": t.unit,
            "source": t.source_text,
            "translation": t.translated_text,
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
        "status": "ready" if text.is_ready else "not_ready",
        "next_ready": text.is_ready,
    }


@router.post("/reading/word-click")
def word_click(
    data: dict,
    request: Request,
    db: Session = Depends(get_db),
):
    """Track word clicks for SRS."""
    from server.services.learning import (
        track_word_click,
        track_interactions_from_session,
    )

    try:
        account_id = getattr(request.state, "user_id", None)
        if not account_id:
            return {"status": "error", "message": "Not authenticated"}

        text_id = data.get("text_id")
        word_data = data.get("word_data", {})
        session_data = data.get("session_data", {})

        if not text_id:
            return {"status": "error", "message": "Missing text_id"}

        # Get profile from account_id
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile:
            return {"status": "error", "message": "Profile not found"}

        # Track individual word click (for immediate feedback)
        track_word_click(
            db=db,
            account_id=account_id,
            profile_id=profile.id,
            text_id=text_id,
            word_info=word_data,
        )

        # Track all interactions from session data
        if session_data.get("words"):
            track_interactions_from_session(
                db=db,
                account_id=account_id,
                profile_id=profile.id,
                text_id=text_id,
                interactions=session_data.get("words", []),
            )

        return {"status": "ok"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}

        text_id = data.get("text_id")
        word_info = data.get("word_info")

        if not text_id or not word_info:
            return {"status": "error", "message": "Missing data"}

        # Get profile from account_id
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile:
            return {"status": "error", "message": "Profile not found"}

        track_word_click(
            db=db,
            account_id=account_id,
            profile_id=profile.id,
            text_id=text_id,
            word_info=word_info,
        )
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/reading/save-session")
def save_session(
    data: dict,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save reading session and update user level."""
    from server.services.learning import update_level_from_text
    from datetime import datetime, timezone
    from server.models import ProfileTextRead

    try:
        account_id = getattr(request.state, "user_id", None)
        if not account_id:
            return {"status": "error", "message": "Not authenticated"}

        text_id = data.get("text_id")
        session_data = data.get("session_data", {})

        if not text_id:
            return {"status": "error", "message": "Missing text_id"}

        # Get profile from account_id
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile:
            return {"status": "error", "message": "Profile not found"}

        # Process session data
        exposed_at = session_data.get("exposed_at")
        words = session_data.get("words", [])
        sentences = session_data.get("sentences", [])
        full_translation_views = session_data.get("full_translation_views", [])

        # Track text exposure (when user first saw the full text)
        if exposed_at:
            # Convert JS timestamp to datetime
            exposed_dt = datetime.fromtimestamp(exposed_at / 1000.0, tz=timezone.utc)

            # Check if this text was already tracked as read
            existing_read = (
                db.query(ProfileTextRead)
                .filter(
                    ProfileTextRead.profile_id == profile.id,
                    ProfileTextRead.text_id == text_id,
                )
                .first()
            )

            if not existing_read:
                # Create read entry when session is saved
                read_entry = ProfileTextRead(
                    profile_id=profile.id,
                    text_id=text_id,
                    read_count=1,
                    first_read_at=exposed_dt,
                    last_read_at=exposed_dt,
                )
                db.add(read_entry)
                db.commit()

        # Process word interactions
        word_interactions = []
        for word in words:
            interaction = {
                "surface": word.get("surface"),
                "lemma": word.get("lemma"),
                "pos": word.get("pos"),
                "span_start": word.get("span_start"),
                "span_end": word.get("span_end"),
                "clicked": word.get("clicked", False),
                "click_count": word.get("click_count", 0),
                "translation_viewed": word.get("translation_viewed", False),
                "translation_viewed_at": word.get("translation_viewed_at"),
                "timestamp": word.get("timestamp"),
            }
            word_interactions.append(interaction)

        # Update user level based on all interactions
        new_level, new_var = update_level_from_text(
            db=db, profile=profile, text_id=text_id, interactions=word_interactions
        )

        return {
            "status": "ok",
            "level_value": new_level,
            "level_var": new_var,
            "processed_words": len(word_interactions),
            "processed_sentences": len(sentences),
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"status": "error", "message": str(e)}
