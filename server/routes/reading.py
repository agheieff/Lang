from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging
from pydantic import BaseModel, Field
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

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reading"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"


def _templates() -> Jinja2Templates:
    try:
        env = Environment(
            loader=FileSystemLoader([str(TEMPLATES_DIR)]),
            autoescape=select_autoescape(["html", "xml"]),
        )
        return Jinja2Templates(env=env)
    except Exception:
        return Jinja2Templates(directory=str(TEMPLATES_DIR))


# Pydantic models for request validation
class WordInteraction(BaseModel):
    surface: str
    lemma: Optional[str] = None
    pos: str = "NOUN"
    span_start: int
    span_end: int
    clicked: bool = False
    click_count: int = 0
    translation_viewed: bool = False
    translation_viewed_at: Optional[float] = None
    timestamp: Optional[float] = None


class SessionData(BaseModel):
    exposed_at: Optional[float] = None
    words: List[WordInteraction] = []
    sentences: Optional[List[Dict[str, Any]]] = None
    full_translation_views: Optional[List[Dict[str, Any]]] = None


class WordClickRequest(BaseModel):
    text_id: int = Field(..., gt=0, description="Text ID")
    word_data: Optional[Dict[str, Any]] = None
    session_data: Optional[SessionData] = None


class SaveSessionRequest(BaseModel):
    text_id: int = Field(..., gt=0, description="Text ID")
    session_data: Optional[SessionData] = None


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
    account_id = getattr(request.state, "account_id", None)

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
        account_id = getattr(request.state, "account_id", None)

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
    data: WordClickRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Track word clicks for SRS."""
    from server.services.learning import (
        track_interactions_from_session,
    )

    try:
        account_id = getattr(request.state, "account_id", None)
        if not account_id:
            return JSONResponse(
                {"status": "error", "message": "Not authenticated"}, status_code=401
            )

        # Get profile from account_id
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile:
            return JSONResponse(
                {"status": "error", "message": "Profile not found"}, status_code=404
            )

        # Track all interactions from session data (includes clicks with timestamps)
        if data.session_data and data.session_data.words:
            track_interactions_from_session(
                db=db,
                account_id=account_id,
                profile_id=profile.id,
                text_id=data.text_id,
                interactions=[
                    {
                        "surface": w.surface,
                        "lemma": w.lemma or w.surface,
                        "pos": w.pos,
                        "span_start": w.span_start,
                        "span_end": w.span_end,
                        "clicked": w.clicked,
                        "click_count": w.click_count,
                        "translation_viewed": w.translation_viewed,
                    }
                    for w in data.session_data.words
                ],
            )

        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error in word_click: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": "Internal server error"}, status_code=500
        )


@router.post("/reading/save-session")
def save_session(
    data: SaveSessionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save reading session and update user level."""
    from server.services.learning import update_level_from_text
    from server.models import ProfileTextRead

    try:
        account_id = getattr(request.state, "account_id", None)
        if not account_id:
            return JSONResponse(
                {"status": "error", "message": "Not authenticated"}, status_code=401
            )

        if not data.text_id:
            return JSONResponse(
                {"status": "error", "message": "Missing text_id"}, status_code=400
            )

        # Get profile from account_id
        profile = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not profile:
            return JSONResponse(
                {"status": "error", "message": "Profile not found"}, status_code=404
            )

        # Process session data
        session_data = data.session_data or SessionData()
        words = session_data.words or []
        sentences = session_data.sentences or []

        # Track text exposure (when user first saw full text)
        if session_data.exposed_at:
            # Convert JS timestamp to datetime
            exposed_dt = datetime.fromtimestamp(
                session_data.exposed_at / 1000.0, tz=timezone.utc
            )

            # Check if this text was already tracked as read
            existing_read = (
                db.query(ProfileTextRead)
                .filter(
                    ProfileTextRead.profile_id == profile.id,
                    ProfileTextRead.text_id == data.text_id,
                )
                .first()
            )

            if not existing_read:
                # Create read entry when session is saved
                read_entry = ProfileTextRead(
                    profile_id=profile.id,
                    text_id=data.text_id,
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
                "surface": word.surface,
                "lemma": word.lemma or word.surface,
                "pos": word.pos,
                "span_start": word.span_start,
                "span_end": word.span_end,
                "clicked": word.clicked,
                "click_count": word.click_count,
                "translation_viewed": word.translation_viewed,
                "translation_viewed_at": word.translation_viewed_at,
                "timestamp": word.timestamp,
            }
            word_interactions.append(interaction)

        # Update user level based on all interactions
        new_level, new_var = update_level_from_text(
            db=db, profile=profile, text_id=data.text_id, interactions=word_interactions
        )

        return {
            "status": "ok",
            "level_value": new_level,
            "level_var": new_var,
            "processed_words": len(word_interactions),
            "processed_sentences": len(sentences),
        }
    except Exception as e:
        logger.error(f"Error in save_session: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": "Internal server error"}, status_code=500
        )
