from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from server.db import get_db, db_transaction
from server.models import (
    Profile,
    ReadingText,
    ReadingTextTranslation,
    ReadingWordGloss,
    ProfileTextRead,
)
from server.deps import get_current_account

# Move service imports to top
from server.services.recommendation import select_best_text
# NOTE: Old tracking system disabled - using new reading-text-log system instead
# from server.services.learning import (
#     track_interactions_from_session,
#     update_level_from_text,
# )

logger = logging.getLogger(__name__)
router = APIRouter(tags=["reading"])

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _get_account_id(request: Request) -> Optional[int]:
    """Get account_id from request state set by middleware."""
    return getattr(request.state, "account_id", None)


def _get_profile_for_account(db: Session, account_id: int) -> Optional[Profile]:
    """Get profile for a given account_id."""
    return db.query(Profile).filter(Profile.account_id == account_id).first()


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
DEMO_TEXT = """Generating your first text...

Please wait while we create a personalized reading text for you.
This usually takes 30-60 seconds.

You can refresh this page to check if your text is ready."""


@router.get("/reading", response_class=HTMLResponse)
def reading_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Reading practice page."""

    # Get current user
    account_id = _get_account_id(request)

    if account_id is None:
        return HTMLResponse(
            "<div><h1>Please log in</h1><p><a href='/login'>Login</a></p></div>"
        )

    # Check for active profile from cookie
    active_profile_id = request.cookies.get("active_profile_id")
    if active_profile_id:
        try:
            profile = db.query(Profile).filter(
                Profile.id == int(active_profile_id),
                Profile.account_id == account_id
            ).first()
        except ValueError:
            profile = None
    else:
        profile = None

    # Fall back to first profile if no active profile specified
    if profile is None:
        profile = _get_profile_for_account(db, account_id)

    if profile is None:
        return HTMLResponse(
            "<div><h1>No profile</h1><p><a href='/profile'>Create a profile</a></p></div>"
        )

    # Use recommendation engine to select best text
    ready_text = select_best_text(db, profile)

    if not ready_text:
        # Use demo text
        context = {
            "title": "Reading Practice",
            "profile": profile,
            "current_profile": profile,
            "text_content": DEMO_TEXT,
            "is_demo": True,
            "text_id": None,
            "word_data": [],
            "account_id": account_id,
            "profile_id": profile.id if profile else None,
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
                "pinyin": g.pinyin,  # Pronunciation (pinyin/IPA)
                "span_start": g.span_start,
                "span_end": g.span_end,
                "grammar": g.grammar,  # Include multi-segment spans
            }
            for g in word_glosses
        ]

        context = {
            "title": ready_text.title or "Reading Practice",
            "profile": profile,
            "current_profile": profile,
            "text_content": ready_text.content,
            "is_demo": False,
            "text_id": ready_text.id,
            "word_data": word_data,
            "account_id": account_id,
            "profile_id": profile.id,
        }

    return _templates.TemplateResponse(request, "pages/reading.html", context)


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

    try:
        # Get current user
        account_id = _get_account_id(request)

        if not account_id:
            return JSONResponse({"status": "error", "message": "Not authenticated"})

        profile = _get_profile_for_account(db, account_id)
        if not profile or not profile.current_text_id:
            return JSONResponse({"status": "error", "message": "No current text"})

        with db_transaction(db):
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

        return JSONResponse(
            {
                "status": "ok",
                "message": "Moving to next text",
                "next_text_id": next_text.id if next_text else None,
            }
        )

    except Exception as e:
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


# @router.post("/reading/word-click")  # DISABLED - using new reading-text-log system
def word_click(
    data: WordClickRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Track word clicks for SRS."""

    try:
        account_id = _get_account_id(request)
        if not account_id:
            return JSONResponse(
                {"status": "error", "message": "Not authenticated"}, status_code=401
            )

        # Get profile from account_id
        profile = _get_profile_for_account(db, account_id)
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


# @router.post("/reading/save-session")  # DISABLED - using new reading-text-log system
def save_session(
    data: SaveSessionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save reading session and update user level."""

    try:
        account_id = _get_account_id(request)
        if not account_id:
            return JSONResponse(
                {"status": "error", "message": "Not authenticated"}, status_code=401
            )

        if not data.text_id:
            return JSONResponse(
                {"status": "error", "message": "Missing text_id"}, status_code=400
            )

        # Get profile from account_id
        profile = _get_profile_for_account(db, account_id)
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
