"""
Text state serving and logging system.
Texts are generated with complete state JSON on server, then served to clients.
Clients return the state when finished reading, saved per (text_id, profile_id).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from server.db import get_db
from server.models import (
    TextState,
    ProfileTextState,
    Profile,
    Lexeme,
)
from server.services.text_state_builder import get_text_state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["text_state"])


class TextStateLog(BaseModel):
    """Complete unified text state - global text data + client metadata."""

    # Core text data (from server)
    text_id: int
    lang: str
    target_lang: str
    title: Optional[str] = None
    content: str
    topic: Optional[str] = None
    difficulty_estimate: Optional[float] = None
    ci_target: Optional[float] = None
    generated_at: Optional[str] = None

    # Text components - hierarchical structure
    words: list[dict]  # All words with spans
    paragraphs: list[dict]  # Paragraphs containing sentences
    word_count: Optional[int] = 0
    paragraph_count: Optional[int] = 0
    sentence_count: Optional[int] = 0

    # Legacy fields (for backward compatibility, will be removed)
    sentence_translations: Optional[list[dict]] = None

    # Client-added metadata
    account_id: Optional[int] = None
    profile_id: Optional[int] = None
    loaded_at: Optional[str] = None
    saved_at: Optional[str] = None

    # Status info
    status: Optional[str] = None
    completed_at: Optional[str] = None


@router.get("/reading/{text_id}/state")
def get_text_state_endpoint(
    text_id: int,
    db: Session = Depends(get_db),
):
    """Get pre-generated complete text state."""
    try:
        state = get_text_state(db, text_id)

        if not state:
            return JSONResponse(
                {"status": "error", "message": "Text state not found"},
                status_code=404,
            )

        # Check if state is ready
        state_obj = db.query(TextState).filter(TextState.text_id == text_id).first()
        if not state_obj or state_obj.status != "ready":
            return JSONResponse(
                {
                    "status": "not_ready",
                    "message": "Text state is still being built",
                    "build_status": state_obj.status if state_obj else "unknown",
                },
                status_code=202,  # Accepted but not ready
            )

        return {
            "status": "ok",
            "state": state,
        }

    except Exception as e:
        logger.error(f"Error getting text state: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )


@router.post("/reading/log-text-state")
def log_text_state(
    data: TextStateLog,
    request: Request,
    db: Session = Depends(get_db),
):
    """Save unified text state from client - stores per (text_id, profile_id) and adds words to profile vocabulary."""
    try:
        profile_id = data.profile_id

        logger.info(f"[TextState] Saving state: text_id={data.text_id}, profile_id={profile_id}, words_count={len(data.words) if data.words else 0}")

        if not profile_id:
            logger.error(f"[TextState] Save FAILED: No profile_id in request data")
            return JSONResponse(
                {"status": "error", "message": "profile_id required"},
                status_code=400,
            )

        # Convert to dict for storage
        state_dict = data.model_dump()

        # Add all words from this text to profile's vocabulary
        result = _add_words_to_profile_vocabulary(
            db=db,
            account_id=data.account_id,
            profile_id=profile_id,
            text_id=data.text_id,
            lang=data.lang,
            words=data.words,
        )

        logger.info(f"[TextState] Vocabulary added: {result}")

        # Create or update profile text state
        existing = (
            db.query(ProfileTextState)
            .filter(
                ProfileTextState.text_id == data.text_id,
                ProfileTextState.profile_id == profile_id,
            )
            .first()
        )

        if existing:
            # Update existing record
            existing.state_data = state_dict
            existing.account_id = data.account_id
            if data.saved_at:
                # Parse ISO string to datetime (strip timezone for simplicity)
                try:
                    # Remove timezone offset to make it compatible with SQLite
                    saved_at_clean = data.saved_at.split('+')[0].split('Z')[0]
                    existing.saved_at = datetime.fromisoformat(saved_at_clean)
                except:
                    existing.saved_at = datetime.now(timezone.utc)
            logger.info(f"Updated text state for text={data.text_id}, profile={profile_id}")
        else:
            # Parse saved_at if provided, otherwise use now
            saved_at = None
            if data.saved_at:
                try:
                    # Remove timezone offset to make it compatible with SQLite
                    saved_at_clean = data.saved_at.split('+')[0].split('Z')[0]
                    saved_at = datetime.fromisoformat(saved_at_clean)
                except:
                    saved_at = datetime.now(timezone.utc)
            else:
                saved_at = datetime.now(timezone.utc)

            # Create new record
            profile_text_state = ProfileTextState(
                text_id=data.text_id,
                profile_id=profile_id,
                account_id=data.account_id,
                state_data=state_dict,
                saved_at=saved_at,
            )
            db.add(profile_text_state)
            logger.info(f"Saved text state for text={data.text_id}, profile={profile_id}")

        db.commit()

        logger.info(f"[TextState] Saved successfully: text_id={data.text_id}, profile_id={profile_id}")

        return {
            "status": "ok",
            "message": "Text state logged and words added to vocabulary",
            "text_id": data.text_id,
            "profile_id": profile_id,
        }

    except Exception as e:
        logger.error(f"Error logging text state: {e}", exc_info=True)
        db.rollback()
        return JSONResponse(
            {"status": "error", "message": str(e)},
            status_code=500,
        )


def _add_words_to_profile_vocabulary(
    db: Session,
    account_id: Optional[int],
    profile_id: int,
    text_id: int,
    lang: str,
    words: list[dict],
):
    """Add all words from text to profile's vocabulary (Lexeme table)."""
    try:
        added_count = 0
        updated_count = 0

        for word_data in words:
            surface = word_data.get("surface", "")
            lemma = word_data.get("lemma") or surface
            pos = word_data.get("pos", "UNKNOWN")

            # Check if lexeme already exists for this profile
            existing = (
                db.query(Lexeme)
                .filter(
                    Lexeme.profile_id == profile_id,
                    Lexeme.lang == lang,
                    Lexeme.lemma == lemma,
                    Lexeme.pos == pos,
                )
                .first()
            )

            if existing:
                # Update existing lexeme
                existing.exposures += 1
                existing.last_seen_at = datetime.now(timezone.utc)
                updated_count += 1
            else:
                # Create new lexeme for this profile
                lexeme = Lexeme(
                    account_id=account_id,
                    profile_id=profile_id,
                    lang=lang,
                    lemma=lemma,
                    pos=pos,
                    first_seen_at=datetime.now(timezone.utc),
                    last_seen_at=datetime.now(timezone.utc),
                    exposures=1,
                    clicks=0,
                    distinct_texts=1,
                    # SRS fields (will be computed later)
                    stability=0.0,
                    difficulty=0.5,
                )
                db.add(lexeme)
                added_count += 1
                # Flush immediately to avoid UNIQUE constraint issues
                db.flush()

        db.commit()
        logger.info(f"[TextState] Vocabulary: added={added_count}, updated={updated_count}, total_words={len(words)}")

        return {
            "added": added_count,
            "updated": updated_count,
            "total": len(words)
        }

    except Exception as e:
        logger.error(f"Error adding words to vocabulary: {e}", exc_info=True)
        raise


@router.get("/reading/{text_id}/status")
def get_text_status(
    text_id: int,
    db: Session = Depends(get_db),
):
    """Check if text state is ready."""
    try:
        state = db.query(TextState).filter(TextState.text_id == text_id).first()

        if not state:
            return {"status": "not_found", "ready": False}

        return {
            "status": "found",
            "ready": state.status == "ready",
            "build_status": state.status,
            "has_content": state.has_content,
            "has_words": state.has_words,
            "has_translations": state.has_translations,
        }

    except Exception as e:
        logger.error(f"Error checking text status: {e}")
        return {"status": "error", "ready": False}
