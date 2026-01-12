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


def _get_active_profile(request: Request, db: Session, account_id: int) -> Optional[Profile]:
    """Get active profile from cookie, falling back to first profile."""
    active_profile_id = request.cookies.get("active_profile_id")
    if active_profile_id:
        try:
            return db.query(Profile).filter(
                Profile.id == int(active_profile_id),
                Profile.account_id == account_id
            ).first()
        except ValueError:
            pass
    return _get_profile_for_account(db, account_id)


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
DEMO_TEXT = """🔄 Generating texts...

We're creating personalized reading texts for you. This takes 45-90 seconds.

The system works in the background:
• Startup generation: 5 texts immediately
• Background worker: runs every 60 seconds
• On-demand: triggers when you click "Next Text" with empty queue

What's happening:
• Texts are shared across all users with your language pair
• Topics vary: fiction, news, science, technology, history, daily_life, culture, sports, business
• Difficulty adapts to your level automatically

Current status: Waiting for generation to complete...

💡 Click "Next Text" to trigger immediate generation, or just refresh this page in a minute."""


@router.get("/reading", response_class=HTMLResponse)
def reading_page(
    request: Request,
    db: Session = Depends(get_db),
):
    """Reading practice page."""
    account_id = _get_account_id(request)
    if account_id is None:
        return HTMLResponse(
            "<div><h1>Please log in</h1><p><a href='/login'>Login</a></p></div>"
        )

    profile = _get_active_profile(request, db, account_id)
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


@router.post("/reading/generate-now")
def generate_text_now(
    request: Request,
    db: Session = Depends(get_db),
):
    """Trigger immediate text generation for current profile.

    Called when there are no ready texts available.
    Generates 1-3 texts depending on pool size.
    Returns immediately with generation status.
    """
    import asyncio
    from server.services.content import generate_text_content, generate_translations
    from server.services.recommendation import select_topic_for_profile, select_diverse_topic

    account_id = _get_account_id(request)
    if not account_id:
        return JSONResponse(
            {"status": "error", "message": "Not authenticated"},
            status_code=401,
        )

    profile = _get_active_profile(request, db, account_id)
    if not profile:
        return JSONResponse(
            {"status": "error", "message": "No profile found"},
            status_code=404,
        )

    # Check if there are already texts being generated for this profile
    from server.services.background_worker import _get_locked_profile_ids
    locked_ids = _get_locked_profile_ids()
    if profile.id in locked_ids:
        return JSONResponse({
            "status": "generating",
            "message": "Text generation already in progress"
        })

    # Count existing ready texts
    ready_count = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == profile.lang,
            ReadingText.target_lang == profile.target_lang,
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
        )
        .count()
    )

    # Determine how many texts to generate
    from server.services.background_worker import URGENT_POOL_THRESHOLD
    if ready_count <= URGENT_POOL_THRESHOLD:
        texts_to_generate = min(3, 3)  # Generate up to 3 texts
    else:
        texts_to_generate = 1

    logger.info(
        f"On-demand generation for profile {profile.id}: "
        f"{profile.lang}->{profile.target_lang}, "
        f"generating {texts_to_generate} texts (current pool: {ready_count})"
    )

    # Trigger background generation (fire and forget)
    async def generate_in_background():
        try:
            from server.services.background_worker import (
                _acquire_generation_lock,
                _release_generation_lock
            )

            # Acquire lock
            if not _acquire_generation_lock(profile.id):
                logger.warning(f"Could not acquire lock for on-demand generation")
                return

            try:
                for i in range(texts_to_generate):
                    # Select topic
                    preferred_topic = select_topic_for_profile(db, profile)
                    selected_topic = select_diverse_topic(db, profile, preferred_topic)

                    logger.info(f"On-demand generation {i+1}/{texts_to_generate}: topic={selected_topic}")

                    # Generate text
                    text_obj = await generate_text_content(
                        account_id=profile.account_id,
                        profile_id=profile.id,
                        lang=profile.lang,
                        target_lang=profile.target_lang,
                        profile=profile,
                        topic=selected_topic,
                    )

                    if not text_obj:
                        logger.error(f"Failed to generate text {i+1}/{texts_to_generate}")
                        continue

                    # Generate translations
                    success = await generate_translations(
                        text_id=text_obj.id,
                        lang=profile.lang,
                        target_lang=profile.target_lang,
                    )

                    if success:
                        logger.info(f"On-demand generation {i+1}/{texts_to_generate} completed: text_id={text_obj.id}")
                    else:
                        logger.error(f"On-demand generation {i+1}/{texts_to_generate} failed translations")

            finally:
                _release_generation_lock(profile.id)

        except Exception as e:
            logger.error(f"Error in on-demand generation: {e}", exc_info=True)

    # Start background task
    asyncio.create_task(generate_in_background())

    return JSONResponse({
        "status": "generating",
        "message": f"Generating {texts_to_generate} text(s) in the background",
        "texts_to_generate": texts_to_generate,
        "current_pool_size": ready_count,
    })


@router.get("/reading/next")
def get_next_text(
    request: Request,
    db: Session = Depends(get_db),
):
    """Get the next text for reading.

    If no ready text is available, triggers on-demand generation.
    Returns:
    - 200 with text_id if a ready text is available
    - 202 with generating status if generation was triggered
    - 404 if no profile found
    """
    account_id = _get_account_id(request)
    if not account_id:
        return JSONResponse(
            {"status": "error", "message": "Not authenticated"},
            status_code=401,
        )

    profile = _get_active_profile(request, db, account_id)
    if not profile:
        return JSONResponse(
            {"status": "error", "message": "No profile found"},
            status_code=404,
        )

    # Try to select best text
    next_text = select_best_text(db, profile)

    if next_text:
        # Mark text as current and record that it was read
        profile.current_text_id = next_text.id

        # Mark previous text as read if exists
        if profile.current_text_id and profile.current_text_id != next_text.id:
            existing_read = db.query(ProfileTextRead).filter(
                ProfileTextRead.profile_id == profile.id,
                ProfileTextRead.text_id == profile.current_text_id
            ).first()

            if not existing_read:
                read_record = ProfileTextRead(
                    profile_id=profile.id,
                    text_id=profile.current_text_id,
                    read_at=datetime.now(timezone.utc),
                )
                db.add(read_record)

        db.commit()

        return JSONResponse({
            "status": "ready",
            "text_id": next_text.id,
            "title": next_text.title,
            "topic": next_text.topic,
        })
    else:
        # No ready text - trigger on-demand generation
        import asyncio
        from server.services.content import generate_text_content, generate_translations
        from server.services.recommendation import select_topic_for_profile, select_diverse_topic

        # Check if already generating
        from server.services.background_worker import _get_locked_profile_ids, URGENT_POOL_THRESHOLD
        locked_ids = _get_locked_profile_ids()

        if profile.id in locked_ids:
            return JSONResponse({
                "status": "generating",
                "message": "Text generation already in progress",
                "text_id": None,
            }, status_code=202)

        # Count existing texts
        ready_count = (
            db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
            )
            .count()
        )

        # Determine how many to generate
        texts_to_generate = min(3, 3) if ready_count <= URGENT_POOL_THRESHOLD else 1

        logger.info(
            f"[NextText] No ready text for profile {profile.id}, "
            f"triggering generation of {texts_to_generate} text(s)"
        )

        # Trigger background generation
        async def generate_in_background():
            try:
                from server.services.background_worker import (
                    _acquire_generation_lock,
                    _release_generation_lock
                )

                if not _acquire_generation_lock(profile.id):
                    return

                try:
                    for i in range(texts_to_generate):
                        preferred_topic = select_topic_for_profile(db, profile)
                        selected_topic = select_diverse_topic(db, profile, preferred_topic)

                        text_obj = await generate_text_content(
                            account_id=profile.account_id,
                            profile_id=profile.id,
                            lang=profile.lang,
                            target_lang=profile.target_lang,
                            profile=profile,
                            topic=selected_topic,
                        )

                        if text_obj:
                            await generate_translations(
                                text_id=text_obj.id,
                                lang=profile.lang,
                                target_lang=profile.target_lang,
                            )
                            logger.info(f"[NextText] Generated text {text_obj.id}")
                finally:
                    _release_generation_lock(profile.id)

            except Exception as e:
                logger.error(f"[NextText] Generation error: {e}", exc_info=True)

        asyncio.create_task(generate_in_background())

        return JSONResponse({
            "status": "generating",
            "message": f"Generating {texts_to_generate} text(s)",
            "text_id": None,
            "texts_to_generate": texts_to_generate,
        }, status_code=202)


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
