from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict, Any, Literal
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..models import (
    Profile,
    ReadingText,
    ReadingWordGloss,
    ReadingTextTranslation,
    ReadingLookup,
    ProfileTextRead,
)
from .selection_service import SelectionService
from .readiness_service import ReadinessService
from .title_extraction_service import TitleExtractionService
from .state_manager import GenerationStateManager
from .session_processing_service import SessionProcessingService

logger = logging.getLogger(__name__)

@dataclass
class ReadingContext:
    status: Literal["loading", "generating", "ready", "error"]
    text_id: Optional[int] = None
    content: Optional[str] = None
    words: List[ReadingWordGloss] = field(default_factory=list)
    title: Optional[str] = None
    title_words: List[Any] = field(default_factory=list)
    title_translation: Optional[str] = None
    is_fully_ready: bool = False
    sse_endpoint: Optional[str] = None
    session_state: Optional[Dict] = None
    is_next_ready: bool = False
    next_ready_reason: str = "waiting"

@dataclass
class ReadinessStatus:
    ready: bool
    text_id: Optional[int]
    reason: str = "waiting"
    retry_info: Optional[Dict[str, Any]] = None
    status: str = "waiting"

class ReadingViewService:
    """
    Service to handle view logic for reading pages.
    Now supports global/per-account DB split:
    - account_db: Profile, ProfileTextRead, lookups, logs
    - global_db: ReadingText, ReadingTextTranslation, ReadingWordGloss
    """

    def __init__(self):
        self.selection_service = SelectionService()
        self.state_manager = GenerationStateManager()
        self.readiness_service = ReadinessService()
        self.title_service = TitleExtractionService()
        self.session_service = SessionProcessingService()

    def get_current_reading_context(
        self,
        account_db: Session,  # Per-account DB
        global_db: Session,   # Global DB
        account_id: int,
    ) -> ReadingContext:
        """
        Prepare all data needed to render the current reading view.
        """
        start_time = time.time()
        
        # Get user profile (per-account)
        prof = account_db.query(Profile).filter(Profile.account_id == account_id).first()
        if not prof:
            logger.error(f"Profile not found for account_id={account_id}")
            return ReadingContext(status="error")

        lang = prof.lang
        target_lang = prof.target_lang

        # Select text (current or new)
        text_obj = None
        try:
            text_obj = self.selection_service.pick_current_or_new(
                account_db, global_db, account_id, lang
            )
        except Exception as e:
            logger.error(f"SelectionService failed: {e}", exc_info=True)
            # Fallback
            try:
                text_obj = self.state_manager.get_unopened_text(global_db, account_id, lang)
            except Exception:
                pass

        # Determine status if no text
        if text_obj is None:
            generating = self.state_manager.get_generating_text(global_db, account_id, lang)
            status = "generating" if generating else "loading"
            return ReadingContext(status=status)

        text_id = text_obj.id
        
        # Get title data (from global DB)
        raw_title, title_translation = self.title_service.get_title(
            global_db, text_id, target_lang
        )
        title_words_list = self.title_service.get_title_words(
            global_db, text_id, target_lang
        )

        # Get words (from global DB)
        rows = (
            global_db.query(ReadingWordGloss)
            .filter(
                ReadingWordGloss.text_id == text_id,
                ReadingWordGloss.target_lang == target_lang
            )
            .order_by(
                ReadingWordGloss.span_start.asc().nullsfirst(),
                ReadingWordGloss.span_end.asc().nullsfirst()
            )
            .all()
        )

        # Check readiness
        try:
            is_fully_ready, _ = self.readiness_service.evaluate(global_db, text_obj)
        except Exception:
            is_fully_ready = (len(rows) > 0)

        sse_endpoint = f"/reading/events/sse?text_id={text_id}"

        # Retrieve any persisted session state
        session_state = self.session_service.get_persisted_session_state(
            account_db, account_id, text_id
        )

        # Check if any backup/next text is ready
        is_next_ready = False
        next_ready_reason = "waiting"
        try:
            backup_text, backup_reason = self.readiness_service.first_ready_backup(
                global_db, account_db, lang, target_lang, prof.id, exclude_text_id=text_id
            )
            if backup_text:
                is_next_ready = True
                next_ready_reason = backup_reason
        except Exception:
            logger.debug("Failed to check backup text readiness", exc_info=True)

        logger.info(f"[READING_VIEW] Context prepared in {time.time() - start_time:.3f}s")

        return ReadingContext(
            status="ready",
            text_id=text_id,
            content=text_obj.content or "",
            words=rows,
            title=raw_title,
            title_words=title_words_list,
            title_translation=title_translation if isinstance(title_translation, str) else None,
            is_fully_ready=is_fully_ready,
            sse_endpoint=sse_endpoint,
            session_state=session_state,
            is_next_ready=is_next_ready,
            next_ready_reason=next_ready_reason
        )

    def check_next_text_readiness(
        self, 
        account_db: Session,
        global_db: Session, 
        account_id: int, 
        lang: str,
        target_lang: str,
        profile_id: int,
        force_check: bool = False
    ) -> ReadinessStatus:
        """
        Check if the next text is ready for the user.
        """
        # Check if ANY backup text is ready
        rt, reason = self.readiness_service.first_ready_backup(
            global_db, account_db, lang, target_lang, profile_id
        )
        
        if rt and reason in ("both", "grace", "content_only"):
            return ReadinessStatus(ready=True, text_id=rt.id, reason=reason, status="complete")
        
        # Fall back to checking newest unopened
        rt = self.readiness_service.next_unopened(
            global_db, account_db, lang, target_lang, profile_id
        )
        
        if not rt:
            return ReadinessStatus(ready=False, text_id=None)

        ready, reason = self.readiness_service.evaluate(global_db, rt)

        is_ready_final = ready and reason in ("both", "grace", "content_only")
        
        return ReadinessStatus(
            ready=is_ready_final,
            text_id=rt.id,
            reason=reason if ready else "waiting",
            retry_info=None,
            status="complete" if is_ready_final else "waiting"
        )
    
    # Legacy single-session methods for backwards compatibility
    def get_current_reading_context_legacy(self, db: Session, account_id: int) -> ReadingContext:
        """Legacy method using single DB session (treats it as both)."""
        return self.get_current_reading_context(db, db, account_id)
