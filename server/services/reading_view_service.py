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
)
from .generation_orchestrator import GenerationOrchestrator
from .selection_service import SelectionService
from .readiness_service import ReadinessService
from .title_extraction_service import TitleExtractionService
from .state_manager import GenerationStateManager
from .retry_service import RetryService
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
    session_state: Optional[Dict] = None  # Restored session state
    is_next_ready: bool = False  # Whether backup/next text is ready
    next_ready_reason: str = "waiting"  # Reason: both, grace, content_only, waiting

@dataclass
class ReadinessStatus:
    ready: bool
    text_id: Optional[int]
    reason: str = "waiting"
    retry_info: Optional[Dict[str, Any]] = None
    status: str = "waiting"  # for SSE: connecting, waiting, complete

class ReadingViewService:
    """
    Service to handle view logic for reading pages.
    Aggregates data from various services to provide a clean context for the UI.
    """

    def __init__(self):
        self.orchestrator = GenerationOrchestrator()
        self.selection_service = SelectionService()
        self.state_manager = GenerationStateManager()
        self.readiness_service = ReadinessService()
        self.title_service = TitleExtractionService()
        self.session_service = SessionProcessingService()

    def get_current_reading_context(self, db: Session, account_id: int) -> ReadingContext:
        """
        Prepare all data needed to render the current reading view.
        """
        start_time = time.time()
        
        # Get user profile
        prof = db.query(Profile).filter(Profile.account_id == account_id).first()
        if not prof:
            logger.error(f"Profile not found for account_id={account_id}")
            return ReadingContext(status="error")

        lang = prof.lang

        # Select text (current or new)
        text_obj = None
        try:
            text_obj = self.selection_service.pick_current_or_new(db, account_id, lang)
        except Exception as e:
            logger.error(f"SelectionService failed: {e}", exc_info=True)
            # Fallback
            try:
                text_obj = self.state_manager.get_unopened_text(db, account_id, lang)
            except Exception:
                pass

        # Ensure generation pipeline is active
        try:
            self.orchestrator.ensure_text_available(db, account_id, lang)
        except Exception:
            logger.exception("Failed to ensure text available")

        # Determine status if no text
        if text_obj is None:
            generating = self.state_manager.get_generating_text(db, account_id, lang)
            status = "generating" if generating else "loading"
            return ReadingContext(status=status)

        # Prepare text data
        text_id = text_obj.id
        
        # Get title data
        raw_title, title_translation = self.title_service.get_title(db, account_id, text_id)
        title_words_list = self.title_service.get_title_words(db, account_id, text_id)

        # Get words
        rows = (
            db.query(ReadingWordGloss)
            .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == text_id)
            .order_by(ReadingWordGloss.span_start.asc().nullsfirst(), ReadingWordGloss.span_end.asc().nullsfirst())
            .all()
        )

        # Check readiness
        try:
            is_fully_ready, _ = self.readiness_service.evaluate(db, text_obj, account_id)
        except Exception:
            is_fully_ready = (len(rows) > 0)

        sse_endpoint = f"/reading/events/sse?text_id={text_id}"

        # Retrieve any persisted session state to restore progress
        session_state = self.session_service.get_persisted_session_state(db, account_id, text_id)

        # Check if backup/next text is ready
        is_next_ready = False
        next_ready_reason = "waiting"
        try:
            backup_text = self.readiness_service.next_unopened(db, account_id, lang)
            if backup_text and backup_text.id != text_id:
                backup_ready, backup_reason = self.readiness_service.evaluate(db, backup_text, account_id)
                if backup_ready and backup_reason in ("both", "grace", "content_only"):
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
        db: Session, 
        account_id: int, 
        lang: str, 
        force_check: bool = False
    ) -> ReadinessStatus:
        """
        Check if the next text is ready for the user.
        Used by both polling endpoint and SSE stream.
        """
        # Ensure something is queued
        try:
            self.orchestrator.check_and_retry_failed_texts(db, account_id, lang)
            self.orchestrator.ensure_text_available(db, account_id, lang)
        except Exception:
            logger.debug("Background tasks in check_next_text_readiness failed", exc_info=True)

        rt = self.readiness_service.next_unopened(db, account_id, lang)
        
        if not rt:
            return ReadinessStatus(ready=False, text_id=None)

        # Check for manual overrides
        if getattr(rt, "content", None):
             if self.readiness_service.consume_if_valid(db, account_id, lang):
                 return ReadinessStatus(ready=True, text_id=rt.id, reason="manual_override", status="complete")
             if force_check:
                 try:
                     self.readiness_service.force_once(db, account_id, lang)
                     return ReadinessStatus(ready=True, text_id=rt.id, reason="manual", status="complete")
                 except Exception:
                     pass

        ready, reason = self.readiness_service.evaluate(db, rt, account_id)
        
        # Check retry status if waiting
        retry_info = None
        if not ready:
             try:
                failed = self.readiness_service.get_failed_components(db, account_id, rt.id)
                if failed["words"] or failed["sentences"]:
                    can_retry, retry_reason = self.orchestrator.retry_service.can_retry(db, account_id, rt.id, failed)
                    if can_retry:
                        retry_info = {
                            "can_retry": True,
                            "reason": retry_reason,
                            "failed_components": failed
                        }
             except Exception:
                 pass

        # Accept "both" (full), "grace" (partial after timeout), or "content_only" (no translations after long timeout)
        is_ready_final = ready and reason in ("both", "grace", "content_only")
        
        return ReadinessStatus(
            ready=is_ready_final,
            text_id=rt.id,
            reason=reason if ready else "waiting",
            retry_info=retry_info,
            status="complete" if is_ready_final else "waiting"
        )
