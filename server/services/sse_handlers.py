"""
SSE (Server-Sent Events) handlers for reading events.

Extracted from routes/reading.py to improve modularity.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Optional

from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..account_db import get_db
from ..auth import Account
from ..db import get_global_db
from ..deps import get_current_account
from ..models import Profile
from ..services.notification_service import get_notification_service
from ..services.generation_orchestrator import get_generation_orchestrator
from ..services.text_state_service import get_text_state_service
from ..settings import get_settings

logger = logging.getLogger(__name__)
_SETTINGS = get_settings()
MAX_WAIT_SEC = float(_SETTINGS.NEXT_READY_MAX_WAIT_SEC)


async def _tick(db: Session, interval: float = 0.5) -> None:
    """Rollback, expire, and sleep to advance long‑poll loops safely."""
    try:
        db.rollback()
    except Exception:
        logger.debug("rollback failed in _tick", exc_info=True)
    try:
        db.expire_all()
    except Exception:
        logger.debug("expire_all failed in _tick", exc_info=True)
    await asyncio.sleep(interval)


async def wait_until(pred: Callable, timeout: float, db: Session, interval: float = 0.5):
    """Poll pred() until it returns a truthy value or timeout elapses.

    Returns the last pred() value (truthy or falsy) at timeout.
    """
    deadline = time.time() + max(0.0, float(timeout))
    while time.time() < deadline:
        val = pred()
        if val:
            return val
        await _tick(db, interval)
    return pred()


def reading_events_sse(
    account: Account = Depends(get_current_account),
    global_db: Session = Depends(get_global_db),
) -> StreamingResponse:
    """
    SSE endpoint for real-time updates about reading events.
    """
    prof = global_db.query(Profile).filter(
        Profile.account_id == account.id
    ).first()
    
    if not prof:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    # Create SSE stream for this account
    notification_service = get_notification_service()
    return notification_service.create_sse_stream(account.id, prof.lang)


def next_ready_sse(
    wait: Optional[int] = None,
    account: Account = Depends(get_current_account),
    global_db: Session = Depends(get_global_db),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    SSE endpoint for next text ready notifications.
    """
    prof = global_db.query(Profile).filter(
        Profile.account_id == account.id
    ).first()
    
    if not prof:
        return create_error_response("Profile not found")

    user_content_service = get_text_state_service()
    
    event_stream_generator = create_event_stream_generator(
        user_content_service, global_db, account, prof
    )
    
    return StreamingResponse(
        content=event_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive", 
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
            "X-Accel-Buffering": "no"
        }
    )


def create_error_response(error_message: str) -> StreamingResponse:
    """Create SSE error response."""
    return StreamingResponse(
        content=f"data: {{\"ready\": false, \"text_id\": null, \"error\": \"{error_message}\"}}\n\n",
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


def create_event_stream_generator(user_content_service, global_db, account, prof):
    """Create generator function for SSE event stream."""
    
    async def event_stream():
        try:
            # Send initial status
            yield "data: {\"ready\": false, \"text_id\": null, \"status\": \"connecting\"}\n\n"
            
            last_status = None
            
            while True:
                try:
                    try:
                        # Note: We need account_db here, but using global_db for rollback
                        global_db.rollback()
                    except Exception:
                        pass
                    
                    status = user_content_service.check_next_ready(global_db, account.id, prof.lang)
                    
                    current_status_dict = {
                        "ready": status.ready,
                        "text_id": status.text_id,
                        "ready_reason": status.reason,
                        "retry_info": status.retry_info,
                        "status": status.status
                    }

                    if current_status_dict != last_status:
                        data = json.dumps(current_status_dict)
                        yield f"data: {data}\n\n"
                        last_status = current_status_dict.copy()
                    
                    if status.ready:
                        yield f"data: {{\"ready\": true, \"text_id\": {status.text_id}, \"ready_reason\": \"{status.reason}\", \"status\": \"complete\"}}\n\n"
                        break
                    
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    logger.error(f"Error in SSE stream: {e}")
                    yield f"data: {{\"error\": \"stream_error\", \"message\": \"{str(e)}\"}}\n\n"
                    break
        
        except asyncio.CancelledError:
            logger.debug("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield f"data: {{\"error\": \"stream_error\", \"message\": \"{str(e)}\"}}\n\n"
        finally:
            logger.debug("SSE connection closed")
    
    return event_stream
