"""Server-Sent Events (SSE) notification service.

This module is used by both the home page and the reading view to
stream generation progress events. It needs to coordinate events
coming from background threads with async FastAPI request handlers
without ever blocking the event loop.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from typing import Any, Dict, List, Optional, Set
from weakref import WeakSet

from fastapi import Response
from fastapi.responses import StreamingResponse

from ..db import GlobalSessionLocal, get_global_db
from ..models import Profile

logger = logging.getLogger(__name__)


class SSEEvent:
    """A server-sent event."""
    def __init__(self, event_type: str, data: Dict[str, Any], event_id: Optional[str] = None):
        self.type = event_type
        self.data = data
        self.id = event_id or str(int(datetime.now(timezone.utc).timestamp() * 1000))
    
    def format(self) -> str:
        """Format the event for SSE protocol. Must end with \\n\\n."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.type:
            lines.append(f"event: {self.type}")
        lines.append(f"data: {json.dumps(self.data)}")
        return "\n".join(lines) + "\n\n"


class ClientConnection:
    """Represents a client SSE connection."""
    def __init__(self, account_id: int, lang: str, queue: Queue):
        self.account_id = account_id
        self.lang = lang
        self.queue = queue
        self.last_ping = datetime.now(timezone.utc)
    
    def put_event(self, event: SSEEvent) -> bool:
        """Try to put an event in the queue.

        Uses a standard :class:`queue.Queue` so producers in background
        threads never block the asyncio event loop. The async SSE
        consumer thread pulls from this queue via ``run_in_executor``.
        """
        try:
            self.queue.put_nowait(event)
            return True
        except (Full, ValueError):
            return False


class NotificationService:
    """
    Manages SSE notifications for text generation progress.
    
    This service provides a clean way to:
    - Subscribe clients to specific notification streams
    - Broadcast events to relevant clients
    - Handle connection lifecycle gracefully
    - Prevent memory leaks and duplication
    
    Events:
        - generation_started: Text generation has begun
        - content_ready: Text content is available
        - translations_ready: Translations are complete
        - generation_failed: Generation failed
        - heartbeat: Regular keep-alive signal
    """
    
    def __init__(self):
        # Use regular dictionary with connection limit per account to prevent DoS
        self._connections: Dict[str, Set[ClientConnection]] = {}
        self._connection_count = 0  # For generating unique connection IDs
        # Track connection counts per account to enforce limits
        self._connection_counts_per_account: Dict[int, int] = {}
        self._MAX_CONNECTIONS_PER_ACCOUNT = 5  # Prevent DoS
    
    def subscribe(self, account_id: int, lang: str) -> Optional[Queue]:
        """
        Subscribe a client to notifications for their language.
        
        Returns:
            Queue that will receive SSEEvent objects or None if limit exceeded
        """
        # Check connection limit per account
        current_count = self._connection_counts_per_account.get(account_id, 0)
        if current_count >= self._MAX_CONNECTIONS_PER_ACCOUNT:
            logger.warning(f"[SSE] Connection limit exceeded for account={account_id}")
            return None
        
        key = f"{account_id}:{lang}"
        queue: Queue = Queue(maxsize=50)  # thread-safe; consumed asynchronously
        connection = ClientConnection(account_id, lang, queue)
        
        if key not in self._connections:
            self._connections[key] = set()
        
        self._connections[key].add(connection)
        # Update connection count for account
        self._connection_counts_per_account[account_id] = current_count + 1
        
        logger.info(f"[SSE] Client subscribed: account={account_id} lang={lang} (total: {self._connection_counts_per_account[account_id]})")
        
        # Send initial connection event
        connection.put_event(SSEEvent("connected", {
            "account_id": account_id,
            "lang": lang,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        
        return queue
    
    def unsubscribe(self, account_id: int, lang: str, queue: Queue) -> None:
        """
        Unsubscribe a client from notifications.
        """
        key = f"{account_id}:{lang}"
        if key in self._connections:
            # Remove connection matching this queue
            to_remove = None
            for conn in self._connections[key]:
                if conn.queue is queue:
                    to_remove = conn
                    break
            
            if to_remove:
                self._connections[key].discard(to_remove)
                # Update connection count for account
                current_count = self._connection_counts_per_account.get(account_id, 0)
                if current_count > 0:
                    self._connection_counts_per_account[account_id] = current_count - 1
                
                logger.info(f"[SSE] Client unsubscribed: account={account_id} lang={lang} (total: {self._connection_counts_per_account.get(account_id, 0)})")
                
                # Clean up empty sets
                if not self._connections[key]:
                    del self._connections[key]
                    # Clean up count for account if no more connections
                    if account_id in self._connection_counts_per_account and self._connection_counts_per_account[account_id] <= 0:
                        del self._connection_counts_per_account[account_id]
    
    def broadcast_to_account(self, account_id: int, lang: str, event: SSEEvent) -> int:
        """
        Broadcast an event to all subscribed connections for an account/language.
        
        Returns:
            Number of clients that received the event
        """
        key = f"{account_id}:{lang}"
        sent_count = 0
        
        # Broadcast via live ClientConnection objects
        if key in self._connections:
            dead_connections = []
            for connection in list(self._connections[key]):
                if connection.put_event(event):
                    sent_count += 1
                else:
                    dead_connections.append(connection)
            
            # Clean up dead connections
            for dead_conn in dead_connections:
                self._connections[key].discard(dead_conn)
                # Update connection count
                current_count = self._connection_counts_per_account.get(account_id, 0)
                if current_count > 0:
                    self._connection_counts_per_account[account_id] = current_count - 1
                # Unsubscribe properly to clean up
                self.unsubscribe(account_id, lang, dead_conn.queue)
            
            if dead_connections:
                logger.info(f"[SSE] Cleaned up {len(dead_connections)} dead connections for account={account_id}")
        
        return sent_count
    
    def send_generation_started(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that text generation has started."""
        event = SSEEvent("generation_started", {
            "text_id": text_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] generation_started: account={account_id} text_id={text_id}")
    
    def send_content_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that text content is ready."""
        event = SSEEvent("content_ready", {
            "text_id": text_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] content_ready: account={account_id} text_id={text_id}")
    
    def send_translations_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that translations are complete."""
        event = SSEEvent("translations_ready", {
            "text_id": text_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] translations_ready: account={account_id} text_id={text_id}")
    
    def send_generation_failed(self, account_id: int, lang: str, text_id: int, error: str) -> None:
        """Notify that generation failed."""
        event = SSEEvent("generation_failed", {
            "text_id": text_id,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] generation_failed: account={account_id} text_id={text_id}")
    
    def send_next_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that the next text is fully ready to read."""
        event = SSEEvent("next_ready", {
            "text_id": text_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] next_ready: account={account_id} text_id={text_id}")
    
    def create_sse_stream(self, account_id: int, lang: str) -> StreamingResponse:
        """
        Create a FastAPI SSE streaming response for a client.
        """
        queue = self.subscribe(account_id, lang)
        if queue is None:
            # Connection limit exceeded
            return StreamingResponse(
                iter(["data: {\"error\": \"Connection limit exceeded\"}\n\n"]),
                media_type="text/event-stream",
                status_code=429
            )

        # Pre-check current text readiness outside the async generator
        initial_events = []
        try:
            from ..db import db_manager
            from ..services.text_state_service import get_text_state_service
            
            with db_manager.read_only(account_id) as db:
                content_service = get_text_state_service()
                
                with get_global_db() as global_db:
                    current_text = content_service.pick_current_or_new(db, global_db, account_id, lang)
                    
                    if current_text:
                        is_ready, reason = content_service.evaluate(global_db, current_text)
                        
                        # Check if content is ready
                        if current_text.content and current_text.generated_at:
                            content_event = SSEEvent("content_ready", {
                                "text_id": current_text.id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            initial_events.append(content_event)
                        
                        # Check if translations are fully ready
                        if is_ready and reason == "both":
                            ready_event = SSEEvent("translations_ready", {
                                "text_id": current_text.id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            initial_events.append(ready_event)
                        
                        # General text ready event
                        text_ready_event = SSEEvent("text_ready", {
                            "text_id": current_text.id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        initial_events.append(text_ready_event)
                    
                    # Check backup text
                    current_id = current_text.id if current_text else None
                    profile = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
                    target_lang = profile.target_lang if profile else "en"
                    backup_text, backup_reason = content_service.first_ready_backup(
                        global_db, db, lang, target_lang=target_lang, profile_id=profile.id if profile else None, exclude_text_id=current_id
                    )
                    if backup_text:
                        next_ready_event = SSEEvent("next_ready", {
                            "text_id": backup_text.id,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        initial_events.append(next_ready_event)
                    
        except Exception as e:
            logger.warning(f"[SSE] Failed to check initial readiness: {e}")
        
        # Send initial connected event
        connected_event = SSEEvent("connected", {
            "account_id": account_id,
            "lang": lang,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        initial_events.append(connected_event)

        async def event_stream():
            try:
                # Yield all pre-computed initial events
                for event in initial_events:
                    yield event.format()

                loop = asyncio.get_running_loop()

                while True:
                    try:
                        # Offload blocking Queue.get() to a worker thread
                        event = await loop.run_in_executor(
                            None, queue.get, True, 30.0
                        )
                        yield event.format()

                    except Empty:
                        # Send heartbeat
                        yield (
                            "data: {\"type\": \"heartbeat\", \"timestamp\": \""
                            + datetime.now(timezone.utc).isoformat()
                            + "\"}\n\n"
                        )
                        
            except Exception as e:
                logger.error(f"[SSE] Stream error for account={account_id}: {e}")
            finally:
                # Clean up when stream ends
                self.unsubscribe(account_id, lang, queue)
                logger.info(f"[SSE] Stream closed for account={account_id}")
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    def get_connection_count(self, account_id: int, lang: str) -> int:
        """Get number of active connections for an account/language."""
        key = f"{account_id}:{lang}"
        if key not in self._connections:
            return 0
        return len(self._connections[key])


# Global instance
_notification_service = NotificationService()


def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    return _notification_service
