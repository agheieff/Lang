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

logger = logging.getLogger(__name__)


class SSEEvent:
    """A server-sent event."""
    def __init__(self, event_type: str, data: Dict[str, Any], event_id: Optional[str] = None):
        self.type = event_type
        self.data = data
        self.id = event_id or str(int(datetime.now(timezone.utc).timestamp() * 1000))
    
    def format(self) -> str:
        """Format the event for SSE protocol."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        if self.type:
            lines.append(f"event: {self.type}")
        lines.append(f"data: {json.dumps(self.data)}")
        lines.append("")  # Empty line to end the event
        return "\n".join(lines)


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
    
    Events:
        - generation_started: Text generation has begun
        - content_ready: Text content is available
        - translations_ready: Translations are complete
        - generation_failed: Generation failed
        - heartbeat: Regular keep-alive signal
    """
    
    def __init__(self):
        # Use WeakSet to avoid memory leaks if clients disconnect without unsubscribing
        self._connections: Dict[str, WeakSet[ClientConnection]] = {}
        self._connection_count = 0  # For generating unique connection IDs
        # Strong references to queues per account/lang to ensure broadcasts work without fragile weakrefs
        self._queues_by_key: Dict[str, set[Queue]] = {}
    
    def subscribe(self, account_id: int, lang: str) -> Queue:
        """
        Subscribe a client to notifications for their language.
        
        Returns:
            Queue that will receive SSEEvent objects
        """
        key = f"{account_id}:{lang}"
        queue: Queue = Queue(maxsize=50)  # thread-safe; consumed asynchronously
        connection = ClientConnection(account_id, lang, queue)
        
        if key not in self._connections:
            self._connections[key] = WeakSet()
        
        self._connections[key].add(connection)
        # Keep a strong ref to the queue so broadcasts always have a live target
        if key not in self._queues_by_key:
            self._queues_by_key[key] = set()
        self._queues_by_key[key].add(queue)
        logger.info(f"[SSE] Client subscribed: account={account_id} lang={lang}")
        
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
                logger.info(f"[SSE] Client unsubscribed: account={account_id} lang={lang}")
                
                # Clean up empty sets
                if not self._connections[key]:
                    del self._connections[key]
        # Also remove from strong queue registry
        if key in self._queues_by_key:
            try:
                self._queues_by_key[key].discard(queue)
                if not self._queues_by_key[key]:
                    del self._queues_by_key[key]
            except Exception:
                pass
    
    def broadcast_to_account(self, account_id: int, lang: str, event: SSEEvent) -> int:
        """
        Broadcast an event to all subscribed connections for an account/language.
        
        Returns:
            Number of clients that received the event
        """
        key = f"{account_id}:{lang}"
        sent_count = 0
        # Broadcast via live ClientConnection objects (best effort)
        if key in self._connections:
            dead_connections = []
            for connection in list(self._connections[key]):
                if connection.put_event(event):
                    sent_count += 1
                else:
                    dead_connections.append(connection)
            for dead_conn in dead_connections:
                self._connections[key].discard(dead_conn)
            if dead_connections:
                logger.info(f"[SSE] Cleaned up {len(dead_connections)} dead connections for account={account_id}")
        # Also broadcast to all registered queues (strong refs) to ensure delivery
        if key in self._queues_by_key:
            for q in list(self._queues_by_key[key]):
                try:
                    q.put_nowait(event)
                    sent_count += 1
                except Exception:
                    # Drop problematic queues
                    try:
                        self._queues_by_key[key].discard(q)
                    except Exception:
                        pass
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
    
    def create_sse_stream(self, account_id: int, lang: str) -> StreamingResponse:
        """
        Create a FastAPI SSE streaming response for a client.
        """
        queue = self.subscribe(account_id, lang)

        async def event_stream():
            try:
                # Check current text readiness immediately on connection
                from ..utils.session_manager import db_manager
                from ..services.readiness_service import ReadinessService
                from ..services.selection_service import SelectionService
                
                try:
                    with db_manager.transaction(account_id) as db:
                        selection_service = SelectionService()
                        current_text = selection_service.pick_current_or_new(db, account_id, lang)
                        
                        if current_text:
                            readiness_service = ReadinessService()
                            is_ready, reason = readiness_service.evaluate(db, current_text, account_id)
                            
                            # Check if content is ready (text exists but translations might not be)
                            if current_text.content and current_text.generated_at:
                                content_event = SSEEvent("content_ready", {
                                    "text_id": current_text.id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                yield content_event.format()
                            
                            # Check if translations are fully ready
                            if is_ready and reason == "both":
                                ready_event = SSEEvent("translations_ready", {
                                    "text_id": current_text.id,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                                yield ready_event.format()
                            # Also send a general "text_ready" event for home page listeners
                            text_ready_event = SSEEvent("text_ready", {
                                "text_id": current_text.id,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            })
                            yield text_ready_event.format()
                except Exception as e:
                    logger.warning(f"[SSE] Failed to check initial readiness: {e}")
                
                # Send initial connected event
                yield (
                    "data: {\"type\": \"connected\", \"timestamp\": \""
                    + datetime.now(timezone.utc).isoformat()
                    + "\"}\n\n"
                )

                loop = asyncio.get_running_loop()

                while True:
                    try:
                        # Offload blocking Queue.get() to a worker thread so we
                        # never block the asyncio event loop. Timeout ensures
                        # we emit heartbeats even when there are no events.
                        event = await loop.run_in_executor(
                            None, queue.get, True, 30.0
                        )
                        yield event.format()

                    except Empty:
                        # No events within the timeout window – send heartbeat
                        # so clients know the connection is still alive.
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
