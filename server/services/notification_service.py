"""
Server-Sent Events (SSE) notification service.
Manages real-time updates to clients about text generation progress.
"""

import asyncio
import json
import logging
from datetime import datetime
from queue import Queue, Empty
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
        self.id = event_id or str(int(datetime.utcnow().timestamp() * 1000))
    
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
        self.last_ping = datetime.utcnow()
    
    def put_event(self, event: SSEEvent) -> bool:
        """Try to put an event in the queue. Returns False if queue is full."""
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
    
    def subscribe(self, account_id: int, lang: str) -> Queue:
        """
        Subscribe a client to notifications for their language.
        
        Returns:
            Queue that will receive SSEEvent objects
        """
        key = f"{account_id}:{lang}"
        queue = Queue(maxsize=50)  # Limit queue size to prevent memory bloat
        connection = ClientConnection(account_id, lang, queue)
        
        if key not in self._connections:
            self._connections[key] = WeakSet()
        
        self._connections[key].add(connection)
        logger.info(f"[SSE] Client subscribed: account={account_id} lang={lang}")
        
        # Send initial connection event
        connection.put_event(SSEEvent("connected", {
            "account_id": account_id,
            "lang": lang,
            "timestamp": datetime.utcnow().isoformat(),
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
    
    def broadcast_to_account(self, account_id: int, lang: str, event: SSEEvent) -> int:
        """
        Broadcast an event to all subscribed connections for an account/language.
        
        Returns:
            Number of clients that received the event
        """
        key = f"{account_id}:{lang}"
        if key not in self._connections:
            return 0
        
        sent_count = 0
        dead_connections = []
        
        for connection in list(self._connections[key]):
            if connection.put_event(event):
                sent_count += 1
            else:
                # Queue is full, connection is probably dead
                dead_connections.append(connection)
        
        # Clean up dead connections
        for dead_conn in dead_connections:
            self._connections[key].discard(dead_conn)
            
        if dead_connections:
            logger.info(f"[SSE] Cleaned up {len(dead_connections)} dead connections for account={account_id}")
        
        return sent_count
    
    def send_generation_started(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that text generation has started."""
        event = SSEEvent("generation_started", {
            "text_id": text_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] generation_started: account={account_id} text_id={text_id}")
    
    def send_content_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that text content is ready."""
        event = SSEEvent("content_ready", {
            "text_id": text_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] content_ready: account={account_id} text_id={text_id}")
    
    def send_translations_ready(self, account_id: int, lang: str, text_id: int) -> None:
        """Notify that translations are complete."""
        event = SSEEvent("translations_ready", {
            "text_id": text_id,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.broadcast_to_account(account_id, lang, event)
        logger.info(f"[SSE] translations_ready: account={account_id} text_id={text_id}")
    
    def send_generation_failed(self, account_id: int, lang: str, text_id: int, error: str) -> None:
        """Notify that generation failed."""
        event = SSEEvent("generation_failed", {
            "text_id": text_id,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
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
                # Send initial event
                yield "data: {\"type\": \"connected\", \"timestamp\": \"" + datetime.utcnow().isoformat() + "\"}\n\n"
                
                while True:
                    try:
                        # Get event from queue with timeout
                        event = queue.get(timeout=30)  # 30 second timeout
                        yield event.format()
                        
                        # If client has disconnected, queue operations will fail
                        # and we should break out of the loop
                        if event.type == "heartbeat" and not hasattr(event, 'keep_alive'):
                            break
                            
                    except Empty:
                        # Send heartbeat to keep connection alive
                        yield "data: {\"type\": \"heartbeat\", \"timestamp\": \"" + datetime.utcnow().isoformat() + "\"}\n\n"
                        
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
