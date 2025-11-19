"""Helpers for working with NotificationService SSE-style queues in tests."""

from queue import Empty
from typing import Any


def next_payload(q, timeout: float = 2.0) -> Any:
    """Return the next non-"connected" event from the queue.

    Skips the initial connected handshake message to make tests resilient
    to implementation details of the subscription lifecycle.
    """
    while True:
        ev = q.get(timeout=timeout)
        if getattr(ev, "type", None) == "connected":
            continue
        return ev
