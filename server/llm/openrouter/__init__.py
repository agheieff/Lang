"""
Local OpenRouter client implementation.
Copied from /home/agheieff/Arcadia/libs/openrouter/openrouter/openrouter.py
"""
from .openrouter import (
    complete,
    astream,
    StreamController,
    Chunk,
    with_api_key,
    build_or_messages,
    content_from,
    content_from_file,
    consume_and_drop
)

__all__ = [
    "complete",
    "astream",
    "StreamController",
    "Chunk",
    "with_api_key",
    "build_or_messages",
    "content_from",
    "content_from_file",
    "consume_and_drop"
]