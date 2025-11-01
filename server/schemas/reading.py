from __future__ import annotations

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel


class NextPayload(BaseModel):
    # Minimal structure for session analytics; extend as needed
    session_id: Optional[str] = None
    read_time_ms: Optional[int] = None
    translation_scope: Optional[str] = None
    translation_shown_at_ms: Optional[int] = None
    clicks: Optional[List[Dict[str, Any]]] = None
    encountered: Optional[List[Dict[str, Any]]] = None


class LookupEvent(BaseModel):
    # Placeholder for lookup events schema
    start: Optional[int] = None
    end: Optional[int] = None
    surface: Optional[str] = None
    lemma: Optional[str] = None
    pos: Optional[str] = None
