from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, List, Any
from pydantic import BaseModel

class SessionWord(BaseModel):
    surface: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    translation: Optional[str] = None
    looked_up_at: Optional[int] = None  # Timestamp in ms

class SessionSentence(BaseModel):
    text: str
    translation: Optional[str] = None
    translated_at: Optional[int] = None
    words: List[SessionWord] = []

class SessionParagraph(BaseModel):
    text: str
    translation: Optional[str] = None
    translated_at: Optional[int] = None
    sentences: List[SessionSentence] = []

class TitleState(BaseModel):
    text: str
    full_translation: Optional[str] = None
    translated_at: Optional[int] = None
    words: List[SessionWord] = []

class SessionAnalytics(BaseModel):
    total_words: int = 0
    words_looked_up: int = 0
    lookup_rate: float = 0.0
    reading_time_ms: int = 0
    average_reading_speed_wpm: int = 0
    completion_status: str = "in_progress"

class TextSessionState(BaseModel):
    session_id: str
    text_id: int
    lang: str
    target_lang: str
    opened_at: int  # Timestamp in ms
    
    title: Optional[TitleState] = None
    full_text: Optional[str] = None
    full_translation: Optional[str] = None
    
    paragraphs: List[SessionParagraph] = []
    analytics: Optional[SessionAnalytics] = None
    
    # Allow extra fields for forward compatibility
    class Config:
        extra = "ignore"
