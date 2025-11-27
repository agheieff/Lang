from __future__ import annotations

from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class WordLookup(BaseModel):
    """A single word lookup event from the client."""
    word: str = Field(..., min_length=1, description="Surface form of the word")
    lemma: Optional[str] = Field(None, description="Dictionary form")
    pos: Optional[str] = Field(None, description="Part of speech")
    span_start: int = Field(..., ge=0, description="Start position in text")
    span_end: int = Field(..., ge=0, description="End position in text")
    translations: List[str] = Field(default_factory=list, description="Translations shown")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of lookup")
    
    @field_validator('span_end')
    @classmethod
    def span_end_after_start(cls, v, info):
        if 'span_start' in info.data and v < info.data['span_start']:
            raise ValueError('span_end must be >= span_start')
        return v


class WordInteraction(BaseModel):
    """A word interaction event (click, exposure, etc.)."""
    word: str = Field(..., min_length=1)
    lemma: Optional[str] = None
    pos: Optional[str] = None
    span_start: int = Field(..., ge=0)
    span_end: int = Field(..., ge=0)
    event_type: Literal["click", "exposure", "hover"] = Field(..., description="Type of interaction")
    timestamp: Optional[str] = None


class SessionAnalytics(BaseModel):
    """Analytics data for a reading session."""
    reading_time_ms: Optional[int] = Field(None, ge=0)
    total_words: Optional[int] = Field(None, ge=0)
    unique_lookups: Optional[int] = Field(None, ge=0)
    average_reading_speed_wpm: Optional[float] = Field(None, ge=0)
    completion_status: Optional[Literal["started", "in_progress", "finished", "abandoned"]] = None


class NextPayload(BaseModel):
    """Payload sent when user moves to next text."""
    # Session identification
    session_id: Optional[str] = None
    text_id: Optional[int] = Field(None, ge=1)
    
    # Timing
    read_time_ms: Optional[int] = Field(None, ge=0)
    opened_at: Optional[int] = Field(None, description="Epoch timestamp when text was opened")
    
    # User interactions
    lookups: List[WordLookup] = Field(default_factory=list, description="Word lookups")
    interactions: List[WordInteraction] = Field(default_factory=list, description="Word interactions")
    
    # Translation panel
    translation_scope: Optional[str] = None
    translation_shown_at_ms: Optional[int] = Field(None, ge=0)
    
    # Analytics
    analytics: Optional[SessionAnalytics] = None
    
    # Preferences
    length_preference: Optional[Literal["longer", "shorter"]] = Field(
        None, description="User's preference for next text length"
    )
    
    # Legacy fields for backwards compatibility
    clicks: Optional[List[Dict[str, Any]]] = None
    encountered: Optional[List[Dict[str, Any]]] = None


class LookupEvent(BaseModel):
    """Individual lookup event for API."""
    start: int = Field(..., ge=0, description="Start position in text")
    end: int = Field(..., ge=0, description="End position in text")
    surface: str = Field(..., min_length=1, description="Surface form")
    lemma: Optional[str] = None
    pos: Optional[str] = None
    
    @field_validator('end')
    @classmethod
    def end_after_start(cls, v, info):
        if 'start' in info.data and v < info.data['start']:
            raise ValueError('end must be >= start')
        return v


class ReadingLookupCreate(BaseModel):
    """Schema for creating a reading lookup record."""
    text_id: int = Field(..., ge=1)
    lang: str = Field(..., min_length=2, max_length=16)
    target_lang: str = Field(..., min_length=2, max_length=8)
    surface: str = Field(..., min_length=1)
    span_start: int = Field(..., ge=0)
    span_end: int = Field(..., ge=0)
    lemma: Optional[str] = None
    pos: Optional[str] = None
    translations: List[str] = Field(default_factory=list)
    
    @field_validator('span_end')
    @classmethod
    def validate_span(cls, v, info):
        if 'span_start' in info.data and v < info.data['span_start']:
            raise ValueError('span_end must be >= span_start')
        return v
