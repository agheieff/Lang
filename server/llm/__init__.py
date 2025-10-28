"""
LLM module - API client and prompt building for language learning.
"""

from .client import chat_complete, resolve_model
from .prompts import (
    build_reading_prompt,
    build_translation_prompt,
    build_structured_translation_prompt,
    build_word_translation_prompt,
    PromptSpec,
    TranslationSpec,
)
from ..services.word_selection import pick_words, compose_level_hint, estimate_level

__all__ = [
    "chat_complete",
    "resolve_model",
    "build_reading_prompt",
    "build_translation_prompt",
    "build_structured_translation_prompt",
    "build_word_translation_prompt",
    "PromptSpec",
    "TranslationSpec",
    "pick_words",
    "compose_level_hint",
    "estimate_level",
]