"""
LLM module - API client and prompt building for language learning.
"""

from .client import chat_complete, get_llm_config
from .prompts import (
    build_reading_prompt,
    build_translation_contexts,
    build_word_translation_prompt,
    PromptSpec,
)

__all__ = [
    "chat_complete",
    "get_llm_config",
    "build_reading_prompt",
    "build_translation_contexts",
    "build_word_translation_prompt",
    "PromptSpec",
]
