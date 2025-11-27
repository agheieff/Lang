"""
Service for parsing user interests into topic weights via LLM.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..config import TOPICS, DEFAULT_TOPIC_WEIGHTS
from ..llm.client import chat_complete

logger = logging.getLogger(__name__)

PARSE_INTERESTS_PROMPT = """You are a classifier that maps user interests to topic categories.

Given the user's interests and a list of available topics, output a JSON object with weights (0.5 to 2.0) for each topic based on relevance to their interests.

- 2.0 = strongly matches their interests
- 1.5 = somewhat matches
- 1.0 = neutral (default)
- 0.5 = probably not interested

Available topics: {topics}

User interests: {interests}

Respond with ONLY a valid JSON object mapping topic names to weights. Example:
{{"fiction": 1.8, "history": 1.5, "news": 0.7, "science": 1.0, ...}}

Include ALL topics in your response."""


def parse_interests_to_weights(
    interests_text: str,
    available_topics: Optional[list[str]] = None,
    user_api_key: Optional[str] = None,
) -> dict[str, float]:
    """
    Parse free-form user interests into topic weights using LLM.
    
    Args:
        interests_text: User's description of their interests
        available_topics: List of topics to map to (defaults to config.TOPICS)
        user_api_key: Optional user-specific API key
        
    Returns:
        Dict mapping topic names to weights (0.5-2.0)
    """
    if not interests_text or not interests_text.strip():
        return DEFAULT_TOPIC_WEIGHTS.copy()
    
    topics = available_topics or TOPICS
    
    prompt = PARSE_INTERESTS_PROMPT.format(
        topics=", ".join(topics),
        interests=interests_text.strip(),
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = chat_complete(
            messages=messages,
            provider="openrouter",
            model="openai/gpt-4o-mini",  # Fast and cheap for this task
            user_api_key=user_api_key,
        )
        
        if not response:
            logger.warning("Empty response from LLM for interests parsing")
            return DEFAULT_TOPIC_WEIGHTS.copy()
        
        # Parse JSON from response
        weights = _parse_weights_json(response, topics)
        return weights
        
    except Exception as e:
        logger.error(f"Failed to parse interests: {e}")
        return DEFAULT_TOPIC_WEIGHTS.copy()


def _parse_weights_json(content: str, topics: list[str]) -> dict[str, float]:
    """Parse and validate weights JSON from LLM response."""
    # Try to extract JSON from response (may have markdown code blocks)
    json_str = content.strip()
    if "```" in json_str:
        # Extract from code block
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = json_str[start:end]
    
    try:
        weights = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse JSON from LLM response: {content[:200]}")
        return DEFAULT_TOPIC_WEIGHTS.copy()
    
    if not isinstance(weights, dict):
        return DEFAULT_TOPIC_WEIGHTS.copy()
    
    # Validate and normalize weights
    result = {}
    for topic in topics:
        if topic in weights:
            try:
                w = float(weights[topic])
                # Clamp to valid range
                result[topic] = max(0.2, min(2.0, w))
            except (ValueError, TypeError):
                result[topic] = 1.0
        else:
            result[topic] = 1.0
    
    return result


def get_effective_weight(topic_weights: dict[str, float], topic_path: str) -> float:
    """
    Get effective weight for a topic, walking up the path hierarchy.
    
    For now with flat topics, just does direct lookup.
    Future: "history/ancient/rome" checks rome -> ancient -> history -> default.
    
    Args:
        topic_weights: User's topic weights dict
        topic_path: Topic path (e.g., "history" or future "history/ancient/rome")
        
    Returns:
        Effective weight (default 1.0 if not found)
    """
    # Direct lookup for flat topics
    if topic_path in topic_weights:
        return topic_weights[topic_path]
    
    # Future: walk up path hierarchy with decay
    # parts = topic_path.split("/")
    # decay = 0.7
    # for i in range(len(parts) - 1, -1, -1):
    #     parent = "/".join(parts[:i+1])
    #     if parent in topic_weights:
    #         levels_up = len(parts) - i - 1
    #         return topic_weights[parent] * (decay ** levels_up)
    
    return 1.0
