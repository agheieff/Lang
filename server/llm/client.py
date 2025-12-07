from __future__ import annotations

import json
import time
import random
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import httpx
from server.utils.nlp import strip_thinking_blocks

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    id: str
    display_name: str
    model: str
    base_url: str
    api_key_env: Optional[str] = None
    max_tokens: Optional[int] = None
    allowed_tiers: Optional[List[str]] = None


def get_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from models.json file."""
    models_file = Path(__file__).parent / "models.json"
    
    try:
        with open(models_file, "r") as f:
            return json.load(f)
    except Exception:
        return {
            "models": [
                {
                    "id": "grok-fast-free",
                    "display_name": "Grok 4.1 Fast (Free)",
                    "model": "x-ai/grok-4.1-fast:free",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_key_env": "OPENROUTER_API_KEY",
                    "max_tokens": 32768,
                    "allowed_tiers": ["Free", "Standard", "Pro", "Pro+", "BYOK", "admin"],
                }
            ],
            "default_model": "grok-fast-free",
        }


def _retry_with_backoff(func: Callable, *args, max_retries: int = 3, **kwargs) -> Any:
    """Execute a function with exponential backoff for rate limits and transient errors."""
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            last_exception = e
            
            should_retry = False
            status_code = e.response.status_code
            retry_after = e.response.headers.get("Retry-After")
            
            if status_code and (status_code == 429 or 500 <= status_code < 600):
                should_retry = True
                
            if not should_retry or attempt >= max_retries:
                raise e
            
            delay = 2 ** attempt
            if retry_after:
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    pass
            
            jitter = random.uniform(0, 0.5 * delay)
            sleep_time = delay + jitter
            
            logger.warning(f"LLM request failed (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            
    raise last_exception


def _pick_openrouter_model(requested: Optional[str]) -> str:
    """Prefer non-thinking model variants by default."""
    if requested:
        return requested
    m = os.getenv("OPENROUTER_MODEL_NONREASONING")
    if m:
        return m
    m2 = os.getenv("OPENROUTER_MODEL")
    return m2 or "x-ai/grok-4.1-fast:free"


def chat_complete(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 16384,
    user_api_key: Optional[str] = None,
) -> str:
    """Call LLM API and return text response."""
    text, _ = chat_complete_with_raw(
        messages,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        user_api_key=user_api_key,
    )
    return text


def chat_complete_with_raw(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: int = 16384,
    user_api_key: Optional[str] = None,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Call LLM API and return (cleaned_text, provider_response_dict_or_none)."""
    # Resolve configuration
    api_key = user_api_key or os.getenv("OPENROUTER_API_KEY")
    
    if not api_key and (not base_url or "openrouter" in (base_url or "")):
        raise RuntimeError("OPENROUTER_API_KEY not set")
    
    # Use OpenRouter if no base_url specified or if explicitly set to OpenRouter
    if not base_url or "openrouter.ai" in base_url:
        return _call_openrouter(messages, model, temperature, max_tokens, api_key)
    else:
        return _call_local_api(messages, model, base_url, temperature, max_tokens, api_key)


def _call_openrouter(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: int,
    api_key: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Make OpenRouter API call using httpx."""
    model_id = _pick_openrouter_model(model)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature or float(os.getenv("ARC_LLM_TEMPERATURE", "0.7")),
        "max_tokens": max_tokens,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Arcadia AI Chat"),
    }
    
    if ref := os.getenv("OPENROUTER_REFERER"):
        headers["HTTP-Referer"] = ref
    
    def _execute_call():
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=data, headers=headers)
            resp.raise_for_status()
            resp_dict = resp.json()
        
        choices = resp_dict.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("no choices in response")
        
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise RuntimeError("no message in choice")
        
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError("no content in message")
        
        return strip_thinking_blocks(content), resp_dict
    
    return _retry_with_backoff(_execute_call, max_retries=3)


def _call_local_api(
    messages: List[Dict[str, str]],
    model: Optional[str],
    base_url: str,
    temperature: Optional[float],
    max_tokens: int,
    api_key: Optional[str],
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Make local OpenAI-compatible API call using httpx."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    data = {
        "model": model or "local",
        "messages": messages,
        "temperature": temperature or float(os.getenv("ARC_LLM_TEMPERATURE", "0.7")),
        "max_tokens": max_tokens,
    }
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    def _execute_call():
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=data, headers=headers)
            resp.raise_for_status()
            resp_dict = resp.json()
        
        choices = resp_dict.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError("no choices in response")
        
        msg = choices[0].get("message")
        if not isinstance(msg, dict):
            raise RuntimeError("no message in choice")
        
        content = msg.get("content")
        if not isinstance(content, str):
            raise RuntimeError("no content in message")
        
        return strip_thinking_blocks(content), resp_dict
    
    return _retry_with_backoff(_execute_call, max_retries=3)
