from __future__ import annotations

import asyncio
import json
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
    except Exception as e:
        logger.warning(f"Failed to load models.json: {e}, using empty config")
        return {
            "models": [],
            "default_model": None,
            "fallback_chain": [],
            "provider_configs": {},
        }


async def _retry_with_backoff(
    func: Callable, *args, max_retries: int = 3, **kwargs
) -> Any:
    """Execute a function with exponential backoff for rate limits and transient errors."""
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.TimeoutException as e:
            last_exception = e
            if attempt >= max_retries:
                logger.error(f"LLM request timed out after {max_retries} attempts: {e}")
                raise RuntimeError(
                    f"LLM request timed out after {max_retries} attempts"
                ) from e

            delay = 2**attempt
            jitter = random.uniform(0, 0.5 * delay)
            sleep_time = delay + jitter

            logger.warning(
                f"LLM request timed out (attempt {attempt + 1}/{max_retries + 1}). Retrying in {sleep_time:.2f}s..."
            )
            await asyncio.sleep(sleep_time)

        except httpx.NetworkError as e:
            last_exception = e
            if attempt >= max_retries:
                logger.error(
                    f"LLM request failed due to network error after {max_retries} attempts: {e}"
                )
                raise RuntimeError(
                    f"Network error connecting to LLM after {max_retries} attempts"
                ) from e

            delay = 2**attempt
            jitter = random.uniform(0, 0.5 * delay)
            sleep_time = delay + jitter

            logger.warning(
                f"LLM request network error (attempt {attempt + 1}/{max_retries + 1}). Retrying in {sleep_time:.2f}s..."
            )
            await asyncio.sleep(sleep_time)

        except httpx.HTTPStatusError as e:
            last_exception = e

            should_retry = False
            status_code = e.response.status_code
            retry_after = e.response.headers.get("Retry-After")

            if status_code and (status_code == 429 or 500 <= status_code < 600):
                should_retry = True

            if not should_retry or attempt >= max_retries:
                logger.error(
                    f"LLM request failed with HTTP {status_code} after {max_retries} attempts: {e}"
                )
                raise e

            delay = 2**attempt
            if retry_after:
                try:
                    delay = float(retry_after)
                except (ValueError, TypeError):
                    pass

            jitter = random.uniform(0, 0.5 * delay)
            sleep_time = delay + jitter

            logger.warning(
                f"LLM request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {sleep_time:.2f}s..."
            )
            await asyncio.sleep(sleep_time)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Retry function completed without success or exception")


def _pick_openrouter_model(requested: Optional[str] = None) -> str:
    """Prefer non-thinking model variants by default."""
    if requested:
        return requested
    m = os.getenv("OPENROUTER_MODEL_NONREASONING")
    if m:
        return m
    m2 = os.getenv("OPENROUTER_MODEL")
    if m2:
        return m2

    config = get_llm_config()
    default_id = config.get("default_model")

    if default_id:
        models = config.get("models", [])
        for model in models:
            if model.get("id") == default_id:
                return model.get("model", default_id)

    models = config.get("models", [])
    if models:
        return models[0].get("model", "unknown")

    logger.warning("No models configured in JSON config, using fallback")
    return "x-ai/grok-4.1-fast:free"


def _get_fallback_models() -> List[str]:
    """Get fallback model IDs from JSON config."""
    config = get_llm_config()
    fallback_chain = config.get("fallback_chain", [])

    if not fallback_chain:
        logger.debug("No fallback chain configured")
        return []

    models = config.get("models", [])
    model_map = {m["id"]: m["model"] for m in models}
    fallback_models = [model_map.get(mid, mid) for mid in fallback_chain if mid]

    logger.debug(f"Loaded {len(fallback_models)} fallback models from config")
    return fallback_models


def _get_provider_config(provider: str) -> Dict[str, Any]:
    """Get provider-specific configuration from JSON config."""
    config = get_llm_config()
    provider_configs = config.get("provider_configs", {})
    return provider_configs.get(provider, {})


async def _call_openrouter_with_model(
    messages: List[Dict[str, str]],
    model_id: str,
    temperature: Optional[float],
    max_tokens: int,
    api_key: str,
):
    """Make OpenRouter API call using httpx with specific model."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    data = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature or float(os.getenv("ARC_LLM_TEMPERATURE", "0.7")),
        "max_tokens": max_tokens,
    }

    provider_cfg = _get_provider_config("openrouter")
    app_title = os.getenv(
        "OPENROUTER_APP_TITLE", provider_cfg.get("app_title", "Arcadia AI Chat")
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_title,
    }

    referer = os.getenv("OPENROUTER_REFERER", provider_cfg.get("referer"))
    if referer:
        headers["HTTP-Referer"] = referer

    timeout = provider_cfg.get("timeout", 60)
    max_retries = provider_cfg.get("max_retries", 3)

    logger.info(
        f"Calling OpenRouter API with model: {model_id}, messages count: {len(messages)}"
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Request data: {json.dumps(data, ensure_ascii=False)[:500]}")
        logger.debug(
            f"Request messages: {json.dumps(messages, ensure_ascii=False)[:500]}"
        )

    async def _execute_call():
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            resp = await client.post(url, json=data, headers=headers)
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

    return await _retry_with_backoff(_execute_call, max_retries=max_retries)


async def _call_with_model_fallback(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: int,
    api_key: str,
):
    """Try primary model, fallback to alternatives on 404 errors."""
    models_to_try = [model] if model else [_pick_openrouter_model()]
    models_to_try.extend(_get_fallback_models())

    last_error = None

    for try_model in models_to_try:
        if try_model in models_to_try[: models_to_try.index(try_model)]:
            continue  # Skip duplicates

        try:
            return await _call_openrouter_with_model(
                messages, try_model, temperature, max_tokens, api_key
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Model {try_model} not found (404), trying next model")
                last_error = e
                continue
            else:
                raise e

    if last_error:
        raise last_error
    raise RuntimeError("No fallback models available")


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
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return _call_openrouter(messages, model, temperature, max_tokens, api_key)
    else:
        return _call_local_api(
            messages, model, base_url, temperature, max_tokens, api_key or ""
        )


async def _call_openrouter(
    messages: List[Dict[str, str]],
    model: Optional[str],
    temperature: Optional[float],
    max_tokens: int,
    api_key: str,
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Make OpenRouter API call using httpx with model fallback."""
    return await _call_with_model_fallback(
        messages, model, temperature, max_tokens, api_key
    )


async def _call_local_api(
    messages: List[Dict[str, str]],
    model: Optional[str],
    base_url: str,
    temperature: Optional[float],
    max_tokens: int,
    api_key: Optional[str],
):
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

    provider_cfg = _get_provider_config("local")
    timeout = provider_cfg.get("timeout", 60)
    max_retries = provider_cfg.get("max_retries", 3)

    async def _execute_call():
        async with httpx.AsyncClient(timeout=float(timeout)) as client:
            resp = await client.post(url, json=data, headers=headers)
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

    return await _retry_with_backoff(_execute_call, max_retries=max_retries)
