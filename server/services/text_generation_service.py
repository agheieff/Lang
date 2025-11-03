"""
Pure text generation service.
Handles only the generation of text content, separating it from translation concerns.
"""

import datetime
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import threading
from sqlalchemy.orm import Session

from ..llm import build_reading_prompt
from ..utils.json_parser import extract_text_from_llm_response, extract_json_from_text
from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from .llm_logging import log_llm_request, llm_call_and_log_to_file


class TextGenerationResult:
    """Result of text generation attempt."""
    def __init__(self, 
                 success: bool,
                 text: str,
                 title: Optional[str] = None,
                 provider: Optional[str] = None,
                 model: Optional[str] = None,
                 response_dict: Optional[Dict] = None,
                 error: Optional[str] = None,
                 log_dir: Optional[Path] = None):
        self.success = success
        self.text = text
        self.title = title
        self.provider = provider
        self.model = model
        self.response_dict = response_dict or {}
        self.error = error
        self.log_dir = log_dir


class TextGenerationService:
    """
    Pure text generation service.
    
    This service handles:
    - Creating placeholder ReadingText records
    - Generating text content via LLM
    - Persisting the content
    - Logging generation requests
    
    It does NOT handle:
    - Translation generation (handled by TranslationService)
    - User state management (handled by StateManager)
    - Notification sending (handled by NotificationService)
    """
    
    def __init__(self):
        self._running: set[Tuple[int, str]] = set()  # (account_id,_lang) pairs
        self._running_lock = threading.Lock()
    
    def create_placeholder_text(self, account_db: Session, account_id: int, lang: str) -> Optional[int]:
        """
        Create a placeholder ReadingText record.
        
        Returns the text_id if created, None if error.
        """
        from ..models import ReadingText
        
        rt = ReadingText(
            account_id=account_id,
            lang=lang,
            content=None,
            request_sent_at=datetime.datetime.utcnow(),
        )
        
        try:
            account_db.add(rt)
            account_db.flush()
            print(f"[TEXT_GEN] Created placeholder for text_id={rt.id} account_id={account_id} lang={lang}")
            return rt.id
        except Exception as e:
            try:
                account_db.rollback()
            except Exception:
                pass
            # Try with empty string if content can't be NULL
            try:
                rt = ReadingText(
                    account_id=account_id,
                    lang=lang,
                    content="",
                    request_sent_at=datetime.datetime.utcnow(),
                )
                account_db.add(rt)
                account_db.flush()
                print(f"[TEXT_GEN] Created placeholder (fallback) for text_id={rt.id} account_id={account_id} lang={lang}")
                return rt.id
            except Exception:
                print(f"[TEXT_GEN] Failed to create placeholder: {e}")
                return None
    
    def generate_text_content(self,
                             account_db: Session,
                             global_db: Session,
                             account_id: int,
                             lang: str,
                             text_id: int,
                             job_dir: Path,
                             messages) -> TextGenerationResult:
        """
        Generate text content for a placeholder text.
        
        Args:
            account_db: Per-account database session
            global_db: Global database session
            account_id: User account ID
            lang: Language code
            text_id: Text record ID
            job_dir: Directory for logging
            messages: LLM prompt messages
            
        Returns:
            TextGenerationResult with the generated content or error
        """
        from ..models import ReadingText
        from ..services.model_registry_service import get_model_registry
        from ..auth import Account
        
        # Get user's subscription tier for model selection from global database
        account = global_db.query(Account).filter(Account.id == account_id).first()
        user_tier = account.subscription_tier if account else "Free"
        
        # Get available models for user's tier
        registry = get_model_registry()
        available_models = registry.get_available_models(user_tier)
        
        if not available_models:
            raise ValueError(f"No models available for tier {user_tier}")
        
        best_result = TextGenerationResult(success=False, text="", error="No providers succeeded")
        
        # Try models in order of preference (fallback chain or first available)
        models_to_try = []
        if registry.config.fallback_chain:
            # Use fallback chain order, filter by available models
            for model_id in registry.config.fallback_chain:
                if any(m.id == model_id for m in available_models):
                    models_to_try.append(registry.get_model_by_id(model_id))
            # Add remaining available models not in fallback chain
            existing_ids = {m.id for m in models_to_try}
            for model in available_models:
                if model.id not in existing_ids:
                    models_to_try.append(model)
        else:
            # No fallback chain, use available models as-is
            models_to_try = available_models
        
        for model_config in models_to_try:
            # Determine number of attempts based on provider
            provider = "openrouter" if "openrouter" in model_config.base_url else "local"
            attempts = int(os.getenv("ARC_OR_READING_ATTEMPTS", "3")) if provider == "openrouter" else 1
            
            for attempt in range(attempts):
                try:
                    print(f"[TEXT_GEN] Attempt {attempt + 1}/{attempts} for text_id={text_id} with model={model_config.display_name}")
                    
                    # Call LLM using new configuration
                    text_buf, resp_dict, used_provider, used_model = self._complete_and_log(
                        account_db,
                        account_id,
                        text_id,
                        messages,
                        model_config=model_config,
                        out_path=job_dir / "text.json"
                    )
                    
                    if text_buf and text_buf.strip():
                        # Success! Parse and store the result
                        final_text_raw = extract_text_from_llm_response(text_buf) or text_buf
                        final_text = str(final_text_raw).replace("\r\n", "\n").replace("\r", "\n")
                        
                        # Extract optional title
                        try:
                            title_val = extract_json_from_text(text_buf, "title")
                            title = str(title_val) if title_val is not None else None
                        except Exception:
                            title = None
                        
                        if not final_text or not final_text.strip():
                            raise ValueError("Empty or invalid text content")
                        
                        # Update the database record
                        rt = account_db.query(ReadingText).filter(ReadingText.id == text_id).first()
                        if rt:
                            rt.content = final_text
                            rt.generated_at = datetime.datetime.utcnow()
                            account_db.commit()
                            
                            result = TextGenerationResult(
                                success=True,
                                text=final_text,
                                title=title,
                                provider=used_provider,
                                model=used_model,
                                response_dict=resp_dict,
                                log_dir=job_dir
                            )
                            
                            print(f"[TEXT_GEN] Successfully generated text_id={text_id} with provider={used_provider}")
                            return result
                        
                except Exception as e:
                    # Print the actual error for debugging
                    error_msg = str(e)
                    error_type = type(e).__name__
                    print(f"[TEXT_GEN] ❌ ERROR with {provider} (attempt {attempt + 1}/{attempts}): {error_type}: {error_msg}")
                    
                    # Additional details for HTTP errors
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        print(f"[TEXT_GEN] ❌ HTTP Status: {e.response.status_code}")
                        if hasattr(e.response, 'text'):
                            resp_text = e.response.text
                            print(f"[TEXT_GEN] ❌ HTTP Response: {resp_text[:200]}...")
                    
                    # Log the error attempt
                    try:
                        log_llm_request(
                            account_db,
                            account_id=account_id,
                            text_id=text_id,
                            kind="reading",
                            provider=provider,
                            model=getattr(model_config, 'model', None),
                            base_url=getattr(model_config, 'base_url', None),
                            request={"messages": messages},
                            response=None,
                            status="error",
                            error=error_msg,
                        )
                    except Exception:
                        pass
                    
                    # Backoff for retriable errors
                    try:
                        status = None
                        retry_after = None
                        try:
                            status = getattr(e, "response", None).status_code  # type: ignore[attr-defined]
                            retry_after = getattr(e, "response", None).headers.get("Retry-After")  # type: ignore[attr-defined]
                        except Exception:
                            status = None
                        
                        retriable = provider == "openrouter" and (status in {429, 500, 502, 503, 504})
                        if not retriable or attempt >= attempts - 1:
                            break
                        
                        try:
                            delay_base = float(retry_after) if retry_after and str(retry_after).isdigit() else float(2 ** (attempt + 1))
                        except Exception:
                            delay_base = float(2 ** (attempt + 1))
                        jitter = random.uniform(0, delay_base * 0.5)
                        time.sleep(delay_base + jitter)
                    except Exception:
                        break
                    
                    best_result = TextGenerationResult(
                        success=False,
                        text="",
                        error=f"Provider {provider} failed: {str(e)}"
                    )
        
        # All providers failed
        print(f"[TEXT_GEN] ❌ ALL PROVIDERS FAILED for text_id={text_id}")
        print(f"[TEXT_GEN] ❌ Final error message: {best_result.error}")
        print(f"[TEXT_GEN] ❌ Tried {[m.display_name for m in models_to_try]} models")
        
        # Clean up the placeholder record
        try:
            rt = account_db.query(ReadingText).filter(ReadingText.id == text_id).first()
            if rt:
                account_db.delete(rt)
                account_db.commit()
                print(f"[TEXT_GEN] ❌ Cleaned up placeholder text_id={text_id} due to failure")
        except Exception as e:
            try:
                account_db.rollback()
                print(f"[TEXT_GEN] ❌ Failed to clean up placeholder: {e}")
            except Exception:
                pass
        
        return best_result
    
    def _complete_and_log(self,
                          account_db: Session,
                          account_id: int,
                          text_id: int,
                          messages: list,
                          model_config=None,
                          provider: str = None,
                          model: Optional[str] = None,
                          base_url: Optional[str] = None,
                          out_path: Path = None) -> tuple[str, Dict, str, Optional[str]]:
        """
        Call LLM and write request+response to a JSON file.
        
        Args:
            account_db: Database session for logging
            account_id: User account ID
            text_id: Text ID for logging
            messages: List of message dicts
            model_config: ModelConfig object (preferred)
            provider: Provider name (legacy)
            model: Model name (legacy)
            base_url: Base URL (legacy)
            out_path: Path to write log file
            
        Returns: (text_content, response_dict, provider_used, model_used)
        """
        if out_path is None:
            # Fallback if no path provided
            out_path = Path(f"/tmp/llm_request_{text_id}_{datetime.datetime.now().isoformat()}.json")
            
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Use shared helper for call + filesystem logging
        text, resp, provider_used, model_used = llm_call_and_log_to_file(
            messages,
            provider,
            model,
            base_url,
            out_path,
            model_config=model_config,
        )
        resp_dict: Dict = resp or {}
        base_url_used = getattr(model_config, 'base_url', base_url)
        
        # Log to database
        try:
            log_llm_request(
                account_db,
                account_id=account_id,
                text_id=text_id,
                kind="reading",
                provider=provider_used,
                model=model_used,
                base_url=base_url_used,
                request={"messages": messages},
                response=resp_dict,
                status="ok",
            )
        except Exception:
            pass

        return text, resp_dict, provider_used, model_used
    
    def is_generation_in_progress(self, account_id: int, lang: str) -> bool:
        """Check if text generation is already in progress for this user/language."""
        key = (int(account_id), str(lang))
        with self._running_lock:
            return key in self._running
    
    def mark_generation_started(self, account_id: int, lang: str) -> bool:
        """Mark that generation has started. Returns False if already in progress."""
        key = (int(account_id), str(lang))
        with self._running_lock:
            if key in self._running:
                return False
            self._running.add(key)
        return True
    
    def mark_generation_completed(self, account_id: int, lang: str) -> None:
        """Mark that generation has completed."""
        key = (int(account_id), str(lang))
        with self._running_lock:
            self._running.discard(key)
