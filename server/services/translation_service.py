"""
Translation generation service.
Handles generation of word translations and sentence translations.
Extracted from gen_queue.py to provide a clean, focused service.
"""

import json
import os
import random
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from ..services.model_registry_service import get_model_registry
from ..auth import Account
from ..llm.prompts import build_translation_contexts, build_word_translation_prompt
from ..utils.text_segmentation import split_sentences
from .llm_logging import log_llm_request, llm_call_and_log_to_file
from .openrouter_key_service import get_openrouter_key_service
from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from ..utils.json_parser import (
    extract_structured_translation,
    extract_word_translations,
    extract_json_from_text,
)
from ..utils.gloss import compute_spans


class TranslationResult:
    """Result of translation generation attempt."""
    def __init__(self, 
                 success: bool,
                 words: bool = False,
                 sentences: bool = False,
                 error: Optional[str] = None,
                 log_dir: Optional[Path] = None):
        self.success = success
        self.words = words  # Whether words were successfully generated
        self.sentences = sentences  # Whether sentences were successfully generated
        self.error = error
        self.log_dir = log_dir


class TranslationService:
    """
    Handles translation generation for texts.
    
    This service manages:
    - Generation of word translations (per-sentence)
    - Generation of sentence translations (structured)
    - Persistence to database
    - Logging of translation requests
    
    It does NOT handle:
    - Text content generation (handled by TextGenerationService)
    - User state management (handled by StateManager)
    - Notifications (handled by NotificationService)
    """
    
    def __init__(self):
        self.max_workers = 10  # Default parallelism for word translations
    
    def generate_translations(self,
                                   account_db: Session,
                                   global_db: Session,
                                   account_id: int,
                                   lang: str,
                                   text_id: int,
                                   text_content: str,
                                   text_title: Optional[str],
                                   job_dir: Path,
                                   reading_messages: List[Dict],
                                   provider: Optional[str] = None,  # Will be determined by model registry
                                   model_id: Optional[str] = None,  # Will be determined by model registry
                                   base_url: Optional[str] = None) -> TranslationResult:
        """
        Generate all translations for a text.
        
        Args:
            account_db: Per-account database session
            global_db: Global database session
            account_id: User account ID
            lang: Source language code
            text_id: Text record ID
            text_content: The raw text content
            text_title: Optional title from the text
            job_dir: Directory for logging
            reading_messages: Original LLM messages used for text generation
            provider: LLM provider to use
            model_id: Model identifier (if applicable)
            base_url: Base URL for LLM API
            
        Returns:
            TranslationResult indicating success/failure of each component
        """
        # Get user's subscription tier for model selection from global database
        account = global_db.query(Account).filter(Account.id == account_id).first()
        user_tier = account.subscription_tier if account else "Free"
        
        # Get user's per-account OpenRouter key if they have one (paid tier)
        user_api_key = None
        if account:
            key_service = get_openrouter_key_service()
            user_api_key = key_service.get_user_key(account)
        
        # Resolve models separately for word and sentence translation
        from .model_resolution import resolve_models_for_task
        word_models = resolve_models_for_task(
            account_db, global_db, account_id, lang, "preferred_word_translation_model"
        )
        sentence_models = resolve_models_for_task(
            account_db, global_db, account_id, lang, "preferred_sentence_translation_model"
        )

        # Fallback to legacy arguments if no resolved models
        def _make_fallback_model():
            from ..llm_config.llm_models import ModelConfig
            return ModelConfig(
                id="legacy-temp",
                display_name=f"Legacy {provider}/{model_id}",
                model=model_id or "x-ai/grok-4.1-fast:free",
                base_url=base_url or "https://openrouter.ai/api/v1",
                api_key_env=None,
                max_tokens=16384,
                allowed_tiers=[],
                capabilities=["text"]
            )
        
        if not word_models and (provider or model_id):
            word_models = [_make_fallback_model()]
        if not sentence_models and (provider or model_id):
            sentence_models = [_make_fallback_model()]
        
        # Try to get default system model if still empty
        if not word_models or not sentence_models:
            try:
                registry = get_model_registry()
                default_model = registry.get_default_model(user_tier)
                if not word_models:
                    word_models = [default_model]
                if not sentence_models:
                    sentence_models = [default_model]
            except Exception:
                pass

        if not word_models and not sentence_models:
            raise ValueError(f"No translation models available for tier {user_tier}")

        # Get user profile for target language
        prof = account_db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == lang).first()
        target_lang = (prof.target_lang if prof and getattr(prof, "target_lang", None) else "en")
        
        # Rebuild translation contexts
        ctx = build_translation_contexts(
            reading_messages,
            source_lang=lang,
                target_lang=target_lang,
            text=text_content,
        )
        
        tr_messages = ctx["structured"]
        w_messages = ctx["words"]
        
        # Use the full first response as context
        if isinstance(tr_messages, list) and len(tr_messages) > 2:
            tr_messages[2]["content"] = text_content
        if isinstance(w_messages, list) and len(w_messages) > 2:
            w_messages[2]["content"] = text_content
        
        best_result = TranslationResult(success=False)
        
        # Generate word translations using word_models
        for model_config in (word_models or []):
            if best_result.words:
                break
            cur_provider = "openrouter" if "openrouter" in model_config.base_url else "local"
            cur_model = model_config.model
            cur_base_url = model_config.base_url
            
            words_success = self._generate_word_translations(
                account_db, account_id, text_id, lang, target_lang,
                text_content, job_dir, cur_provider, cur_model, cur_base_url,
                w_messages, reading_messages, text_content, user_tier,
                model_config=model_config
            )
            if words_success:
                best_result.words = True
                # Set words_complete flag on the text
                try:
                    rt = account_db.get(ReadingText, text_id)
                    if rt:
                        rt.words_complete = True
                        account_db.flush()
                except Exception as e:
                    logger.warning(f"[TRANSLATION] Failed to set words_complete flag: {e}")

        # Generate sentence translations using sentence_models
        for model_config in (sentence_models or []):
            if best_result.sentences:
                break
            cur_provider = "openrouter" if "openrouter" in model_config.base_url else "local"
            cur_model = model_config.model
            cur_base_url = model_config.base_url
            
            sentences_success = self._generate_sentence_translations(
                account_db, account_id, text_id, lang, target_lang,
                text_content, job_dir, cur_provider, cur_model, cur_base_url,
                tr_messages,
                model_config=model_config
            )
            if sentences_success:
                best_result.sentences = True
                # Set sentences_complete flag on the text
                try:
                    rt = account_db.get(ReadingText, text_id)
                    if rt:
                        rt.sentences_complete = True
                        account_db.flush()
                except Exception as e:
                    logger.warning(f"[TRANSLATION] Failed to set sentences_complete flag: {e}")
        
        # Generate title translation using sentence model (or first available)
        if text_title:
            title_model = (sentence_models or word_models or [None])[0]
            if title_model:
                cur_provider = "openrouter" if "openrouter" in title_model.base_url else "local"
                self._generate_title_translation(
                    account_db, account_id, text_id, lang, target_lang,
                    text_title, job_dir, cur_provider, title_model.model, title_model.base_url,
                    model_config=title_model
                )
        
        if best_result.words and best_result.sentences:
            best_result.success = True
        
        # Final check (title is optional)
        if best_result.words or best_result.sentences:
             best_result.success = True
             
        return best_result
    
    def _generate_word_translations(self,
                                    account_db: Session,
                                    account_id: int,
                                    text_id: int,
                                    lang: str,
                                    target_lang: str,
                                    text_content: str,
                                    job_dir: Path,
                                    provider: str,
                                    model_id: Optional[str],
                                    base_url: Optional[str],
                                    w_messages: List[Dict],
                                    reading_messages: List[Dict],
                                    full_response: str,
                                    user_tier: str = "Free",
                                    user_api_key: Optional[str] = None,
                                    model_config=None) -> bool:
        """Generate word translations split by sentence."""
        try:
            # Split text into sentences
            sent_spans = split_sentences(text_content, lang)
            
            # Build per-sentence messages preserving reading context
            per_msgs: List[Tuple[int, str, List[Dict]]] = []
            for (s, e, seg) in sent_spans:
                msgs = build_word_translation_prompt(lang, target_lang, seg)
                # Include original reading context if available
                if reading_messages and len(reading_messages) >= 2:
                    words_ctx = [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": reading_messages[1]["content"]},
                        {"role": "assistant", "content": full_response},
                        {"role": "user", "content": msgs[1]["content"]},
                    ]
                else:
                    # Fallback without reading context
                    words_ctx = [
                        {"role": "system", "content": msgs[0]["content"]},
                        {"role": "user", "content": msgs[1]["content"]},
                    ]
                per_msgs.append((s, seg, words_ctx))
            
            # Process sentences in parallel
            wd_seg_results: List[Tuple[int, str, Optional[tuple], List[Dict]]] = []
            
            # Adjust concurrency based on tier
            # Free tier gets minimal concurrency to avoid rate limits
            if user_tier == "Free" or provider == "openrouter":
                concurrency = 2
            else:
                concurrency = self.max_workers
                
            logger.info(f"[TRANSLATION] Starting word translation with concurrency={concurrency}")
            
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futures = []
                for i, (s, seg, m) in enumerate(per_msgs):
                    # Shift by +1 so words_1 is the first sentence
                    out_path = job_dir / f"words_{i+1}.json"
                    futures.append((s, seg, m, ex.submit(
                        llm_call_and_log_to_file,
                        m, provider, model_id, base_url, out_path,
                        user_api_key=user_api_key,
                        model_config=model_config
                    )))
                
                for s, seg, m, f in futures:
                    try:
                        wd_seg_results.append((s, seg, f.result(), m))
                    except Exception:
                        wd_seg_results.append((s, seg, None, m))
            
            # Retry failed or empty responses (up to 2 retries per sentence)
            max_retries = 2
            for retry_round in range(max_retries):
                failed_indices = []
                for i, (s, seg, tup, msgs_used) in enumerate(wd_seg_results):
                    # Check if response is empty or None
                    if tup is None or (isinstance(tup, tuple) and (not tup[0] or len(tup[0].strip()) == 0)):
                        failed_indices.append(i)
                
                if not failed_indices:
                    break
                    
                logger.info(f"[TRANSLATION] Retry round {retry_round + 1}: {len(failed_indices)} sentences need retry")
                
                # Retry failed sentences sequentially (more reliable for retries)
                for idx in failed_indices:
                    s, seg, _, msgs_used = wd_seg_results[idx]
                    out_path = job_dir / f"words_{idx+1}_retry{retry_round+1}.json"
                    try:
                        result = llm_call_and_log_to_file(
                            msgs_used, provider, model_id, base_url, out_path,
                            user_api_key=user_api_key,
                            model_config=model_config
                        )
                        if result and result[0] and len(result[0].strip()) > 0:
                            wd_seg_results[idx] = (s, seg, result, msgs_used)
                            logger.info(f"[TRANSLATION] Retry succeeded for sentence {idx + 1}")
                    except Exception as e:
                        logger.warning(f"[TRANSLATION] Retry failed for sentence {idx + 1}: {e}")
        
            # Track existing and new word spans to avoid duplicates
            existing = self._get_existing_spans(account_db, account_id, text_id)
            seen = set()
            
            # Persist word translations
            for s, seg, tup, msgs_used in wd_seg_results:
                # Log to DB per-call (main thread) before persisting
                try:
                    if tup:
                        log_llm_request(
                            account_db,
                            account_id=account_id,
                            text_id=text_id,
                            kind="word_translation",
                            provider=tup[2],
                            model=tup[3],
                            base_url=base_url,
                            request={"messages": msgs_used},
                            response=tup[1],
                            status="ok",
                        )
                except Exception:
                    pass
                self._persist_word_translations(
                    account_db, account_id, text_id, lang, s, seg, tup, msgs_used, existing, seen
                )
        
            # Flush the session to ensure word translations are written
            try:
                account_db.flush()
            except Exception as e:
                logger.error(f"[TRANSLATION] Failed to flush word translations: {e}", exc_info=True)
                return False
        
            return True
            
        except Exception as e:
            logger.error(f"[TRANSLATION] Word translation failed: {e}", exc_info=True)
            return False
    
    def _generate_sentence_translations(self,
                                      account_db: Session,
                                      account_id: int,
                                      text_id: int,
                                      lang: str,
                                      target_lang: str,
                                      text_content: str,
                                      job_dir: Path,
                                      provider: str,
                                      model_id: Optional[str],
                                      base_url: Optional[str],
                                      tr_messages: List[Dict],
                                      user_api_key: Optional[str] = None,
                                      model_config=None) -> bool:
        """Generate structured sentence translations."""
        try:
            # Debug logging
            logger.info(f"[TRANSLATION] Sentence translation starting: provider={provider}, model_id={model_id}, base_url={base_url}")
            if model_config:
                logger.info(f"[TRANSLATION] model_config: model={getattr(model_config, 'model', None)}, base_url={getattr(model_config, 'base_url', None)}")
            
            # Call LLM for structured translation with retry
            tr_buf, tr_resp, used_provider, used_model = None, None, None, None
            max_retries = 2
            
            for attempt in range(max_retries + 1):
                try:
                    out_path = job_dir / f"structured{'_retry' + str(attempt) if attempt > 0 else ''}.json"
                    tr_buf, tr_resp, used_provider, used_model = llm_call_and_log_to_file(
                        tr_messages, provider, model_id, base_url, out_path,
                        user_api_key=user_api_key,
                        model_config=model_config
                    )
                    if tr_buf and len(tr_buf.strip()) > 0:
                        break
                    logger.warning(f"[TRANSLATION] Empty response on attempt {attempt + 1}, retrying...")
                except Exception as e:
                    logger.warning(f"[TRANSLATION] Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries:
                        raise
            
            if not tr_buf:
                logger.error("[TRANSLATION] No response buffer from LLM for structured translation after retries")
                return False
            
            # Log to DB (main thread) for structured translation
            try:
                log_llm_request(
                    account_db,
                    account_id=account_id,
                    text_id=text_id,
                    kind="structured_translation",
                    provider=used_provider,
                    model=used_model,
                    base_url=base_url,
                    request={"messages": tr_messages},
                    response=tr_resp,
                    status="ok",
                )
            except Exception:
                pass

            # Parse and save sentence translations
            tr_parsed = extract_structured_translation(tr_buf)
            if not tr_parsed:
                logger.error(f"[TRANSLATION] Failed to parse structured translation: {tr_buf[:100]}...")
                return False
            # Compute sentence spans based on current text
            sent_spans = self._split_sentences(text_content, lang)
            idx = 0
            for p in tr_parsed.get("paragraphs", []):
                for s in p.get("sentences", []):
                    if "text" in s and "translation" in s:
                        # Map by order; if we have spans, attach them
                        span_start = None
                        span_end = None
                        actual_source_text = s["text"]  # Fallback to LLM response
                        try:
                            if 0 <= idx < len(sent_spans):
                                span_start, span_end, actual_source_text = sent_spans[idx]
                        except Exception:
                            span_start, span_end = None, None
                        account_db.add(ReadingTextTranslation(
                            account_id=account_id,
                            text_id=text_id,
                            unit="sentence",
                            target_lang=target_lang,
                            segment_index=idx,
                            span_start=span_start,
                            span_end=span_end,
                            source_text=actual_source_text,
                            translated_text=s["translation"],
                            provider=provider,
                            model=model_id,
                        ))
                        idx += 1
            
            # Synthesize and persist full text translation
            try:
                full_translation_parts = []
                # Re-iterate parsed structure or use the DB objects if flushed?
                # Using the parsed structure is safer as DB objects might be detached or pending
                for p in tr_parsed.get("paragraphs", []):
                    p_sents = []
                    for s in p.get("sentences", []):
                        if "translation" in s:
                            p_sents.append(s["translation"])
                    if p_sents:
                        full_translation_parts.append(" ".join(p_sents))
                
                if full_translation_parts:
                    full_translation = "\n\n".join(full_translation_parts)
                    
                    # Check if already exists (idempotency)
                    exists = account_db.query(ReadingTextTranslation).filter(
                        ReadingTextTranslation.account_id == account_id,
                        ReadingTextTranslation.text_id == text_id,
                        ReadingTextTranslation.unit == "text",
                        # Exclude title translation which usually has segment_index=0 and different source
                        ReadingTextTranslation.segment_index == 1 
                    ).first()
                    
                    if not exists:
                        account_db.add(ReadingTextTranslation(
                            account_id=account_id,
                            text_id=text_id,
                            unit="text",
                            target_lang=target_lang,
                            segment_index=1, # 0 is usually title
                            span_start=0,
                            span_end=len(text_content),
                            source_text=text_content,
                            translated_text=full_translation,
                            provider=provider,
                            model=model_id,
                        ))
                        # Force flush here to ensure it's visible to queries immediately
                        account_db.flush()
                        logger.info(f"[TRANSLATION] Synthesized full text translation for text_id={text_id}")
            except Exception as e:
                logger.error(f"[TRANSLATION] Failed to synthesize full text translation: {e}", exc_info=True)
                # Non-blocking failure
            
            # Flush the session to ensure data is written
            try:
                account_db.flush()
            except Exception as e:
                logger.error(f"[TRANSLATION] Failed to flush sentence translations: {e}", exc_info=True)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"[TRANSLATION] Sentence translation failed: {e}", exc_info=True)
            return False
    
    def _generate_title_translation(self,
                                account_db: Session,
                                account_id: int,
                                text_id: int,
                                lang: str,
                                target_lang: str,
                                text_title: str,
                                job_dir: Path,
                                provider: str,
                                model_id: Optional[str],
                                base_url: Optional[str],
                                user_api_key: Optional[str] = None,
                                model_config=None) -> bool:
        """Generate and persist title translation."""
        try:
            # Build title translation prompt
            from ..llm.prompts import build_title_translation_prompt
            title_messages = build_title_translation_prompt(lang, target_lang, text_title)
            
            # Call LLM for title translation
            title_buf, title_resp, used_provider, used_model = llm_call_and_log_to_file(
                title_messages, provider, model_id, base_url, job_dir / "title_translation.json",
                user_api_key=user_api_key,
                model_config=model_config
            )
            
            if not title_buf:
                return False
            
            # Log to DB (main thread) for title translation
            try:
                log_llm_request(
                    account_db,
                    account_id=account_id,
                    text_id=text_id,
                    kind="title_translation",
                    provider=used_provider,
                    model=used_model,
                    base_url=base_url,
                    request={"messages": title_messages},
                    response=title_resp,
                    status="ok",
                )
            except Exception:
                pass
            
            # Extract title translation
            try:
                # Try to parse as structured response
                title_parsed = extract_structured_translation(title_buf)
                title_translation = None
                
                if title_parsed and isinstance(title_parsed, dict):
                    # Try to get translation from structured response
                    paragraphs = title_parsed.get("paragraphs", [])
                    if paragraphs and len(paragraphs) > 0:
                        sentences = paragraphs[0].get("sentences", [])
                        if sentences and len(sentences) > 0:
                            title_translation = sentences[0].get("translation")
                
                # If structured parsing failed, try direct extraction
                if not title_translation:
                    title_translation = extract_json_from_text(title_buf, "translation")
                
                # Still no result, use the response content directly if it's simple
                if not title_translation and isinstance(title_buf, str) and len(title_buf) < 200:
                    title_translation = title_buf.strip()
                
                if not title_translation:
                    logger.warning(f"[TRANSLATION] Could not extract title translation from: {title_buf}")
                    return False
                
                # Save title translation to database
                account_db.add(ReadingTextTranslation(
                    account_id=account_id,
                    text_id=text_id,
                    unit="text",
                    target_lang=target_lang,
                    segment_index=0,
                    span_start=0,
                    span_end=len(text_title),
                    source_text=text_title,
                    translated_text=title_translation,
                    provider=used_provider,
                    model=used_model,
                ))
                
                # Flush the session to ensure title translation is written
                try:
                    account_db.flush()
                except Exception as e:
                    logger.error(f"[TRANSLATION] Failed to flush title translation: {e}", exc_info=True)
                    return False
                
                return True
                
            except Exception as e:
                logger.error(f"[TRANSLATION] Failed to parse title translation: {e}", exc_info=True)
                return False
            
        except Exception as e:
            logger.error(f"[TRANSLATION] Title translation failed: {e}", exc_info=True)
            return False
    
    def _split_sentences(self, text: str, lang: str) -> List[Tuple[int, int, str]]:
        """Deprecated; delegate to utils.text_segmentation.split_sentences."""
        return split_sentences(text, lang)

    def backfill_sentence_spans(self, account_db: Session, account_id: int, text_id: int) -> bool:
        """Backfill span_start/span_end for existing sentence translations by index order.

        Returns True if any updates were attempted.
        """
        try:
            rt = account_db.query(ReadingText).filter(ReadingText.id == text_id, ReadingText.account_id == account_id).first()
            if not rt or not getattr(rt, "content", None):
                return False
            sent_spans = self._split_sentences(rt.content, rt.lang)
            rows = (
                account_db.query(ReadingTextTranslation)
                .filter(
                    ReadingTextTranslation.account_id == account_id,
                    ReadingTextTranslation.text_id == text_id,
                    ReadingTextTranslation.unit == "sentence",
                )
                .order_by(ReadingTextTranslation.segment_index.asc().nullsfirst())
                .all()
            )
            changed = False
            for r in rows:
                if r.segment_index is None:
                    continue
                if r.span_start is not None and r.span_end is not None:
                    continue
                idx = int(r.segment_index)
                if 0 <= idx < len(sent_spans):
                    s, e, _ = sent_spans[idx]
                    r.span_start = s
                    r.span_end = e
                    changed = True
            if changed:
                try:
                    account_db.commit()
                except Exception:
                    account_db.rollback()
            return True
        except Exception:
            try:
                account_db.rollback()
            except Exception:
                pass
            return False
    
    # llm_call_and_log_to_file moved to llm_logging; keep class focused on orchestration
    
    def _persist_word_translations(self,
                                  account_db: Session,
                                  account_id: int,
                                  text_id: int,
                                  lang: str,
                                  seg_start: int,
                                  seg_text: str,
                                  tup: Optional[tuple],
                                  msgs_used: List[Dict],
                                  existing: set,
                                  seen: set) -> None:
        """Persist word translations for a sentence segment."""
        if not tup:
            return
        
        wd_buf, wd_resp = tup[0], tup[1]
        wd_parsed = extract_word_translations(wd_buf)
        if not wd_parsed:
            return
        
        words_list = [w for w in wd_parsed.get("words", []) if isinstance(w, dict) and w.get("word")]
        if not words_list:
            return
        
        spans = compute_spans(seg_text, words_list, key="word")
        for it, sp in zip(words_list, spans):
            if sp is None:
                continue
            gs = (seg_start + sp[0], seg_start + sp[1])
            
            if gs in existing or gs in seen:
                continue
            
            _pos_local = it.get("pos") if isinstance(it, dict) else None
            if not _pos_local and isinstance(it, dict):
                _pos_local = it.get("part_of_speech")
            
            account_db.add(ReadingWordGloss(
                account_id=account_id,
                text_id=text_id,
                lang=lang,
                surface=it["word"],
                lemma=(None if str(lang).startswith("zh") else it.get("lemma")),
                pos=_pos_local,
                pinyin=it.get("pinyin"),
                translation=it["translation"],
                lemma_translation=it.get("lemma_translation"),
                grammar=it.get("grammar", {}),
                span_start=gs[0],
                span_end=gs[1],
            ))
            seen.add(gs)
    
    def _get_existing_spans(self, account_db: Session, account_id: int, text_id: int) -> set:
        """Get existing word spans to avoid duplicates."""
        try:
            return set(
                (rw.span_start, rw.span_end)
                for rw in account_db.query(ReadingWordGloss.span_start, ReadingWordGloss.span_end)
                .filter(ReadingWordGloss.account_id == account_id, ReadingWordGloss.text_id == text_id)
                .all()
            )
        except Exception:
            return set()
