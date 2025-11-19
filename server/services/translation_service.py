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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from ..services.model_registry_service import get_model_registry
from ..auth import Account
from ..llm.prompts import build_translation_contexts, build_word_translation_prompt
from ..utils.text_segmentation import split_sentences
from .llm_logging import log_llm_request, llm_call_and_log_to_file
from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from ..utils.json_parser import (
    extract_structured_translation,
    extract_word_translations,
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
                                   provider: str = None,  # Will be determined by model registry
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
        
        # Get model configuration from registry or fall back to legacy behavior
        if provider is None or model_id is None:
                try:
                    registry = get_model_registry()
                    model_config = registry.get_default_model(user_tier)
                    provider = "openrouter" if "openrouter" in model_config.base_url else "local"
                    model_id = model_config.model
                    base_url = model_config.base_url
                except Exception:
                    # Fallback to legacy defaults
                    provider = provider or "openrouter"
                    model_id = model_id or _pick_openrouter_model(None)
                    base_url = base_url or "http://localhost:1234/v1"
        
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
        
        # Execute generation steps sequentially to avoid sharing a DB session across threads
        words_success = self._generate_word_translations(
            account_db, account_id, text_id, lang, target_lang,
            text_content, job_dir, provider, model_id, base_url,
            w_messages, reading_messages, text_content
        )
        sentences_success = self._generate_sentence_translations(
            account_db, account_id, text_id, lang, target_lang,
            text_content, job_dir, provider, model_id, base_url,
            tr_messages
        )
            
            # Create result
        success = words_success or sentences_success  # Partial success is ok
        return TranslationResult(
            success=success,
            words=words_success,
            sentences=sentences_success,
            log_dir=job_dir
        )
    
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
                                    full_response: str) -> bool:
        """Generate word translations split by sentence."""
        try:
            # Split text into sentences
            sent_spans = split_sentences(text_content, lang)
            
            # Build per-sentence messages preserving reading context
            per_msgs: List[Tuple[int, str, List[Dict]]] = []
            for (s, e, seg) in sent_spans:
                msgs = build_word_translation_prompt(lang, target_lang, seg)
                words_ctx = [
                    {"role": "system", "content": msgs[0]["content"]},
                    {"role": "user", "content": reading_messages[1]["content"]},
                    {"role": "assistant", "content": full_response},
                    {"role": "user", "content": msgs[1]["content"]},
                ]
                per_msgs.append((s, seg, words_ctx))
            
            # Process sentences in parallel
            wd_seg_results: List[Tuple[int, str, Optional[tuple], List[Dict]]] = []
            
            max_workers = max(1, min(self.max_workers, len(per_msgs)))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for i, (s, seg, m) in enumerate(per_msgs):
                    # Shift by +1 so words_1 is the first sentence
                    out_path = job_dir / f"words_{i+1}.json"
                    futures.append((s, seg, m, ex.submit(
                        llm_call_and_log_to_file,
                        m, provider, model_id, base_url, out_path
                    )))
                
                for s, seg, m, f in futures:
                    try:
                        wd_seg_results.append((s, seg, f.result(), m))
                    except Exception:
                        wd_seg_results.append((s, seg, None, m))
        
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
        
            return True
            
        except Exception as e:
            print(f"[TRANSLATION] Word translation failed: {e}")
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
                                      tr_messages: List[Dict]) -> bool:
        """Generate structured sentence translations."""
        try:
            # Call LLM for structured translation
            tr_buf, tr_resp, used_provider, used_model = llm_call_and_log_to_file(
                tr_messages, provider, model_id, base_url, job_dir / "structured.json"
            )
            
            if not tr_buf:
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
                        try:
                            if 0 <= idx < len(sent_spans):
                                span_start, span_end, _ = sent_spans[idx]
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
                            source_text=s["text"],
                            translated_text=s["translation"],
                            provider=provider,
                            model=model_id,
                        ))
                        idx += 1
            
            return True
            
        except Exception as e:
            print(f"[TRANSLATION] Sentence translation failed: {e}")
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
