from __future__ import annotations

"""
Retry actions for missing components (words, sentences).
These are extracted from the legacy gen_queue implementation and adapted
to the new modular services. They rely on existing job logs (text.json)
to reconstruct prompts and context, and then persist results into the DB.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from ..utils.session_manager import db_manager

from ..models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from ..utils.json_parser import extract_structured_translation, extract_word_translations
from ..utils.gloss import compute_spans
from ..llm.client import _pick_openrouter_model, chat_complete_with_raw
from ..utils.text_segmentation import split_sentences
    return out


def retry_missing_words(account_id: int, text_id: int, log_dir: Path) -> bool:
    """Retry generation of missing word translations for a text.
    Uses the original text.json messages and full response to build per-sentence word prompts.
    """
    with db_manager.transaction(account_id) as db:
        rt = db.query(ReadingText).filter(ReadingText.id == text_id).first()
        if not rt or not rt.content:
            return False

        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == rt.lang).first()
        target_lang = (prof.target_lang if prof and getattr(prof, "target_lang", None) else "en")

        text_json = log_dir / "text.json"
        if not text_json.exists():
            return False
        text_log = json.loads(text_json.read_text(encoding="utf-8"))
        original_messages = text_log.get("request", {}).get("messages")
        text_content = (
            text_log.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
        )
        if not original_messages or not text_content:
            return False

        # Extract main text string (handle possible JSON-in-text cases)
        import re as _re
        m = _re.search(r'\{[^}]*"text":\s*"([^"]+)"', text_content)
        main_text = m.group(1) if m else text_content
        main_text = main_text.replace("\\n", "\n")

        sent_spans = split_sentences(main_text, rt.lang)

        provider = "openrouter"
        model_id = _pick_openrouter_model(None)

        from ..llm.prompts import build_word_translation_prompt

        for i, (s, e, seg) in enumerate(sent_spans):
            try:
                msgs = build_word_translation_prompt(rt.lang, target_lang, seg)
                words_ctx = [
                    {"role": "system", "content": msgs[0]["content"]},
                    {"role": "user", "content": original_messages[1]["content"] if len(original_messages) > 1 else ""},
                    {"role": "assistant", "content": text_content},
                    {"role": "user", "content": msgs[1]["content"]},
                ]
                words_text, words_resp = chat_complete_with_raw(
                    words_ctx, provider=provider, model=model_id, max_tokens=16384
                )
                # Log file for this retry call
                try:
                    words_file = log_dir / f"words_retry_{i+1}.json"
                    words_log = {
                        "request": {"provider": provider, "model": model_id, "messages": words_ctx, "retry": True},
                        "response": words_resp,
                    }
                    words_file.write_text(json.dumps(words_log, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

                # Parse and persist
                wd_parsed = extract_word_translations(words_text or "")
                if not wd_parsed or not wd_parsed.get("words"):
                    continue
                words_list = [w for w in wd_parsed.get("words", []) if isinstance(w, dict) and w.get("word")]
                if not words_list:
                    continue
                spans = compute_spans(seg, words_list, key="word")
                for it, sp in zip(words_list, spans):
                    if sp is None:
                        continue
                    gs = (s + sp[0], s + sp[1])
                    pos_local = it.get("pos") if isinstance(it, dict) else None
                    if not pos_local and isinstance(it, dict):
                        pos_local = it.get("part_of_speech")
                    db.add(ReadingWordGloss(
                        account_id=account_id,
                        text_id=text_id,
                        lang=rt.lang,
                        surface=it["word"],
                        lemma=(None if str(rt.lang).startswith("zh") else it.get("lemma")),
                        pos=pos_local,
                        pinyin=it.get("pinyin"),
                        translation=it["translation"],
                        lemma_translation=it.get("lemma_translation"),
                        grammar=it.get("grammar", {}),
                        span_start=gs[0],
                        span_end=gs[1],
                    ))
            except Exception:
                # continue best-effort on per-sentence failures
                continue

        return True
        # Auto commit/rollback handled by db_manager.transaction context manager


def retry_missing_sentences(account_id: int, text_id: int, log_dir: Path) -> bool:
    """Retry generation of missing structured sentence translations for a text."""
    with db_manager.transaction(account_id) as db:
        rt = db.query(ReadingText).filter(ReadingText.id == text_id).first()
        if not rt or not rt.content:
            return False

        prof = db.query(Profile).filter(Profile.account_id == account_id, Profile.lang == rt.lang).first()
        target_lang = (prof.target_lang if prof and getattr(prof, "target_lang", None) else "en")

        text_json = log_dir / "text.json"
        if not text_json.exists():
            return False
        text_log = json.loads(text_json.read_text(encoding="utf-8"))
        original_messages = text_log.get("request", {}).get("messages")
        text_content = (
            text_log.get("response", {})
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
        )
        if not original_messages or not text_content:
            return False

        # Extract main text string
        import re as _re
        m = _re.search(r'\{[^}]*"text":\s*"([^"]+)"', text_content)
        main_text = m.group(1) if m else text_content
        main_text = main_text.replace("\\n", "\n")

        from ..llm.prompts import build_translation_contexts
        ctx = build_translation_contexts(
            original_messages,
            source_lang=rt.lang,
            target_lang=target_lang,
            text=main_text,
        )
        tr_messages = ctx["structured"]
        if isinstance(tr_messages, list) and len(tr_messages) > 2:
            tr_messages[2]["content"] = text_content

        provider = "openrouter"
        model_id = _pick_openrouter_model(None)
        tr_text, tr_resp = chat_complete_with_raw(
            tr_messages, provider=provider, model=model_id, max_tokens=16384
        )

        # Write a retry log file for structured
        try:
            structured_file = log_dir / "structured_retry.json"
            structured_log = {
                "request": {"provider": provider, "model": model_id, "messages": tr_messages, "retry": True},
                "response": tr_resp,
            }
            structured_file.write_text(json.dumps(structured_log, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        tr_parsed = extract_structured_translation(tr_text or "")
        if not tr_parsed or not tr_parsed.get("paragraphs"):
            return False

        idx = 0
        for p in tr_parsed.get("paragraphs", []):
            for s in p.get("sentences", []):
                if "text" in s and "translation" in s:
                    db.add(ReadingTextTranslation(
                        account_id=account_id,
                        text_id=text_id,
                        unit="sentence",
                        target_lang=(tr_parsed.get("target_lang") or target_lang),
                        segment_index=idx,
                        span_start=None,
                        span_end=None,
                        source_text=s["text"],
                        translated_text=s["translation"],
                        provider=provider,
                        model=model_id,
                    ))
                    idx += 1
        return True
        # Auto commit/rollback handled by db_manager.transaction context manager
