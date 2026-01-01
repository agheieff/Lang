from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt_from_file(filename: str) -> Optional[str]:
    """Load a prompt file directly from the prompts directory."""
    try:
        file_path = PROMPTS_DIR / filename
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None
    except Exception as e:
        logger.warning(f"Failed to load prompt file {filename}: {e}")
        return None


def _load_prompt_file(category: str, lang: str) -> Optional[str]:
    """Load a prompt file from the prompts directory."""
    try:
        lang_base = lang.split("-", 1)[0].split("_", 1)[0]

        file_path = PROMPTS_DIR / category / f"{lang}.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")

        file_path = PROMPTS_DIR / category / f"{lang_base}.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")

        file_path = PROMPTS_DIR / category / "default.md"
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")

        return None
    except Exception as e:
        logger.warning(f"Failed to load prompt file for {category}/{lang}: {e}")
        return None


def _safe_format(s: str, mapping: Dict[str, str]) -> str:
    """Replace only known {keys} using a safe formatter, leaving all other braces intact.

    Avoids KeyError and preserves embedded JSON examples like {"text": "..."}.
    """
    import re

    def repl(m) -> str:
        key = m.group(1)
        return str(mapping.get(key, m.group(0)))

    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, s)


@dataclass
class PromptSpec:
    lang: str
    unit: str
    approx_len: int
    user_level_hint: Optional[str]
    include_words: Optional[List[str]]
    script: Optional[str] = None
    ci_target: Optional[float] = None
    recent_titles: Optional[List[str]] = None
    topic: Optional[str] = None


def get_prompt(lang: str, key: str, **kwargs) -> str:
    """Get a prompt template for a given language and key, with formatting."""
    if key == "system":
        template = _load_prompt_from_file("system.md")
        if not template:
            return "You are a language tutor creating texts for comprehensible input training."
        return _safe_format(template, kwargs)

    category_map = {
        "text": "text_generation",
        "translation": "sentence_translation",
        "words": "word_analysis",
    }

    category = category_map.get(key)
    if not category:
        logger.warning(f"Unknown prompt key: {key}")
        return "Please assist with language learning."

    template = _load_prompt_file(category, lang)
    if not template:
        logger.warning(f"Prompt not found for lang={lang}, key={key}, using default")
        template = _load_prompt_file(category, "default")

    if not template:
        return "Please assist with language learning."

    return _safe_format(template, kwargs)


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    """Build reading prompt messages for LLM."""
    level = (
        spec.user_level_hint.split(":")[0]
        if spec.user_level_hint and ":" in spec.user_level_hint
        else spec.user_level_hint or ""
    )

    topic_line = ""
    if spec.topic:
        topic_display_map = {
            "fiction": "fiction/creative writing",
            "news": "news/current events",
            "science": "science",
            "technology": "technology",
            "history": "history",
            "daily_life": "daily life/practical situations",
            "culture": "culture/traditions",
            "sports": "sports",
            "business": "business/economics",
        }
        topics = [t.strip() for t in spec.topic.split(",") if t.strip()]
        if len(topics) == 1:
            topic_display = topic_display_map.get(topics[0], topics[0])
            topic_line = f"The text should be about {topic_display}.\n"
        elif len(topics) == 2:
            displays = [topic_display_map.get(t, t) for t in topics]
            topic_line = f"The text should combine {displays[0]} and {displays[1]}.\n"
        elif len(topics) >= 3:
            displays = [topic_display_map.get(t, t) for t in topics]
            topic_line = f"The text should touch on {', '.join(displays[:-1])}, and {displays[-1]}.\n"

    system_content = get_prompt(spec.lang, "system")
    user_content = get_prompt(
        spec.lang,
        "text",
        level=level,
        length=str(spec.approx_len),
        include_words=", ".join(spec.include_words or []),
        topic_line=topic_line,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_structured_translation_prompt(
    source_lang: str, target_lang: str, text: str
) -> List[Dict[str, str]]:
    """Build prompt for structured sentence-by-sentence translation."""
    lang_display = _get_language_display(source_lang)
    target_display = _get_language_display(target_lang)

    system_content = ""
    user_content = get_prompt(
        source_lang,
        "translation",
        source_lang=lang_display,
        target_lang=target_display,
        text=text,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_word_translation_prompt(
    source_lang: str, target_lang: str, text: str
) -> List[Dict[str, str]]:
    """Build prompt for word-by-word translation with linguistic analysis."""
    lang_display = _get_language_display(source_lang)
    target_display = _get_language_display(target_lang)

    system_content = ""
    user_content = get_prompt(
        source_lang,
        "words",
        source_lang=lang_display,
        target_lang=target_display,
        text=text,
        sentence=text,
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_translation_contexts(
    reading_messages: List[Dict[str, str]],
    *,
    source_lang: str,
    target_lang: str,
    text: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Return canonical 4-message contexts for structured and word translations."""
    reading_user_content = (
        reading_messages[1]["content"]
        if (reading_messages and len(reading_messages) > 1)
        else ""
    )

    tr_msgs = build_structured_translation_prompt(source_lang, target_lang, text)
    tr_system = tr_msgs[0]["content"]
    tr_user = tr_msgs[1]["content"]
    structured = [
        {"role": "system", "content": tr_system},
        {"role": "user", "content": reading_user_content},
        {"role": "assistant", "content": text},
        {"role": "user", "content": tr_user},
    ]

    w_msgs = build_word_translation_prompt(source_lang, target_lang, text)
    w_system = w_msgs[0]["content"]
    w_user = w_msgs[1]["content"]
    words = [
        {"role": "system", "content": w_system},
        {"role": "user", "content": reading_user_content},
        {"role": "assistant", "content": text},
        {"role": "user", "content": w_user},
    ]

    return {"structured": structured, "words": words}


def build_title_translation_prompt(
    source_lang: str, target_lang: str, title: str
) -> List[Dict[str, str]]:
    """Build prompt for title translation."""
    lang_display = _get_language_display(source_lang)
    target_display = _get_language_display(target_lang)

    system_content = f"You are a professional translator. Translate the following {lang_display} title to {target_display}. Provide only the translated title without any additional text or explanations."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": title},
    ]


def _get_language_display(lang_code: str) -> str:
    """Get language display name from database or fallback to hardcoded mapping."""
    try:
        from server.db import SessionLocal
        from server.models import Language

        with SessionLocal() as db:
            lang = db.query(Language).filter(Language.code == lang_code).first()
            if lang:
                return lang.name
    except Exception:
        pass

    fallback_map = {
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "en": "English",
    }
    return fallback_map.get(lang_code, lang_code)
