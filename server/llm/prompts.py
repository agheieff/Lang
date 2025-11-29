from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
def _safe_format(s: str, mapping: Dict[str, str]) -> str:
    """Replace only known {keys} using a safe formatter, leaving all other braces intact.

    Avoids KeyError and preserves embedded JSON examples like {"text": "..."}.
    """
    import re
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return str(mapping.get(key, m.group(0)))
    return re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, s)


def _load_prompt(name: str, lang_code: Optional[str] = None) -> str:
    base = Path(__file__).resolve().parent / "prompts"
    # Strict language-based lookup only: exact code, then base code. No inline defaults.
    candidates: List[Path] = []
    if lang_code:
        code = str(lang_code).strip()
        if code:
            candidates.append(base / code / name)
            base_code = code.split("-", 1)[0].split("_", 1)[0]
            if base_code and base_code != code:
                candidates.append(base / base_code / name)
    for p in candidates:
        try:
            if p.exists():
                return p.read_text(encoding="utf-8")
        except Exception:
            continue
    raise FileNotFoundError(f"Prompt file not found for lang={lang_code!r}, name={name!r}")


@dataclass
class PromptSpec:
    lang: str
    unit: str
    approx_len: int
    user_level_hint: Optional[str]
    include_words: Optional[List[str]]
    script: Optional[str] = None  # for zh source formatting
    ci_target: Optional[float] = None
    recent_titles: Optional[List[str]] = None
    topic: Optional[str] = None  # Topic category: fiction, news, science, history, daily_life, culture


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    # Use new file names (system.md, text.md)
    sys_tpl = _load_prompt("system.md", spec.lang)
    user_tpl = _load_prompt("text.md", spec.lang)

    lang_display = _lang_display(spec.lang)
    script_line = ""
    if spec.script and spec.lang.startswith("zh"):
        if spec.script == "Hans":
            script_line = "Use simplified Chinese characters.\n"
        elif spec.script == "Hant":
            script_line = "Use traditional Chinese characters.\n"
    level_line = f"The student is around {spec.user_level_hint}; please use appropriate language for this level.\n" if spec.user_level_hint else ""
    length_line = (
        f"The text should be around {spec.approx_len} characters long.\n" if spec.unit == "chars" else f"The text should be around {spec.approx_len} words long.\n"
    )
    include_words_line = ""
    if spec.include_words:
        words = ", ".join(spec.include_words)
        include_words_line = f"Please include these words naturally: {words}.\n"
    ci_line = ""
    if isinstance(spec.ci_target, (int, float)) and spec.ci_target:
        pct = int(round(float(spec.ci_target) * 100))
        ci_line = f"Aim for about {pct}% of tokens to be familiar for the learner; limit new vocabulary.\n"
    
    recent_titles_line = ""
    if spec.recent_titles:
        titles_str = ", ".join(f'"{t}"' for t in spec.recent_titles)
        recent_titles_line = f"Here are the titles of the last {len(spec.recent_titles)} texts the user read: {titles_str}. Please generate something different/new.\n"
    
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
        # Handle multiple comma-separated topics
        topics = [t.strip() for t in spec.topic.split(',') if t.strip()]
        if len(topics) == 1:
            topic_display = topic_display_map.get(topics[0], topics[0])
            topic_line = f"The text should be about {topic_display}.\n"
        elif len(topics) == 2:
            displays = [topic_display_map.get(t, t) for t in topics]
            topic_line = f"The text should combine {displays[0]} and {displays[1]}.\n"
        elif len(topics) >= 3:
            displays = [topic_display_map.get(t, t) for t in topics]
            topic_line = f"The text should touch on {', '.join(displays[:-1])}, and {displays[-1]}.\n"

    # Build mapping for both legacy "*_line" placeholders and simplified {level}/{length}/{include_words}/{script}
    # Prefer a concise level code (e.g., HSK2, A2) before any description
    if spec.user_level_hint and ":" in spec.user_level_hint:
        simple_level = spec.user_level_hint.split(":", 1)[0].strip()
    else:
        simple_level = spec.user_level_hint or ""
    simple_length = str(spec.approx_len)
    simple_include = ", ".join(spec.include_words or [])
    script_label = ""
    if spec.script and spec.lang.startswith("zh"):
        if spec.script == "Hans":
            script_label = "simplified"
        elif spec.script == "Hant":
            script_label = "traditional"
    mapping = {
        "lang_display": lang_display,
        # legacy multi-line placeholders
        "script_line": script_line,
        "level_line": level_line,
        "length_line": length_line,
        "include_words_line": include_words_line,
        "ci_line": ci_line,
        "recent_titles_line": recent_titles_line,
        "topic_line": topic_line,
        # simplified single-value placeholders
        "level": simple_level,
        "length": simple_length,
        "include_words": simple_include,
        "script": script_label,
        "recent_titles": (", ".join(spec.recent_titles) if spec.recent_titles else ""),
        "topic": (spec.topic or ""),
    }

    sys_content = _safe_format(sys_tpl, mapping)
    user_content = _safe_format(user_tpl, mapping)
    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]


@dataclass
class TranslationSpec:
    lang: str
    target_lang: str
    unit: str  # "sentence" | "paragraph" | "text"
    content: Union[str, List[str]]
    continue_with_reading: bool = False
    script: Optional[str] = None  # for zh source formatting


def build_translation_prompt(spec: TranslationSpec, prev_messages: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    # Map to the new structured translation template for consistency
    tpl = _load_prompt("translation.md", spec.lang)
    src_lang = _lang_display(spec.lang)
    tgt_lang = _lang_display(spec.target_lang)
    text = spec.content if isinstance(spec.content, str) else "\n".join(spec.content)
    user_content = _safe_format(tpl, {"source_lang": src_lang, "target_lang": tgt_lang, "text": text})

    msgs: List[Dict[str, str]] = []
    if prev_messages:
        msgs.extend(prev_messages)
    msgs.append({"role": "system", "content": ""})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def build_structured_translation_prompt(source_lang: str, target_lang: str, text: str) -> List[Dict[str, str]]:
    """Build prompt for structured sentence-by-sentence translation."""
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    # Use single-file translation.md (structured by parts)
    tpl = _load_prompt("translation.md", source_lang)
    user_content = _safe_format(
        tpl,
        {
            "source_lang": _lang_display(source_lang),
            "target_lang": _lang_display(target_lang),
            "text": text,
        },
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]


def build_word_translation_prompt(source_lang: str, target_lang: str, text: str) -> List[Dict[str, str]]:
    """Build prompt for word-by-word translation with linguistic analysis."""
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        if code.startswith("fr"):
            return "French"
        return code

    # Use single-file words.md (word-by-word)
    tpl = _load_prompt("words.md", source_lang)
    # Support both {text} and {sentence} placeholders in templates
    user_content = _safe_format(
        tpl,
        {
            "source_lang": _lang_display(source_lang),
            "target_lang": _lang_display(target_lang),
            "text": text,
            "sentence": text,
        },
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
    ]


def build_translation_contexts(
    reading_messages: List[Dict[str, str]],
    *,
    source_lang: str,
    target_lang: str,
    text: str,
) -> Dict[str, List[Dict[str, str]]]:
    """Return canonical 4-message contexts for structured and word translations.

    Messages shape (both kinds):
      [
        {system: translation_system},
        {user: reading_user_content},
        {assistant: generated_text},
        {user: task_user_prompt}
      ]
    """
    reading_user_content = reading_messages[1]["content"] if (reading_messages and len(reading_messages) > 1) else ""

    # Structured
    tr_msgs = build_structured_translation_prompt(source_lang, target_lang, text)
    tr_system = tr_msgs[0]["content"]
    tr_user = tr_msgs[1]["content"]
    structured = [
        {"role": "system", "content": tr_system},
        {"role": "user", "content": reading_user_content},
        {"role": "assistant", "content": text},
        {"role": "user", "content": tr_user},
    ]

    # Word-by-word
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

def build_title_translation_prompt(source_lang: str, target_lang: str, title: str) -> List[Dict[str, str]]:
    """Build prompt for title translation."""
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        if code.startswith("fr"):
            return "French"
        return code
    
    # Simple prompt for title translation
    system_content = f"You are a professional translator. Translate the following {_lang_display(source_lang)} title to {_lang_display(target_lang)}. Provide only the translated title without any additional text or explanations."
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": title},
    ]