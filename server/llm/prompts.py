from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    sys_tpl = _load_prompt("reading_system.txt", spec.lang)
    user_tpl = _load_prompt("reading_user.txt", spec.lang)

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

    sys_content = sys_tpl.format(lang_display=lang_display)
    user_content = user_tpl.format(
        lang_display=lang_display,
        script_line=script_line,
        level_line=level_line,
        length_line=length_line,
        include_words_line=include_words_line,
        ci_line=ci_line,
    )
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

    # Use regular translation prompts
    sys_tpl = _load_prompt("translation_system.txt", spec.lang)
    user_tpl = _load_prompt("translation_user.txt", spec.lang)

    src_lang = _lang_display(spec.lang)
    tgt_lang = _lang_display(spec.target_lang)
    line_mode_line = (
        "Translate each input line independently and return exactly one line per input line in the same order."
        if isinstance(spec.content, list) else ""
    )
    script_line = ""
    if spec.script and spec.lang.startswith("zh"):
        if spec.script == "Hans":
            script_line = "Source text may be in simplified Chinese."
        elif spec.script == "Hant":
            script_line = "Source text may be in traditional Chinese."

    sys_content = sys_tpl.format(src_lang=src_lang, tgt_lang=tgt_lang, line_mode_line=line_mode_line, script_line=script_line)
    lines = spec.content if isinstance(spec.content, list) else [spec.content]
    user_content = user_tpl.format(content="\n".join(lines))

    msgs: List[Dict[str, str]] = []
    if prev_messages:
        msgs.extend(prev_messages)
    msgs.append({"role": "system", "content": sys_content})
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

    sys_tpl = _load_prompt("structured_translation_system.txt", source_lang)
    user_tpl = _load_prompt("structured_translation_user.txt", source_lang)

    sys_content = sys_tpl
    user_content = user_tpl.format(
        source_lang=_lang_display(source_lang),
        target_lang=_lang_display(target_lang),
        text=text
    )

    return [
        {"role": "system", "content": sys_content},
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

    sys_tpl = _load_prompt("word_translation_system.txt", source_lang)
    user_tpl = _load_prompt("word_translation_user.txt", source_lang)

    sys_content = sys_tpl
    user_content = user_tpl.format(
        source_lang=_lang_display(source_lang),
        target_lang=_lang_display(target_lang),
        text=text
    )

    return [
        {"role": "system", "content": sys_content},
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