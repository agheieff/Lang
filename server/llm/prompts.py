from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


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

    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    sys_tpl = _load_prompt("reading_system.txt") or (
        "You are a tutor of {lang_display}. Please generate a text for learning and comprehensible input practice, given the following parameters."
    )
    user_tpl = _load_prompt("reading_user.txt") or (
        "Write in {lang_display}.\n{script_line}{level_line}{length_line}{include_words_line}Constraints:\n- Do not include translations or vocabulary lists.\n- Avoid English unless the target language is English.\n- Gently reinforce the target words in context and keep the language natural and engaging.\n- Do not include meta commentary.\n- Separate paragraphs with double newlines (\\n\\n)."
    )

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

    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    # Use regular translation prompts
    sys_tpl = _load_prompt("translation_system.txt") or (
        "You are a professional translator from {src_lang} to {tgt_lang}. Output only the translation, no explanations. {line_mode_line}{script_line}"
    )
    user_tpl = _load_prompt("translation_user.txt") or ("{content}")

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
    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        return code

    sys_tpl = _load_prompt("structured_translation_system.txt") or (
        "You are a professional translator. Please translate the provided text sentence-by-sentence, preserving paragraph structure. "
        "Return a JSON object with 'text' (original), 'paragraphs' array containing 'sentences' arrays with 'text' and 'translation'."
    )
    user_tpl = _load_prompt("structured_translation_user.txt") or (
        "Translate this text from {source_lang} to {target_lang}:\n\n{text}"
    )

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
    def _load_prompt(name: str) -> str:
        p = Path(__file__).resolve().parent / "prompts" / name
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _lang_display(code: str) -> str:
        if code.startswith("zh"):
            return "Chinese"
        if code.startswith("es"):
            return "Spanish"
        if code.startswith("fr"):
            return "French"
        return code

    sys_tpl = _load_prompt("word_translation_system.txt") or (
        "You are a professional linguist and translator. Please analyze the provided text and provide detailed word-by-word translations. "
        "Return a JSON object with 'text' (original) and 'words' array containing word objects with translations, lemmas, and grammatical information."
    )
    user_tpl = _load_prompt("word_translation_user.txt") or (
        "Analyze and translate each word in this {source_lang} text:\n\n{text}"
    )

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