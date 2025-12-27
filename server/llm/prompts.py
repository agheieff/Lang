from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

PROMPTS = {
    "zh": {
        "system": "You're a language tutor, creating texts for learners to practice comprehensible input.",
        "text": """Please generate a text in Chinese for the comprehensible input training.
Write it in {script} characters.
The learner is around {level} level, please try to make the text correspond to that.
Please try to make the text around {length} characters long.
Try to include at least some of the following words in the text: {include_words}. Gently reinforce the target words in context and keep the language natural and engaging.
Separate paragraphs with double newlines (\\n\\n).
{topic_line}
Please output the title and text in pipe-separated CSV format with a header row:
title|text
去公园散步|今天天气很好，阳光明媚。微风轻轻吹过脸庞。\\n\\n我们一起去公园散步吧。""",
        "translation": """Now please translate this text into English sentence by sentence. Return pipe-separated CSV format:
source|translation
今天天气很好，阳光明媚。|The weather is great today, with bright sunshine.
微风轻轻吹过脸庞。|A gentle breeze brushes across the face.
我们一起去公园散步吧。|Let's go for a walk in the park together.

Translate the title as well:
title|translation
去公园散步|A walk in the park""",
        "words": """Now please analyze the following sentence: {sentence}

Return pipe-separated CSV format for each word in order:
word|translation|pos|lemma|pinyin

Tokenize into smallest meaningful units (words, not characters unless single-char words).
For words with multiple characters like "吃饭", tokenize as one word.
For compounds like "一个", tokenize as two words.
Skip punctuation.
Include all words in order, including duplicates.

Example for the sentence "今天天气很好，阳光明媚。":
今天|today|NOUN|今天|jīntiān
天气|weather|NOUN|天气|tiānqì
很|very|ADV|很|hěn
好|good|ADJ|好|hǎo
阳光|sunshine|NOUN|阳光|yáng guāng
明媚|bright and beautiful|ADJ|明媚|míng mèi""",
    },
    "es": {
        "system": "You're a language tutor, creating texts for learners to practice comprehensible input.",
        "text": """Please generate a text in Spanish for the comprehensible input training.
The learner is around {level} level; please write accordingly.
Please try to make the text around {length} words long.
Try to include at least some of the following words in the text: {include_words}. Gently reinforce the target words in context and keep the language natural and engaging.
Separate paragraphs with double newlines (\\n\\n).
{topic_line}
Please output the title and text in pipe-separated CSV format with a header row:
title|text
Un paseo en el parque|Hoy hace buen tiempo.\\n\\nVamos juntos a pasear por el parque.

For words with separable prefixes (like German), mark them with ... to indicate non-continuity:
word|translation|pos|lemma
ruf...an|call|VERB|anrufen""",
        "translation": """Translate this text from {source_lang} to {target_lang} sentence by sentence:
source|translation

{text}

Example format:
Hola.|Hello.
¿Cómo estás?|How are you?""",
        "words": """Analyze and translate each word in this {source_lang} text.

Return pipe-separated CSV format:
word|translation|pos|lemma

For German separable prefixes, use ... notation:
Ich|I|PRON|ich
ruf...an|call|VERB|anrufen
meinen|my|DET|mein
Freund|friend|NOUN|Freund
an|at|PREP|an

Example format:
Hola|Hello|INTJ|hola
mundo|world|NOUN|mundo""",
    },
}


def _safe_format(s: str, mapping: Dict[str, str]) -> str:
    """Replace only known {keys} using a safe formatter, leaving all other braces intact.

    Avoids KeyError and preserves embedded JSON examples like {"text": "..."}.
    """
    import re

    def repl(m: re.Match[str]) -> str:
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
    script: Optional[str] = None  # for zh source formatting
    ci_target: Optional[float] = None
    recent_titles: Optional[List[str]] = None
    topic: Optional[str] = (
        None  # Topic category: fiction, news, science, history, daily_life, culture
    )


def get_prompt(lang: str, key: str, **kwargs) -> str:
    """Get a prompt template for a given language and key, with formatting."""
    base_lang = lang.split("-", 1)[0].split("_", 1)[0]  # Get base language code
    template = PROMPTS.get(lang, {}).get(key) or PROMPTS.get(base_lang, {}).get(key)

    if not template:
        raise ValueError(f"Prompt not found for lang={lang}, key={key}")

    return _safe_format(template, kwargs)


def build_reading_prompt(spec: PromptSpec) -> List[Dict[str, str]]:
    """Build reading prompt messages for LLM."""
    # Simplified prompt building
    script = (
        "simplified"
        if spec.script == "Hans"
        else "traditional"
        if spec.script == "Hant"
        else ""
    )
    level = (
        spec.user_level_hint.split(":")[0]
        if spec.user_level_hint and ":" in spec.user_level_hint
        else spec.user_level_hint or ""
    )

    # Format topic
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
        script=script,
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
    lang_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        source_lang, source_lang
    )
    target_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        target_lang, target_lang
    )

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
    lang_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        source_lang, source_lang
    )
    target_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        target_lang, target_lang
    )

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


def build_title_translation_prompt(
    source_lang: str, target_lang: str, title: str
) -> List[Dict[str, str]]:
    """Build prompt for title translation."""
    lang_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        source_lang, source_lang
    )
    target_display = {"zh": "Chinese", "es": "Spanish", "fr": "French"}.get(
        target_lang, target_lang
    )

    system_content = f"You are a professional translator. Translate the following {lang_display} title to {target_display}. Provide only the translated title without any additional text or explanations."

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": title},
    ]
