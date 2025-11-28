from __future__ import annotations

"""
HTML renderer for the reading block.

Only concerns markup assembly; no DB or business logic.
"""

import json
from typing import Optional, Iterable


def _safe_html(text: Optional[str]) -> str:
    """Escape untrusted text for safe HTML display. Preserve newlines as <br>."""
    if not text:
        return ""
    try:
        from markupsafe import escape  # type: ignore
        esc = str(escape(str(text).replace("\r\n", "\n").replace("\r", "\n")))
    except Exception:
        norm = str(text).replace("\r\n", "\n").replace("\r", "\n")
        esc = (
            norm
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    return esc.replace("\n", "<br>")


def _words_json(rows: Iterable) -> list[dict]:
    return [
        {
            "surface": w.surface,
            "lemma": w.lemma,
            "pos": w.pos,
            "pinyin": w.pinyin,
            "translation": w.translation,
            "lemma_translation": w.lemma_translation,
            "grammar": w.grammar,
            "span_start": w.span_start,
            "span_end": w.span_end,
        }
        for w in rows
    ]


def render_loading_block(kind: str = "loading") -> str:
    """Skeleton/loading placeholder for the reading area.

    Always returns a wrapper with id="reading-block" so callers and HTMX
    can reliably target it regardless of whether a text exists yet.
    """
    if kind == "generating":
        message = "Generating text…"
    else:
        message = "Loading text…"

    skeleton = (
        '<div class="text-center py-8">'
        '  <div class="animate-pulse space-y-3">'
        '    <div class="h-4 bg-gray-200 rounded w-3/4"></div>'
        '    <div class="h-4 bg-gray-200 rounded w-5/6"></div>'
        '    <div class="h-4 bg-gray-200 rounded w-2/3"></div>'
        '    <div class="h-4 bg-gray-200 rounded w-4/5"></div>'
        '  </div>'
        f'  <div class="mt-2 text-sm text-gray-500">{message}</div>'
        '</div>'
    )

    return '<div id="reading-block">' + skeleton + '</div>'


def _get_status_text(reason: str) -> str:
    """Get human-readable status text for next button readiness reason."""
    return {
        "both": "Ready",
        "grace": "Ready",
        "content_only": "Ready (text only)",
        "waiting": "Processing...",
        "no_content": "Generating...",
    }.get(reason, "Loading next...")


def render_reading_block(
    text_id: int,
    raw_content: str,
    words_rows: Iterable,
    *,
    title: Optional[str] = None,
    title_words: Optional[list] = None,
    title_translation: Optional[str] = None,
    session_state: Optional[dict] = None,
    is_next_ready: bool = False,
    next_ready_reason: str = "waiting",
) -> str:
    html_content = _safe_html(raw_content)
    words_json = _words_json(words_rows)
    json_text = json.dumps(words_json, ensure_ascii=False).replace('</', '<\\/')
    title_part = (f'<h2 id="reading-title" class="text-2xl font-bold mb-3">{_safe_html(title)}</h2>' if title else '')
    title_words_json = json.dumps(title_words or [], ensure_ascii=False).replace('</', '<\\/')
    
    # Use session state if available
    session_state_json = ""
    if session_state:
        session_state_json = json.dumps(session_state, ensure_ascii=False).replace('</', '<\\/')

    # Button state based on next text readiness
    btn_disabled = "" if is_next_ready else "disabled"
    btn_aria_disabled = "false" if is_next_ready else "true"
    status_text = _get_status_text(next_ready_reason)
    status_class = "text-green-500" if is_next_ready else "text-gray-500"

    return (
        '<div id="reading-block">'
        + title_part
        + f'<div id="reading-text" class="prose max-w-none" data-text-id="{text_id}">{html_content}</div>'
        + '<div class="mt-4 flex items-end w-full">'
        + '  <div class="flex items-center gap-3 flex-1">'
        + '    <button id="next-btn"'
        + '      onclick="window.handleNextText()"'
        + '      class="px-4 py-2 rounded-lg transition-colors text-white bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed"'
        + f'      {btn_disabled} aria-disabled="{btn_aria_disabled}">Next text</button>'
        + f'    <span id="next-status" class="ml-3 text-sm {status_class}" aria-live="polite">{status_text}</span>'
        + '  </div>'
        + '  <button id="see-translation-btn" type="button"'
        + '    class="ml-4 shrink-0 px-3 py-1.5 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50"'
        + '    onclick="window.arcToggleTranslation && window.arcToggleTranslation(event)"'
        + '    aria-expanded="false">See translation</button>'
        + '</div>'
        + '<div id="translation-panel" class="hidden mt-4" hidden>'
        + '  <div id="translation-content" class="prose max-w-none text-gray-800">'
        + '    <hr class="my-3">'
        + '    <div id="translation-text" class="whitespace-pre-wrap"></div>'
        + '  </div>'
        + '</div>'
        + f'<script id="reading-words-json" type="application/json">{json_text}</script>'
        + (f'<script id="reading-title-words-json" type="application/json">{title_words_json}</script>' if (title and title_words_json) else '')
        + (f'<script id="reading-title-translation" type="application/json">{json.dumps(title_translation, ensure_ascii=False) if title_translation else ""}</script>' if title_translation else '')
        + (f'<script id="reading-session-state" type="application/json">{session_state_json}</script>' if session_state_json else '')
        + '<div id="word-tooltip" class="absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm max-w-xs hidden"></div>'
        + '<script src="/static/reading-sse.js"></script>'
        + '<script src="/static/reading.js" defer></script>'
        + '</div>'
    )
