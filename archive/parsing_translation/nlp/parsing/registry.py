from __future__ import annotations

from typing import Any, Dict, Optional

from .base import LanguageEngine
from .es import analyze_word_es
from .zh import analyze_word_zh


class _EsEngine:
    def analyze_word(self, surface: str, context: Optional[str] = None) -> Dict[str, Any]:
        return analyze_word_es(surface, context)


ENGINES: Dict[str, LanguageEngine] = {
    "es": _EsEngine(),
    "zh": type("ZhEngine", (), {"analyze_word": staticmethod(analyze_word_zh)})(),
    "zh-Hans": type("ZhEngine", (), {"analyze_word": staticmethod(analyze_word_zh)})(),
    "zh-Hant": type("ZhEngine", (), {"analyze_word": staticmethod(analyze_word_zh)})(),
}
