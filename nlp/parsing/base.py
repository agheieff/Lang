from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class LanguageEngine(Protocol):
    def analyze_word(self, surface: str, context: Optional[str] = None) -> Dict[str, Any]:
        ...


LabelFormatter = callable
