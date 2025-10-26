from __future__ import annotations

# Re-export implementations from Lang/Lang
from Lang.parsing.registry import ENGINES  # type: ignore
from Lang.parsing.morph_format import format_morph_label  # type: ignore

__all__ = ["ENGINES", "format_morph_label"]
