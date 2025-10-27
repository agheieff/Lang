from __future__ import annotations

from typing import Dict
from .base import Tokenizer
from .latin import LatinTokenizer
from .zh import ZhTokenizer


TOKENIZERS: Dict[str, Tokenizer] = {
    # Default Latin-script tokenizer; language-specific ones can override later
    "default": LatinTokenizer(),
    "es": LatinTokenizer(),
    "zh": ZhTokenizer(),
    "zh-Hans": ZhTokenizer(),
    "zh-Hant": ZhTokenizer(),
}
