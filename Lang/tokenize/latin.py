from __future__ import annotations

import regex as re  # use the 'regex' module for Unicode \p{L}/\p{M}
from typing import List
from .base import Token, Tokenizer

# Match only word spans (letters/combining marks). Non-word runs are intentionally skipped.
_WORD_RE = re.compile(r"[\p{L}\p{M}]+", re.UNICODE)


class LatinTokenizer(Tokenizer):
    """Baseline Latin-script tokenizer with whitespace/punct preservation.

    MWE merging is left to language-specific implementations via a later pass.
    """

    def tokenize(self, text: str) -> List[Token]:
        out: List[Token] = []
        # Emit tokens only for word spans; punctuation/whitespace are excluded.
        for m in _WORD_RE.finditer(text):
            s, e = m.span()
            out.append(Token(text=m.group(0), start=s, end=e, is_word=True))
        return out
