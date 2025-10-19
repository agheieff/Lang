from __future__ import annotations

import regex as re  # use the 'regex' module for Unicode \p{L}/\p{M}
from typing import List
from .base import Token, Tokenizer


_SEG_RE = re.compile(r"([\p{L}\p{M}]+)|([^\p{L}\p{M}]+)", re.UNICODE)


class LatinTokenizer(Tokenizer):
    """Baseline Latin-script tokenizer with whitespace/punct preservation.

    MWE merging is left to language-specific implementations via a later pass.
    """

    def tokenize(self, text: str) -> List[Token]:
        out: List[Token] = []
        for m in _SEG_RE.finditer(text):
            span = m.span()
            out.append(Token(text=m.group(0), start=span[0], end=span[1], is_word=bool(m.group(1))))
        return out
