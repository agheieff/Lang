from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    is_word: bool
    is_mwe: bool = False


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> List[Token]:
        ...
