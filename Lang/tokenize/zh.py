from __future__ import annotations

from typing import List, Optional, Tuple

from .base import Token, Tokenizer
from pathlib import Path
import os

_CEDICT_WORDS: Optional[set[str]] = None
_CEDICT_MAXLEN: int = 6


def _load_cedict_words() -> None:
    global _CEDICT_WORDS, _CEDICT_MAXLEN
    if _CEDICT_WORDS is not None:
        return
    try:
        # Reuse cedict helper to locate file
        from Lang.parsing.dicts.cedict import _find_cedict  # type: ignore
    except Exception:
        _CEDICT_WORDS = set()
        _CEDICT_MAXLEN = 6
        return
    base = Path(os.getenv("ARCADIA_DICT_ROOT", Path.cwd() / "data" / "dicts")) / "zh-en"
    p = _find_cedict(base)
    words: set[str] = set()
    if p and p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line or line.startswith("#"):
                        continue
                    try:
                        head = line.split("/", 1)[0].strip()
                        parts = head.split()
                        if len(parts) >= 2:
                            trad, simp = parts[0], parts[1]
                            words.add(trad)
                            words.add(simp)
                    except Exception:
                        continue
        except Exception:
            pass
    _CEDICT_WORDS = words
    _CEDICT_MAXLEN = max((len(w) for w in words), default=6)


def _fmm_segment_han(text: str, start: int, end: int) -> List[Tuple[str, int, int]]:
    """Forward maximum matching on a Han run using CEDICT words.
    Returns list of (word, s, e) offsets.
    """
    _load_cedict_words()
    words = _CEDICT_WORDS or set()
    maxlen = _CEDICT_MAXLEN
    out: List[Tuple[str, int, int]] = []
    i = start
    while i < end:
        L = min(maxlen, end - i)
        found = None
        while L > 0:
            cand = text[i : i + L]
            if cand in words:
                found = cand
                break
            L -= 1
        if not found:
            found = text[i : i + 1]
            L = 1
        out.append((found, i, i + L))
        i += L
    return out


def _jieba_tokenize(text: str):
    try:
        import jieba
        from jieba import tokenize as jt
    except Exception:
        # Fallback: naive per-character segmentation
        for i, ch in enumerate(text):
            yield ch, i, i + 1
        return
    # ensure initialized
    try:
        # jieba.tokenize returns (word, start, end)
        for word, start, end in jt(text):
            yield word, start, end
    except Exception:
        for i, ch in enumerate(text):
            yield ch, i, i + 1


class ZhTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[Token]:
        # Use jieba.tokenize to get words with offsets; rely on its default model
        tokens: List[Token] = []
        raw: List[Tuple[str, int, int]] = [(w, s, e) for (w, s, e) in _jieba_tokenize(text)]
        if not raw:
            # per-character fallback
            return [Token(text=ch, start=i, end=i + 1, is_word=bool(ch.strip())) for i, ch in enumerate(text)]

        # If jieba collapsed into one massive token or mostly single chars, repair Han runs using FMM
        def is_han(ch: str) -> bool:
            try:
                import regex as re
                return bool(re.match(r"\p{Script=Han}$", ch))
            except Exception:
                return '\u4e00' <= ch <= '\u9fff'

        single_ratio = sum(1 for w, _, _ in raw if len(w) == 1) / max(len(raw), 1)
        use_fmm = single_ratio > 0.6 or any((e - s) >= int(0.6 * len(text)) and any(is_han(c) for c in w) for w, s, e in raw)

        if use_fmm:
            # Walk the text and apply FMM only on contiguous Han ranges
            i = 0
            L = len(text)
            while i < L:
                if is_han(text[i]):
                    j = i + 1
                    while j < L and is_han(text[j]):
                        j += 1
                    for w, s, e in _fmm_segment_han(text, i, j):
                        tokens.append(Token(text=w, start=s, end=e, is_word=True))
                    i = j
                else:
                    tokens.append(Token(text=text[i], start=i, end=i + 1, is_word=bool(text[i].strip())))
                    i += 1
            return tokens

        # Normal case: trust jieba tokenization
        for w, s, e in raw:
            tokens.append(Token(text=w, start=s, end=e, is_word=True))
        return tokens
        return out
