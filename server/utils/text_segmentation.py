from __future__ import annotations

"""
Shared text segmentation utilities.
"""

from typing import List, Tuple
import re


def split_sentences(text: str, lang: str) -> List[Tuple[int, int, str]]:
    """Split text into sentences, returning (start, end, segment) tuples.

    Uses language-aware regex for Chinese vs. others.
    """
    if not text:
        return []
    if str(lang).startswith("zh"):
        pattern = r"[^。！？!?…]+(?:[。！？!?…]+|$)"
    else:
        pattern = r"[^\.!?]+(?:[\.!?]+|$)"
    out: List[Tuple[int, int, str]] = []
    for m in re.finditer(pattern, text):
        s, e = m.span()
        seg = text[s:e]
        if seg and seg.strip():
            out.append((s, e, seg))
    return out


def split_paragraphs(text: str) -> List[Tuple[int, int, str]]:
    """Split text into paragraphs separated by blank lines (\n\n+)."""
    if not text:
        return []
    out: List[Tuple[int, int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        # skip leading newlines
        while i < n and text[i] == "\n":
            i += 1
        if i >= n:
            break
        s = i
        # paragraph ends at next blank line boundary
        while i < n - 1 and not (text[i] == "\n" and text[i + 1] == "\n"):
            i += 1
        # include until end of line
        while i < n and text[i] != "\n":
            i += 1
        e = min(n, i)
        seg = text[s:e]
        if seg and seg.strip():
            out.append((s, e, seg))
    return out
