"""Frequency-ordered vocabulary lists used to propose new words."""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
# Base language tag -> list file; BCP-47 variants use their base language's list.
LISTS = {"zh": ("zh-hsk.tsv", "hsk")}


@dataclass(frozen=True)
class WordListEntry:
    lemma: str
    pronunciation: str | None
    gloss: str
    frequency_rank: int | None
    level: int | None
    source: str


def normalized_lemma(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def word_list(language_tag: str) -> tuple[WordListEntry, ...]:
    base = language_tag.strip().replace("_", "-").split("-")[0].casefold()
    return _load(base)


@lru_cache(maxsize=8)
def _load(base: str) -> tuple[WordListEntry, ...]:
    configured = LISTS.get(base)
    if configured is None:
        return ()
    filename, source = configured
    with (DIRECTORY / filename).open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        return tuple(
            WordListEntry(
                lemma=unicodedata.normalize("NFC", row["lemma"]),
                pronunciation=row["pronunciation"] or None,
                gloss=row["gloss"],
                frequency_rank=int(row["frequency_rank"]) if row["frequency_rank"] else None,
                level=int(row["level"]) if row["level"] else None,
                source=source,
            )
            for row in rows
        )
