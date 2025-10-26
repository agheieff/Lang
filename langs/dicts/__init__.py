from __future__ import annotations

from Lang.parsing.dicts.provider import (
    DictionaryProviderChain,
    StarDictProvider,
)
from Lang.parsing.dicts.cedict import CedictProvider

__all__ = [
    "DictionaryProviderChain",
    "StarDictProvider",
    "CedictProvider",
]
