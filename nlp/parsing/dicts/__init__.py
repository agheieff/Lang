from __future__ import annotations

from .provider import (
    DictionaryProviderChain,
    StarDictProvider,
)
from .cedict import CedictProvider

__all__ = [
    "DictionaryProviderChain",
    "StarDictProvider",
    "CedictProvider",
]
