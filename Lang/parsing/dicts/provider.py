from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional
from collections import OrderedDict


class DictionaryProvider:
    def translations(self, src: str, tgt: str, lemma: str) -> List[str]:
        raise NotImplementedError


class StarDictProvider(DictionaryProvider):
    """Reads StarDict dictionaries from data/dicts/{src}-{tgt}.

    Expects .dict/.idx/.ifo files. Requires pystardict; if not available or no files found,
    provider yields no translations.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        env_root = os.getenv("ARCADIA_DICT_ROOT")
        if env_root:
            self.root = Path(env_root)
        else:
            # Resolve relative to project root regardless of current working dir
            base = Path(__file__).resolve().parents[3] / "data" / "dicts"
            self.root = root or base
        self._ok = False
        try:
            import pystardict  # noqa: F401
            self._ok = True
        except Exception:
            self._ok = False

    def _dict_paths(self, src: str, tgt: str) -> List[Path]:
        pair = f"{src}-{tgt}"
        d = self.root / pair
        if not d.exists() or not d.is_dir():
            return []
        # Search recursively for .ifo files; that reliably yields the base
        bases: List[Path] = []
        for p in d.rglob("*.ifo"):
            bases.append(p.with_suffix(""))
        # Fallback: discover by other files if .ifo not present
        if not bases:
            seen = set()
            for p in d.rglob("*"):
                if p.suffix in {".dict", ".idx", ".ifo"} or p.name.endswith(".dict.dz") or p.name.endswith(".idx.gz"):
                    name = p.name
                    if name.endswith('.dict.dz'):
                        base = p.parent / name[:-9]
                    elif name.endswith('.idx.gz'):
                        base = p.parent / name[:-7]
                    else:
                        base = p.with_suffix("")
                    if str(base) not in seen:
                        seen.add(str(base))
                        bases.append(base)
        return bases

    def translations(self, src: str, tgt: str, lemma: str) -> List[str]:
        if not self._ok:
            return []
        try:
            from pystardict import Dictionary
        except Exception:
            return []
        results: List[str] = []
        for base in self._dict_paths(src, tgt):
            try:
                d = Dictionary(str(base))
                if lemma in d:
                    raw = d[lemma]
                    # Prefer extracting <li> entries; fall back to text cleanup
                    items = re.findall(r"<li[^>]*>(.*?)</li>", raw, flags=re.I | re.S)
                    extracted: List[str] = []
                    def strip_tags(s: str) -> str:
                        s = re.sub(r"<[^>]+>", " ", s)
                        s = re.sub(r"\s+", " ", s)
                        return s.strip()
                    if items:
                        for it in items:
                            txt = strip_tags(it)
                            if txt:
                                extracted.append(txt)
                    else:
                        txt = strip_tags(raw)
                        # split by common separators
                        tmp: List[str] = []
                        for sep in [";", "\n", "/", "·", "|"]:
                            parts = []
                            for chunk in (tmp or [txt]):
                                parts.extend([p.strip() for p in chunk.split(sep)])
                            tmp = [p for p in parts if p]
                        extracted = tmp or ([txt] if txt else [])
                    for p in extracted:
                        if p and p not in results:
                            results.append(p)
            except Exception:
                continue
        return results


class DictionaryProviderChain(DictionaryProvider):
    def __init__(self, providers: Optional[List[DictionaryProvider]] = None, cache_limit: int = 5000) -> None:
        self.providers = providers or [StarDictProvider()]
        self._cache: OrderedDict[tuple[str, str, str], List[str]] = OrderedDict()
        self._cache_limit = cache_limit

    def translations(self, src: str, tgt: str, lemma: str) -> List[str]:
        key = (src, tgt, lemma)
        if key in self._cache:
            # LRU: move to end
            val = self._cache.pop(key)
            self._cache[key] = val
            return val
        for p in self.providers:
            vals = p.translations(src, tgt, lemma)
            if vals:
                self._maybe_prune()
                self._cache[key] = vals
                return vals
        self._maybe_prune()
        self._cache[key] = []
        return []

    def _maybe_prune(self) -> None:
        while len(self._cache) > self._cache_limit:
            # pop least recently used
            self._cache.popitem(last=False)
