from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .provider import DictionaryProvider


def _find_cedict(root: Path) -> Optional[Path]:
    for name in ("cedict_ts.u8", "CEDICT.TXT", "cedict_ts.txt"):
        p = root / name
        if p.exists():
            return p
    # search recursively
    for p in root.rglob("cedict_ts.u8"):
        return p
    return None


def _load_cedict(path: Path) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                # Format: Trad Simpl [pinyin] /def1/def2/.../
                try:
                    head, defs = line.strip().split(" /", 1)
                    parts = head.split(" ")
                    if len(parts) < 3:
                        continue
                    trad, simp = parts[0], parts[1]
                    glosses = [d.strip().strip("/") for d in defs.split("/") if d.strip()]
                    for key in (trad, simp):
                        lst = idx.setdefault(key, [])
                        for g in glosses:
                            if g not in lst:
                                lst.append(g)
                except Exception:
                    continue
    except Exception:
        return {}
    return idx


class CedictProvider(DictionaryProvider):
    def __init__(self, root: Optional[Path] = None) -> None:
        import os
        base = Path(os.getenv("ARCADIA_DICT_ROOT", Path.cwd() / "data" / "dicts"))
        d = (root or base) / "zh-en"
        p = _find_cedict(d)
        self._ok = False
        self._idx: Dict[str, List[str]] = {}
        if p and p.exists():
            self._idx = _load_cedict(p)
            self._ok = bool(self._idx)

    def translations(self, src: str, tgt: str, lemma: str) -> List[str]:
        if not self._ok or not src.startswith("zh") or tgt not in ("en", "eng"):
            return []
        vals = self._idx.get(lemma)
        if vals:
            return vals
        # try script conversion
        try:
            from opencc import OpenCC  # type: ignore
            cc_t2s = OpenCC("t2s")
            cc_s2t = OpenCC("s2t")
            simp = cc_t2s.convert(lemma)
            trad = cc_s2t.convert(lemma)
            return self._idx.get(simp) or self._idx.get(trad) or []
        except Exception:
            return []
