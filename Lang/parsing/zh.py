from __future__ import annotations

from typing import Any, Dict, Optional


def _pos_flag(surface: str) -> Optional[str]:
    try:
        import jieba.posseg as pseg  # type: ignore
        words = list(pseg.cut(surface))
        if not words:
            return None
        return words[0].flag
    except Exception:
        return None


_FLAG2POS = {
    # Common jieba flags mapping (coarse)
    "n": "NOUN", "nr": "PROPN", "ns": "PROPN", "nt": "PROPN", "nz": "NOUN",
    "v": "VERB", "vd": "VERB", "vn": "VERB",
    "a": "ADJ", "ad": "ADJ", "an": "ADJ",
    "d": "ADV", "r": "PRON", "p": "ADP", "m": "NUM",
}


def _coarse_pos(flag: Optional[str]) -> Optional[str]:
    if not flag:
        return None
    # take base class (first char) if exact not found
    return _FLAG2POS.get(flag, _FLAG2POS.get(flag[:1], None))


def analyze_word_zh(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
    s = surface.strip()
    if not s:
        return {"surface": surface, "lemma": surface, "pos": None, "morph": {}, "script": None, "pronunciation": None}
    # Script normalization
    try:
        from opencc import OpenCC  # type: ignore
        cc_t2s = OpenCC("t2s")
        cc_s2t = OpenCC("s2t")
        simp = cc_t2s.convert(s)
        trad = cc_s2t.convert(s)
        script = "Hant" if s == trad and s != simp else "Hans"
    except Exception:
        simp = s
        script = None
    # POS
    pos = _coarse_pos(_pos_flag(s))
    # Pinyin
    try:
        from pypinyin import lazy_pinyin, Style  # type: ignore
        syllables_mark = lazy_pinyin(s, style=Style.TONE)
        pron = {"pinyin": " ".join(syllables_mark)}
    except Exception:
        pron = None
    return {
        "surface": s,
        "lemma": simp,
        "pos": pos,
        "morph": {},
        "script": script,
        "pronunciation": pron,
    }
