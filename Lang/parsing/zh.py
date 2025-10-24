from __future__ import annotations

from typing import Any, Dict, Optional


_OPENCC_T2S: Optional[Any] = None
_OPENCC_S2T: Optional[Any] = None
_OPENCC_INIT_FAILED = False


def _opencc_converters() -> tuple[Optional[Any], Optional[Any]]:
    global _OPENCC_T2S, _OPENCC_S2T, _OPENCC_INIT_FAILED
    if _OPENCC_INIT_FAILED:
        return None, None
    if _OPENCC_T2S is None or _OPENCC_S2T is None:
        try:
            from opencc import OpenCC  # type: ignore
            _OPENCC_T2S = OpenCC("t2s")
            _OPENCC_S2T = OpenCC("s2t")
        except Exception:
            _OPENCC_INIT_FAILED = True
            _OPENCC_T2S = None
            _OPENCC_S2T = None
    return _OPENCC_T2S, _OPENCC_S2T


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
    cc_t2s, cc_s2t = _opencc_converters()
    if cc_t2s is not None and cc_s2t is not None:
        simp = cc_t2s.convert(s)
        trad = cc_s2t.convert(s)
        script = "Hant" if s == trad and s != simp else "Hans"
    else:
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
