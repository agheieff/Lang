from __future__ import annotations

from typing import Any, Dict, Optional


def _simple_spanish_rules(surface: str) -> Dict[str, Any]:
    """Very light heuristics as a fallback if NLP libs are not installed.

    - Lowercase; keep diacritics as-is.
    - Attempt to detect finite verb endings for -ar/-er/-ir verbs and propose lemma.
    - Mark pos as VERB when heuristic hits; otherwise None.
    - This is intentionally simplistic for MVP and will be replaced by spaCy later.
    """
    s = surface.strip()
    lower = s.lower()
    lemma: Optional[str] = None
    pos: Optional[str] = None
    morph: Dict[str, str] = {}

    # Heuristic: preterite 3sg -ó for -ar verbs (e.g., habló -> hablar)
    if lower.endswith("ó"):
        stem = lower[:-1]
        lemma = stem + "ar"
        pos = "VERB"
        morph = {"Person": "3", "Number": "Sing", "Mood": "Ind", "Tense": "Past"}

    # Present 1sg -o (hablo -> hablar), -er/-ir could also produce -o, but we default to -ar
    elif lower.endswith("o") and len(lower) > 2:
        stem = lower[:-1]
        lemma = stem + "ar"
        pos = "VERB"
        morph = {"Person": "1", "Number": "Sing", "Mood": "Ind", "Tense": "Pres"}

    # Plural nouns ending -s/-es (casas -> casa), handle feminine 'madres' -> 'madre'
    elif lower.endswith("es") and len(lower) > 3:
        stem = lower[:-2]
        lemma = stem
        pos = "NOUN"
        morph = {"Number": "Plur"}
    elif lower.endswith("s") and len(lower) > 2:
        lemma = lower[:-1]
        pos = "NOUN"
        morph = {"Number": "Plur"}

    return {
        "surface": s,
        "lemma": lemma,
        "pos": pos,
        "morph": morph,
    }


def analyze_word_es(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a single Spanish word. Returns dict with lemma, pos, and morph.

    If spaCy is available (es_core_news_sm), use it; otherwise, fall back to simple rules.
    """
    try:
        import spacy  # type: ignore
        try:
            nlp = spacy.load("es_core_news_sm")
        except Exception:
            # If model isn't installed, fall back
            return _simple_spanish_rules(surface)
        doc = nlp(surface)
        token = doc[0] if len(doc) else None
        if not token:
            return _simple_spanish_rules(surface)
        morph = {}
        for feat in ["Person", "Number", "Mood", "Tense", "Gender"]:
            val = token.morph.get(feat)
            if val:
                morph[feat] = val[0]
        pos_out = token.pos_
        # Treat lowercase proper nouns as common nouns in Spanish (heuristic)
        if pos_out == "PROPN" and surface.islower():
            pos_out = "NOUN"
        return {
            "surface": surface,
            "lemma": token.lemma_,
            "pos": pos_out,
            "morph": morph,
        }
    except Exception:
        # spaCy not installed; use simple rules
        return _simple_spanish_rules(surface)
