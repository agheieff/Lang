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
    """Analyze a single Spanish word using external lemmatizers when available.

    Preference order:
    1) spaCy es_core_news_* model (best POS+morph+lemma)
    2) simplemma (dictionary-based lemma only)
    3) very light heuristics as a last resort
    """
    # 1) Try spaCy with Spanish model
    try:
        import spacy  # type: ignore
        for model in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
            try:
                nlp = spacy.load(model)
                break
            except Exception:
                nlp = None  # type: ignore
        if nlp is not None:  # type: ignore
            doc = nlp(surface)
            token = doc[0] if len(doc) else None
            if token:
                morph: Dict[str, str] = {}
                for feat in ["Person", "Number", "Mood", "Tense", "Gender"]:
                    val = token.morph.get(feat)
                    if val:
                        morph[feat] = val[0]
                pos_out = token.pos_
                if pos_out == "PROPN" and surface.islower():
                    pos_out = "NOUN"
                return {
                    "surface": surface,
                    "lemma": token.lemma_,
                    "pos": pos_out,
                    "morph": morph,
                }
    except Exception:
        pass

    # 2) Try simplemma dictionary-based lemmatizer
    try:
        from simplemma import lemmatize  # type: ignore
        lemma = lemmatize(surface, lang="es")
        return {"surface": surface, "lemma": lemma, "pos": None, "morph": {}}
    except Exception:
        pass

    # 3) Fallback to simple rules
    return _simple_spanish_rules(surface)
