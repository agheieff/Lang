from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


_CLITIC_COMBOS = (
    "melo","mela","melos","melas",
    "telo","tela","telos","telas",
    "selo","sela","selos","selas",
    "lelo","lela","lelos","lelas",
)
_CLITICS = ("nos","os","se","me","te","lo","la","los","las","le","les")


def _strip_clitics(word: str) -> Tuple[str, Optional[str]]:
    w = word.lower()
    for combo in sorted(_CLITIC_COMBOS, key=len, reverse=True):
        if w.endswith(combo):
            return w[: -len(combo)], combo
    for cl in sorted(_CLITICS, key=len, reverse=True):
        if w.endswith(cl):
            return w[: -len(cl)], cl
    return w, None


def _morph_guess(surface: str, lemma: Optional[str]) -> Tuple[Optional[str], Dict[str, str]]:
    s = surface.lower().strip()
    base, clitic = _strip_clitics(s)
    pos: Optional[str] = None
    morph: Dict[str, str] = {}

    def with_clitic(m: Dict[str, str]) -> Dict[str, str]:
        if clitic:
            m = dict(m)
            m["Clitic"] = clitic
        return m

    # Non-finite
    if base.endswith("ando") or base.endswith("iendo") or base.endswith("yendo"):
        return "VERB", with_clitic({"VerbForm": "Ger"})
    if base.endswith("ado") or base.endswith("ido") or base.endswith("to") or base.endswith("so") or base.endswith("cho"):
        return "VERB", with_clitic({"VerbForm": "Part"})
    if base.endswith("ar") or base.endswith("er") or base.endswith("ir"):
        return "VERB", with_clitic({"VerbForm": "Inf"})

    # Finite verb patterns (simplified)
    # Preterite -ar
    for end, feats in (
        ("é",  {"Person":"1","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("aste",{"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("ó",  {"Person":"3","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("amos",{"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Past"}),
        ("asteis",{"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Past"}),
        ("aron",{"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Past"}),
    ):
        if base.endswith(end):
            return "VERB", with_clitic(feats)
    # Preterite -er/-ir
    for end, feats in (
        ("í",  {"Person":"1","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("iste",{"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("ió", {"Person":"3","Number":"Sing","Mood":"Ind","Tense":"Past"}),
        ("imos",{"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Past"}),
        ("isteis",{"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Past"}),
        ("ieron",{"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Past"}),
    ):
        if base.endswith(end):
            return "VERB", with_clitic(feats)
    # Imperfect -ar
    for end, feats in (
        ("aba", {"Person":"1","Number":"Sing","Mood":"Ind","Tense":"Imp"}),
        ("abas", {"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Imp"}),
        ("ábamos", {"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
        ("abais", {"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
        ("aban", {"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
    ):
        if base.endswith(end):
            return "VERB", with_clitic(feats)
    # Imperfect -er/-ir
    for end, feats in (
        ("ía", {"Person":"1","Number":"Sing","Mood":"Ind","Tense":"Imp"}),
        ("ías", {"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Imp"}),
        ("íamos", {"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
        ("íais", {"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
        ("ían", {"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Imp"}),
    ):
        if base.endswith(end):
            return "VERB", with_clitic(feats)
    # Present endings; only label as VERB if lemma looks verbal
    is_verbal_lemma = bool(lemma and lemma.endswith(("ar","er","ir")))
    if is_verbal_lemma:
        for end, feats in (
            ("o",   {"Person":"1","Number":"Sing","Mood":"Ind","Tense":"Pres"}),
            ("as",  {"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Pres"}),
            ("a",   {"Person":"3","Number":"Sing","Mood":"Ind","Tense":"Pres"}),
            ("amos",{"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("áis", {"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("an",  {"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("es",  {"Person":"2","Number":"Sing","Mood":"Ind","Tense":"Pres"}),
            ("e",   {"Person":"3","Number":"Sing","Mood":"Ind","Tense":"Pres"}),
            ("emos",{"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("éis", {"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("en",  {"Person":"3","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("imos",{"Person":"1","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
            ("ís",  {"Person":"2","Number":"Plur","Mood":"Ind","Tense":"Pres"}),
        ):
            if base.endswith(end):
                return "VERB", with_clitic(feats)

    # Plural noun guess: avoid accented verb endings
    if any(base.endswith(e) for e in ("es","s")) and not any(ch in base for ch in "áéíóú"):
        return "NOUN", {"Number": "Plur"}

    return None, {}


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

    # 2) Try simplemma for lemma; enrich with lightweight morph guess
    try:
        from simplemma import lemmatize  # type: ignore
        lemma = lemmatize(surface, lang="es")
        pos, morph = _morph_guess(surface, lemma)
        return {"surface": surface, "lemma": lemma, "pos": pos, "morph": morph}
    except Exception:
        pass

    # 3) Fallback to simple rules
    return _simple_spanish_rules(surface)
