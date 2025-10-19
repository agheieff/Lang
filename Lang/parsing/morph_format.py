from __future__ import annotations

from typing import Dict, Optional


def _pn(morph: Dict[str, str]) -> Optional[str]:
     p = morph.get("Person")
     n = morph.get("Number")
     if not p or not n:
         return None
     n2 = "sg" if n.lower().startswith("sing") else "pl" if n.lower().startswith("plur") else None
     if not n2:
         return None
     return f"{p}{n2}"


def _mood(morph: Dict[str, str]) -> Optional[str]:
     mood = morph.get("Mood")
     if not mood:
         return None
     mapping = {
         "Ind": "ind",
         "Sub": "subj",
         "Imp": "imp",
         "Cnd": "cond",
     }
     return mapping.get(mood, mood.lower())


def _tense(morph: Dict[str, str]) -> Optional[str]:
     tense = morph.get("Tense")
     if not tense:
         return None
     mapping = {
         "Pres": "pres",
         # UD often uses Past; we map to pret by default for Spanish finite indicative
         "Past": "pret",
         "Imp": "impf",
         "Fut": "fut",
         "Pqp": "plup",
     }
     return mapping.get(tense, tense.lower())


def _gender_number(morph: Dict[str, str]) -> Optional[str]:
     g = morph.get("Gender")
     n = morph.get("Number")
     if not g and not n:
         return None
     g2 = None
     if g:
         g2 = {"Masc": "m", "Fem": "f", "Com": "c", "Neut": "n"}.get(g, g.lower()[0])
     n2 = None
     if n:
         n2 = "sg" if n.lower().startswith("sing") else "pl" if n.lower().startswith("plur") else n.lower()
     parts = [p for p in [n2, g2] if p]
     return " ".join(parts) if parts else None


def format_morph_label(pos: Optional[str], morph: Dict[str, str]) -> Optional[str]:
     """Format a compact human-readable label from UD-style morph features.

     Examples:
       VERB: Person=3 Number=Sing Mood=Ind Tense=Past -> "3sg ind pret"
       NOUN: Gender=Fem Number=Plur -> "pl f"
     """
     if not morph:
         return None
     if pos and pos.upper() in {"VERB", "AUX"}:
         pn = _pn(morph)
         mood = _mood(morph)
         tense = _tense(morph)
         parts = [p for p in [pn, mood, tense] if p]
         return " ".join(parts) if parts else None
     # Nouns/adjectives
     gn = _gender_number(morph)
     return gn
