"""Conservative use of Han recognition when choosing Chinese vocabulary offers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime

from server.clock import as_utc, optional_utc
from server.han import distinct_han_characters
from server.models import CharacterState, LexemeState
from server.spaced_repetition import estimated_recall

CHARACTER_SUPPORT_WEIGHT = 0.15
CHARACTER_GEOMETRIC_WEIGHT = 0.7
CHARACTER_FLOOR = 0.05
DIRECT_WORD_EVIDENCE_TO_FADE = 3.0
MINIMUM_OFFER_MULTIPLIER = 0.85
MAXIMUM_OFFER_MULTIPLIER = 1.15


def character_retrievability(state: CharacterState, *, at: datetime) -> float:
    """Return posterior recognition after time decay, or zero before evidence exists."""

    last_evidence_at = optional_utc(state.last_evidence_at)
    if state.first_evidence_at is None or last_evidence_at is None:
        return 0.0
    retention = estimated_recall(
        mastery=state.mastery,
        stability_days=state.stability_days,
        last_seen_at=last_evidence_at,
        at=as_utc(at),
    )
    return max(0.0, min(1.0, state.mastery * retention))


def word_character_support(
    term: LexemeState,
    character_states: Mapping[str, CharacterState],
    *,
    at: datetime,
) -> float | None:
    """Aggregate distinct Han identities; ``None`` means the lemma contains no Han."""

    characters = distinct_han_characters(term.lemma)
    if not characters:
        return None
    recalls = [
        character_retrievability(state, at=at)
        if (state := character_states.get(character)) is not None
        else 0.0
        for character in characters
    ]
    geometric = math.exp(
        sum(math.log(max(CHARACTER_FLOOR, recall)) for recall in recalls) / len(recalls)
    )
    return CHARACTER_GEOMETRIC_WEIGHT * geometric + (1.0 - CHARACTER_GEOMETRIC_WEIGHT) * min(
        recalls
    )


def character_offer_multiplier(
    term: LexemeState,
    character_states: Mapping[str, CharacterState],
    *,
    at: datetime,
) -> float:
    """Slightly prefer readable new words; direct word evidence quickly takes control."""

    support = word_character_support(term, character_states, at=at)
    if support is None:
        return 1.0
    direct_word_evidence = term.qualified_exposures + term.reveal_failures
    fade = max(0.0, 1.0 - direct_word_evidence / DIRECT_WORD_EVIDENCE_TO_FADE)
    multiplier = 1.0 + CHARACTER_SUPPORT_WEIGHT * fade * (2.0 * support - 1.0)
    return max(MINIMUM_OFFER_MULTIPLIER, min(MAXIMUM_OFFER_MULTIPLIER, multiplier))


def character_adjusted_offer_score(
    urgency: float,
    term: LexemeState,
    character_states: Mapping[str, CharacterState],
    *,
    at: datetime,
) -> float:
    """Apply character support to ordering only, leaving serialized urgency untouched."""

    return urgency * character_offer_multiplier(term, character_states, at=at)
