"""Replay append-only reading evidence into derived lexeme estimates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from server.clock import as_utc, utc_now
from server.learning_units import LearningUnitIndex, learning_unit_indexes
from server.lesson_content import lesson_document
from server.models import Interaction, Lesson, LexemeState
from server.reading_evidence import (
    DEFAULT_READING_EVIDENCE_POLICY,
    ReadingEvidencePolicy,
    qualified_completion_and_considered,
)
from server.schemas import LessonDocument, LessonTerm
from server.spaced_repetition import (
    DEFAULT_STABILITY_POLICY,
    StabilityPolicy,
    estimated_recall,
    stability_after_failure,
    stability_after_weighted_success,
)


@dataclass(frozen=True)
class LexemeLearningPolicy:
    """All tunable evidence and scheduling weights for vocabulary."""

    qualification: ReadingEvidencePolicy = DEFAULT_READING_EVIDENCE_POLICY
    passive_success_weight: float = 1.5
    failure_weight: float = 1.0
    repeated_reveal_grades: tuple[float, ...] = (0.25, 0.6, 1.0)
    full_failure_due_days: float = 0.25
    weak_failure_due_days: float = 2.0
    stability_gain_days_per_success_mass: float = 0.5
    stability: StabilityPolicy = DEFAULT_STABILITY_POLICY

    def __post_init__(self) -> None:
        weights = (
            self.passive_success_weight,
            self.failure_weight,
            self.stability_gain_days_per_success_mass,
        )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("learning weights must be finite and non-negative")
        grades = self.repeated_reveal_grades
        if (
            not grades
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in grades)
            or tuple(sorted(grades)) != grades
        ):
            raise ValueError("repeated_reveal_grades must be ordered values between 0 and 1")
        if (
            not math.isfinite(self.full_failure_due_days)
            or not math.isfinite(self.weak_failure_due_days)
            or not 0 <= self.full_failure_due_days <= self.weak_failure_due_days
        ):
            raise ValueError("failure due-day bounds must be finite, non-negative, and ordered")


DEFAULT_LEXEME_LEARNING_POLICY = LexemeLearningPolicy()


@dataclass
class _LexemeEstimate:
    term: LessonTerm
    learning_language: str
    translation_language: str
    alpha: float = 2.0
    beta: float = 2.0
    stability_days: float = 0.5
    qualified_exposures: int = 0
    reveal_failures: float = 0.0
    lesson_ids: set[int] = field(default_factory=set)
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    last_revealed_at: datetime | None = None
    next_due_at: datetime | None = None

    @property
    def mastery(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass(frozen=True)
class _LearningEvidence:
    at: datetime
    order: int
    kind: Literal["failure", "exclude", "success"]
    learning_language: str
    translation_language: str
    lesson_id: int
    session_id: str
    term_key: str | None = None
    display_key: str | None = None


def rebuild_lexeme_states(
    db: Session,
    *,
    policy: LexemeLearningPolicy = DEFAULT_LEXEME_LEARNING_POLICY,
) -> list[LexemeState]:
    """Replace the derived lexeme cache with a deterministic evidence replay."""

    lessons = db.scalars(select(Lesson).order_by(Lesson.imported_at, Lesson.id)).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    interactions = db.scalars(
        select(Interaction).order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    estimates = _replay_lexeme_states(documents, interactions, policy)

    db.execute(delete(LexemeState))
    rows = [_state_row(key[2], state) for key, state in sorted(estimates.items())]
    db.add_all(rows)
    db.commit()
    return rows


def _replay_lexeme_states(
    documents: Mapping[int, LessonDocument],
    interactions: Sequence[Interaction],
    policy: LexemeLearningPolicy,
) -> dict[tuple[str, str, str], _LexemeEstimate]:
    indexes = learning_unit_indexes(documents.values())
    estimates = _initial_estimates(indexes)
    evidence = _ordered_evidence(documents, interactions, indexes, policy)

    failure_totals: dict[tuple[int, str, str], float] = defaultdict(float)
    exclusions: dict[tuple[int, str], set[str]] = defaultdict(set)
    successes: dict[tuple[int, str], set[str]] = defaultdict(set)
    for item in evidence:
        index = indexes[(item.learning_language, item.translation_language)]
        targets = _evidence_targets(item, index, estimates)
        if not targets:
            continue
        session = (item.lesson_id, item.session_id)
        if item.kind == "failure":
            weights = (
                _composite_failure_weights(targets, item, estimates)
                if item.display_key is not None
                else {targets[0]: 1.0}
            )
            exclusions[session].update(targets)
            for term_key, requested_weight in weights.items():
                cap_key = (*session, term_key)
                weight = min(requested_weight, 1.0 - failure_totals[cap_key])
                if weight <= 1e-12:
                    continue
                state = estimates[(item.learning_language, item.translation_language, term_key)]
                _apply_failure(state, item, weight, policy)
                failure_totals[cap_key] = round(failure_totals[cap_key] + weight, 12)
        elif item.kind == "exclude":
            exclusions[session].update(targets)
        else:
            for term_key in targets:
                if term_key in exclusions[session] or term_key in successes[session]:
                    continue
                state = estimates[(item.learning_language, item.translation_language, term_key)]
                _apply_success(state, item, policy)
                successes[session].add(term_key)
    return estimates


def _initial_estimates(
    indexes: Mapping[tuple[str, str], LearningUnitIndex],
) -> dict[tuple[str, str, str], _LexemeEstimate]:
    return {
        (*languages, term_key): _LexemeEstimate(
            term=term,
            learning_language=languages[0],
            translation_language=languages[1],
        )
        for languages, index in indexes.items()
        for term_key, term in index.learning_catalog().items()
    }


def _ordered_evidence(
    documents: Mapping[int, LessonDocument],
    interactions: Sequence[Interaction],
    indexes: Mapping[tuple[str, str], LearningUnitIndex],
    policy: LexemeLearningPolicy,
) -> list[_LearningEvidence]:
    sessions: dict[tuple[int, str], list[Interaction]] = defaultdict(list)
    for interaction in interactions:
        sessions[(interaction.lesson_id, interaction.session_id)].append(interaction)

    evidence: list[_LearningEvidence] = []
    for (lesson_id, _session_id), session_events in sessions.items():
        document = documents.get(lesson_id)
        if document is None:
            continue
        index = indexes[(document.learning_language, document.translation_language)]
        evidence.extend(_session_evidence(lesson_id, document, session_events, index, policy))
    evidence.sort(
        key=lambda item: (
            item.at,
            item.order,
            {"failure": 0, "exclude": 1, "success": 2}[item.kind],
            item.term_key or item.display_key or "",
        )
    )
    return evidence


def _evidence_targets(
    evidence: _LearningEvidence,
    index: LearningUnitIndex,
    estimates: Mapping[tuple[str, str, str], _LexemeEstimate],
) -> tuple[str, ...]:
    if evidence.term_key is not None:
        return (evidence.term_key,)
    if evidence.display_key is None:
        return ()
    candidates = index.component_candidates(evidence.display_key)
    establishment = {
        term_key: state.qualified_exposures + state.reveal_failures
        for item in candidates
        for term_key in item.canonical_keys
        if (
            state := estimates.get(
                (evidence.learning_language, evidence.translation_language, term_key)
            )
        )
        is not None
    }
    selected = index.select_evidence_targets(evidence.display_key, establishment)
    return tuple(dict.fromkeys(selected)) if len(set(selected)) >= 2 else ()


def _composite_failure_weights(
    targets: tuple[str, ...],
    evidence: _LearningEvidence,
    estimates: Mapping[tuple[str, str, str], _LexemeEstimate],
) -> dict[str, float]:
    selected = [
        estimates[(evidence.learning_language, evidence.translation_language, key)]
        for key in targets
    ]
    uncertainty = sum(min(1.0, 4.0 / (state.alpha + state.beta)) for state in selected) / len(
        selected
    )
    contrast = 1.0 + 2.0 * (1.0 - uncertainty)
    scores = [math.exp((1.0 - state.mastery) * contrast) for state in selected]
    total = sum(scores)
    return {key: score / total for key, score in zip(targets, scores, strict=True)}


def _session_evidence(
    lesson_id: int,
    document: LessonDocument,
    events: Sequence[Interaction],
    index: LearningUnitIndex,
    policy: LexemeLearningPolicy,
) -> list[_LearningEvidence]:
    revealed: dict[str, Interaction] = {}
    composite_reveals: dict[str, Interaction] = {}
    translated_sentences: dict[str, Interaction] = {}
    full_translation = False

    qualified_completion, considered = qualified_completion_and_considered(
        events, policy.qualification
    )
    for event in considered:
        if event.event_type == "term.revealed":
            term_key = event.payload.get("term_key")
            if isinstance(term_key, str):
                canonical = index.resolve(term_key)
                if canonical is not None:
                    revealed.setdefault(canonical, event)
                elif index.component_candidates(term_key):
                    composite_reveals.setdefault(term_key, event)
        elif event.event_type == "translation.revealed":
            if event.payload.get("scope") == "lesson":
                full_translation = True
            elif event.payload.get("scope") == "sentence" and isinstance(
                event.payload.get("sentence_key"), str
            ):
                translated_sentences.setdefault(event.payload["sentence_key"], event)

    result = [
        _term_evidence("failure", lesson_id, document, event, term_key=key)
        for key, event in revealed.items()
    ]
    result.extend(
        _term_evidence("failure", lesson_id, document, event, display_key=key)
        for key, event in composite_reveals.items()
    )
    if qualified_completion is None or full_translation:
        return result

    sentence_terms = document.sentence_terms()
    for sentence_key, event in translated_sentences.items():
        for display_key in sentence_terms.get(sentence_key, set()):
            canonical = index.resolve(display_key)
            if canonical is not None:
                result.append(
                    _term_evidence("exclude", lesson_id, document, event, term_key=canonical)
                )
            elif index.component_candidates(display_key):
                result.append(
                    _term_evidence("exclude", lesson_id, document, event, display_key=display_key)
                )

    passive_terms = sorted(
        {
            canonical
            for term_key in document.term_catalog()
            if (canonical := index.resolve(term_key)) is not None
        }
    )
    result.extend(
        _term_evidence(
            "success",
            lesson_id,
            document,
            qualified_completion,
            term_key=term_key,
        )
        for term_key in passive_terms
    )
    composite_terms = sorted(
        {
            display_key
            for display_key in document.term_catalog()
            if index.resolve(display_key) is None and index.component_candidates(display_key)
        }
    )
    result.extend(
        _term_evidence(
            "success",
            lesson_id,
            document,
            qualified_completion,
            display_key=display_key,
        )
        for display_key in composite_terms
    )
    return result


def _term_evidence(
    kind: Literal["failure", "exclude", "success"],
    lesson_id: int,
    document: LessonDocument,
    event: Interaction,
    *,
    term_key: str | None = None,
    display_key: str | None = None,
) -> _LearningEvidence:
    return _LearningEvidence(
        at=as_utc(event.occurred_at),
        order=event.id,
        kind=kind,
        learning_language=document.learning_language,
        translation_language=document.translation_language,
        lesson_id=lesson_id,
        session_id=event.session_id,
        term_key=term_key,
        display_key=display_key,
    )


def _apply_failure(
    state: _LexemeEstimate,
    evidence: _LearningEvidence,
    weight: float,
    policy: LexemeLearningPolicy,
) -> None:
    penalty = weight * _failure_grade(state.reveal_failures, policy)
    recall = estimated_recall(
        mastery=state.mastery,
        stability_days=state.stability_days,
        last_seen_at=state.last_seen_at,
        at=evidence.at,
        policy=policy.stability,
    )
    state.beta += policy.failure_weight * penalty
    state.reveal_failures = round(state.reveal_failures + weight, 12)
    state.stability_days = stability_after_failure(
        stability_days=state.stability_days,
        recall=recall,
        penalty=penalty,
        policy=policy.stability,
    )
    state.lesson_ids.add(evidence.lesson_id)
    state.first_seen_at = state.first_seen_at or evidence.at
    state.last_seen_at = evidence.at
    state.last_revealed_at = evidence.at
    due_days = (
        policy.weak_failure_due_days
        - (policy.weak_failure_due_days - policy.full_failure_due_days) * penalty
    )
    state.next_due_at = evidence.at + timedelta(days=due_days)


def _failure_grade(failures: float, policy: LexemeLearningPolicy) -> float:
    grades = policy.repeated_reveal_grades
    lower = min(math.floor(failures), len(grades) - 1)
    upper = min(lower + 1, len(grades) - 1)
    fraction = failures - math.floor(failures)
    return grades[lower] + fraction * (grades[upper] - grades[lower])


def _apply_success(
    state: _LexemeEstimate,
    evidence: _LearningEvidence,
    policy: LexemeLearningPolicy,
) -> None:
    recall = (
        1.0
        if state.last_seen_at is None
        else estimated_recall(
            mastery=state.mastery,
            stability_days=state.stability_days,
            last_seen_at=state.last_seen_at,
            at=evidence.at,
            policy=policy.stability,
        )
    )
    state.stability_days = stability_after_weighted_success(
        stability_days=state.stability_days,
        recall=recall,
        evidence_mass=policy.passive_success_weight,
        gain_days_per_mass=policy.stability_gain_days_per_success_mass,
        policy=policy.stability,
    )
    state.alpha += policy.passive_success_weight
    state.qualified_exposures += 1
    state.lesson_ids.add(evidence.lesson_id)
    state.first_seen_at = state.first_seen_at or evidence.at
    state.last_seen_at = evidence.at
    state.next_due_at = evidence.at + timedelta(days=state.stability_days)


def _state_row(term_key: str, state: _LexemeEstimate) -> LexemeState:
    return LexemeState(
        learning_language=state.learning_language,
        translation_language=state.translation_language,
        term_key=term_key,
        lemma=state.term.lemma,
        pos=state.term.pos,
        gloss=state.term.gloss,
        pronunciation=state.term.pronunciation,
        frequency_rank=state.term.frequency_rank,
        alpha=state.alpha,
        beta=state.beta,
        stability_days=state.stability_days,
        qualified_exposures=state.qualified_exposures,
        reveal_failures=state.reveal_failures,
        distinct_lessons=len(state.lesson_ids),
        first_seen_at=state.first_seen_at,
        last_seen_at=state.last_seen_at,
        last_revealed_at=state.last_revealed_at,
        next_due_at=state.next_due_at,
        updated_at=utc_now(),
    )
