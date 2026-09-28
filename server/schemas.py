"""Validated API and lesson contracts for the local reader."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, get_args
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from server.clock import utc_now
from server.grammar_catalog import grammar_catalog, grammar_construction

LESSON_SCHEMA_VERSION: Literal[1] = 1
CefrLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2"]
LevelSource = Literal["unknown", "self_reported", "estimated"]
CalibrationStatus = Literal["unstarted", "collecting", "rough", "stable"]
TermKnowledgeBand = Literal["familiar", "expected", "uncertain", "focus", "incidental"]
FeedbackTag = Literal[
    "shorter",
    "longer",
    "easier",
    "more_challenging",
    "more_grammar",
    "less_grammar",
    "same_topic",
    "new_topic",
]
GenerationTaskState = Literal["pending", "running", "completed", "failed"]
GenerationRequestKind = Literal["queue_fill", "topic_request"]
TextStatus = Literal["queued", "in_progress", "read", "skipped"]
EventType = Literal[
    "lesson.started",
    "term.revealed",
    "translation.revealed",
    "lesson.completed",
    "lesson.rated",
]

_PROPER_NOUN_POS = frozenset({"proper noun", "propernoun", "propn", "name"})


def is_proper_noun_pos(value: str) -> bool:
    normalized = "".join(character if character.isalnum() else " " for character in value)
    return " ".join(normalized.casefold().split()) in _PROPER_NOUN_POS


def _normalize_language_tag(value: str) -> str:
    parts = value.strip().split("-")
    normalized = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            normalized.append(part.upper())
        else:
            normalized.append(part.lower())
    return "-".join(normalized)


LanguageTag = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$"),
    AfterValidator(_normalize_language_tag),
]
LANGUAGE_TAG_DESCRIPTION = (
    "BCP-47-like language tag; examples: de-DE, es-ES, es-419, zh-Hans, zh-Hant."
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LessonTerm(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    lemma: str = Field(min_length=1, max_length=200)
    pos: str = Field(min_length=1, max_length=40)
    gloss: str = Field(min_length=1, max_length=500)
    pronunciation: str | None = Field(default=None, max_length=300)
    frequency_rank: int | None = Field(default=None, ge=1)

    @field_validator("key", "lemma", "pos", "gloss", "pronunciation")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LessonRun(StrictModel):
    text: str = Field(min_length=1)
    term: LessonTerm | None = None
    pronunciation: str | None = Field(default=None, max_length=300)

    @field_validator("pronunciation")
    @classmethod
    def strip_pronunciation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_term_surface(self) -> LessonRun:
        if self.term is not None and not self.text.strip():
            raise ValueError("a term run must contain visible text")
        return self


class GrammarOccurrence(StrictModel):
    """One contextual use of a cataloged construction, anchored to whole lesson runs."""

    key: str = Field(
        min_length=1,
        max_length=200,
        description="Stable lesson-unique identity for this occurrence.",
    )
    construction_key: str = Field(
        min_length=1,
        max_length=100,
        description="Stable key from the tracked grammar catalog.",
    )
    run_start: int = Field(
        ge=0,
        description="Inclusive zero-based index of the first covered sentence run.",
    )
    run_end: int = Field(
        ge=1,
        description="Exclusive zero-based index after the last covered sentence run.",
    )
    note: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional contextual explanation of this use; omit when the catalog description "
            "already suffices."
        ),
    )

    @field_validator("key", "construction_key")
    @classmethod
    def strip_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("note")
    @classmethod
    def strip_optional_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("construction_key")
    @classmethod
    def require_catalog_membership(cls, value: str) -> str:
        try:
            grammar_construction(value)
        except LookupError as error:
            raise ValueError(str(error)) from error
        return value

    @model_validator(mode="after")
    def validate_run_range(self) -> GrammarOccurrence:
        if self.run_end <= self.run_start:
            raise ValueError("grammar occurrence run_end must be greater than run_start")
        return self


class LessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[LessonRun] = Field(min_length=1)
    translation: str | None = None
    grammar: list[GrammarOccurrence] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Optional grammar occurrences anchored to half-open ranges of this sentence's runs."
        ),
    )

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_grammar_ranges(self) -> LessonSentence:
        occurrence_keys = [occurrence.key for occurrence in self.grammar]
        if len(occurrence_keys) != len(set(occurrence_keys)):
            raise ValueError("grammar occurrence keys must be unique within a sentence")
        for occurrence in self.grammar:
            if occurrence.run_end > len(self.runs):
                raise ValueError(
                    f"grammar occurrence range exceeds sentence runs: {occurrence.key}"
                )
        return self

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


class LessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[LessonSentence] = Field(min_length=1)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


def _collect_sentence_terms(
    sentence: LessonSentence, terms: dict[str, LessonTerm], *, generated: bool = False
) -> None:
    for run in sentence.runs:
        if run.term is None:
            continue
        previous = terms.get(run.term.key)
        if previous is not None and previous != run.term:
            qualifier = " generated" if generated else ""
            raise ValueError(f"term key has conflicting{qualifier} definitions: {run.term.key}")
        terms[run.term.key] = run.term


class CalibrationProbe(StrictModel):
    term_key: str = Field(min_length=1, max_length=200)
    difficulty: float = Field(ge=0.0, le=1.0)

    @field_validator("term_key")
    @classmethod
    def strip_term_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LessonCalibration(StrictModel):
    sequence: int = Field(ge=1)
    probes: list[CalibrationProbe] = Field(min_length=8, max_length=20)

    @model_validator(mode="after")
    def validate_unique_probes(self) -> LessonCalibration:
        keys = [probe.term_key for probe in self.probes]
        if len(keys) != len(set(keys)):
            raise ValueError("calibration probe term keys must be unique")
        return self


class LessonDocument(StrictModel):
    schema_version: Literal[1] = LESSON_SCHEMA_VERSION
    key: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    title_sentence: LessonSentence | None = None
    learning_language: LanguageTag = Field(description=LANGUAGE_TAG_DESCRIPTION)
    translation_language: LanguageTag = Field(description=LANGUAGE_TAG_DESCRIPTION)
    topic: str | None = Field(default=None, max_length=200)
    level: CefrLevel = "A1"
    difficulty: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Fine-grained difficulty within and across CEFR bands.",
    )
    blocks: list[LessonBlock] = Field(min_length=1)
    target_term_keys: list[str] = Field(default_factory=list)
    calibration: LessonCalibration | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "key",
        "title",
        "topic",
    )
    @classmethod
    def strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_structure(self) -> LessonDocument:
        block_keys: set[str] = set()
        sentence_keys: set[str] = set()
        terms: dict[str, LessonTerm] = {}

        if self.title_sentence is not None:
            if self.title_sentence.text != self.title:
                raise ValueError("title_sentence text must exactly equal title")
            sentence_keys.add(self.title_sentence.key)
            _collect_sentence_terms(self.title_sentence, terms)

        for block in self.blocks:
            if block.key in block_keys:
                raise ValueError(f"duplicate block key: {block.key}")
            block_keys.add(block.key)

            for sentence in block.sentences:
                if sentence.key in sentence_keys:
                    raise ValueError(f"duplicate sentence key: {sentence.key}")
                sentence_keys.add(sentence.key)
                _collect_sentence_terms(sentence, terms)

        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("target_term_keys must be unique")
        unknown_targets = set(self.target_term_keys) - terms.keys()
        if unknown_targets:
            unknown = ", ".join(sorted(unknown_targets))
            raise ValueError(f"target_term_keys contains unknown terms: {unknown}")

        _validate_calibration_structure(self.calibration, self.blocks, self.title_sentence)
        _validate_grammar_structure(
            self.learning_language,
            self.blocks,
            self.title_sentence,
        )

        return self

    def term_catalog(self) -> dict[str, LessonTerm]:
        terms = {
            run.term.key: run.term
            for block in self.blocks
            for sentence in block.sentences
            for run in sentence.runs
            if run.term is not None
        }
        if self.title_sentence is not None:
            terms.update(
                {run.term.key: run.term for run in self.title_sentence.runs if run.term is not None}
            )
        return terms

    def sentence_terms(self) -> dict[str, set[str]]:
        terms = {
            sentence.key: {run.term.key for run in sentence.runs if run.term is not None}
            for block in self.blocks
            for sentence in block.sentences
        }
        if self.title_sentence is not None:
            terms[self.title_sentence.key] = {
                run.term.key for run in self.title_sentence.runs if run.term is not None
            }
        return terms

    def calibration_probe_sentences(self) -> dict[str, str]:
        if self.calibration is None:
            return {}
        probe_keys = {probe.term_key for probe in self.calibration.probes}
        sentences = {
            run.term.key: sentence.key
            for block in self.blocks
            for sentence in block.sentences
            for run in sentence.runs
            if run.term is not None and run.term.key in probe_keys
        }
        if self.title_sentence is not None:
            sentences.update(
                {
                    run.term.key: self.title_sentence.key
                    for run in self.title_sentence.runs
                    if run.term is not None and run.term.key in probe_keys
                }
            )
        return sentences

    def grammar_occurrences(self) -> dict[str, GrammarOccurrence]:
        return {
            occurrence.key: occurrence
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }

    def grammar_occurrence_sentences(self) -> dict[str, str]:
        return {
            occurrence.key: sentence.key
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }

    def grammar_construction_keys(self) -> set[str]:
        return {
            occurrence.construction_key
            for sentence in _lesson_sentences(self.blocks, self.title_sentence)
            for occurrence in sentence.grammar
        }


def _validate_calibration_structure(
    calibration: LessonCalibration | None,
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> None:
    if calibration is None:
        return
    occurrences: dict[str, int] = {}
    sentences = [
        *(sentence for block in blocks for sentence in block.sentences),
        *([title_sentence] if title_sentence is not None else []),
    ]
    for sentence in sentences:
        for run in sentence.runs:
            if run.term is not None:
                occurrences[run.term.key] = occurrences.get(run.term.key, 0) + 1
    unknown = [probe.term_key for probe in calibration.probes if probe.term_key not in occurrences]
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"calibration contains unknown probe terms: {names}")
    repeated = [probe.term_key for probe in calibration.probes if occurrences[probe.term_key] != 1]
    if repeated:
        names = ", ".join(sorted(repeated))
        raise ValueError(f"calibration probe terms must occur exactly once: {names}")


def _lesson_sentences(
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> list[LessonSentence]:
    return [
        *([title_sentence] if title_sentence is not None else []),
        *(sentence for block in blocks for sentence in block.sentences),
    ]


def _validate_unique_grammar_occurrence_keys(sentences: Sequence[LessonSentence]) -> None:
    occurrence_keys = [occurrence.key for sentence in sentences for occurrence in sentence.grammar]
    if len(occurrence_keys) != len(set(occurrence_keys)):
        raise ValueError("grammar occurrence keys must be unique across a lesson")


def _validate_grammar_structure(
    learning_language: str,
    blocks: Sequence[LessonBlock],
    title_sentence: LessonSentence | None = None,
) -> None:
    sentences = _lesson_sentences(blocks, title_sentence)
    _validate_unique_grammar_occurrence_keys(sentences)
    construction_keys = {
        occurrence.construction_key for sentence in sentences for occurrence in sentence.grammar
    }
    if not construction_keys:
        return
    try:
        catalog = grammar_catalog(learning_language)
    except LookupError as error:
        raise ValueError(str(error)) from error
    unknown = construction_keys - {construction.key for construction in catalog.constructions}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(
            f"grammar constructions do not belong to learning language {learning_language}: {names}"
        )


class ProfileUpdate(StrictModel):
    learning_language: LanguageTag | None = Field(
        default=None, description=LANGUAGE_TAG_DESCRIPTION
    )
    translation_language: LanguageTag | None = Field(
        default=None, description=LANGUAGE_TAG_DESCRIPTION
    )
    level: CefrLevel | None = None
    difficulty: float | None = Field(default=None, ge=0.0, le=1.0)
    interests: list[str] | None = None
    preferences: dict[str, Any] | None = None

    @field_validator("learning_language", "translation_language")
    @classmethod
    def strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("interests")
    @classmethod
    def normalize_interests(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class ProficiencyView(StrictModel):
    source: LevelSource = "self_reported"
    status: CalibrationStatus = "unstarted"
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    lower_level: CefrLevel | None = None
    upper_level: CefrLevel | None = None
    qualified_attempts: int = Field(default=0, ge=0)
    usable_probes: int = Field(default=0, ge=0)
    qualified_readings: int = Field(default=0, ge=0)


class ProfileView(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    interests: list[str]
    preferences: dict[str, Any]
    proficiency: ProficiencyView = Field(default_factory=ProficiencyView)


class WorkspaceSummary(StrictModel):
    profile_id: str
    label: str
    learning_language: LanguageTag
    translation_language: LanguageTag


class WorkspaceList(StrictModel):
    selected_profile_id: str
    profiles: list[WorkspaceSummary]


class InteractionIn(StrictModel):
    event_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    lesson_id: int = Field(gt=0)
    type: EventType
    occurred_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "session_id")
    @classmethod
    def strip_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_payload(self) -> InteractionIn:
        if self.type == "term.revealed":
            _require_string(self.payload, "term_key")
        elif self.type == "translation.revealed":
            scope = self.payload.get("scope")
            if scope not in {"sentence", "lesson", "grammar"}:
                raise ValueError(
                    "translation.revealed payload scope must be sentence, lesson, or grammar"
                )
            if scope in {"sentence", "grammar"}:
                _require_string(self.payload, "sentence_key")
            if scope == "grammar":
                _require_string(self.payload, "occurrence_key")
                _require_string(self.payload, "construction_key")
        elif self.type == "lesson.completed":
            _validate_completion_payload(self.payload)
        elif self.type == "lesson.rated":
            rating = self.payload.get("rating")
            if rating not in {-1, 1, None}:
                raise ValueError("lesson.rated payload rating must be -1, 1, or null")
            _validate_feedback(self.payload)
            if rating is None and not self.payload.get("feedback"):
                raise ValueError("lesson.rated payload must include a rating or feedback")
        return self


class InteractionBatch(StrictModel):
    events: list[InteractionIn] = Field(min_length=1)


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"payload.{key} must be a non-blank string")
    return value


def _validate_completion_payload(payload: dict[str, Any]) -> None:
    active_seconds = payload.get("active_seconds")
    completion_ratio = payload.get("completion_ratio")
    if active_seconds is not None and (
        not isinstance(active_seconds, (int, float)) or active_seconds < 0
    ):
        raise ValueError("payload.active_seconds must be non-negative")
    if completion_ratio is not None and (
        not isinstance(completion_ratio, (int, float)) or not 0 <= completion_ratio <= 1
    ):
        raise ValueError("payload.completion_ratio must be between 0 and 1")


def _validate_feedback(payload: dict[str, Any]) -> None:
    if "feedback" not in payload:
        return
    feedback = payload["feedback"]
    allowed = set(get_args(FeedbackTag))
    if not isinstance(feedback, list) or any(
        not isinstance(item, str) or item not in allowed for item in feedback
    ):
        raise ValueError("payload.feedback must be a list of known feedback tags")
    if len(feedback) != len(set(feedback)):
        raise ValueError("payload.feedback must contain unique tags")
    contradictions = (
        {"shorter", "longer"},
        {"easier", "more_challenging"},
        {"more_grammar", "less_grammar"},
        {"same_topic", "new_topic"},
    )
    if any(pair <= set(feedback) for pair in contradictions):
        raise ValueError("payload.feedback contains contradictory tags")


class EventRecordResult(StrictModel):
    accepted: int
    duplicates: int
    accepted_event_ids: list[str]


class ReaderProgress(StrictModel):
    session_id: str | None = None
    started: bool = False
    completed: bool = False
    rating: Literal[-1, 1] | None = None
    feedback: list[FeedbackTag] = Field(default_factory=list)
    revealed_term_keys: list[str] = Field(default_factory=list)
    revealed_sentence_keys: list[str] = Field(default_factory=list)
    revealed_grammar_occurrence_keys: list[str] = Field(default_factory=list)
    full_translation_revealed: bool = False


class AgentGrammarCatalogEntry(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    generation_hint: str | None = None


class ReaderState(StrictModel):
    profile: ProfileView
    lesson_id: int | None = None
    lesson: LessonDocument | None = None
    progress: ReaderProgress = Field(default_factory=ReaderProgress)
    term_bands: dict[str, TermKnowledgeBand] = Field(default_factory=dict)
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    comfortable_grammar_construction_keys: list[str] = Field(default_factory=list)
    generation_status: Literal["pending", "running", "failed"] | None = None


class TextView(StrictModel):
    id: int = Field(gt=0)
    key: str
    title: str
    topic: str | None
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    imported_at: datetime
    status: TextStatus
    queue_position: int | None = Field(default=None, ge=1)
    opened_at: datetime | None
    last_completed_at: datetime | None
    skipped_at: datetime | None = None
    session_count: int = Field(ge=0)
    completion_count: int = Field(ge=0)
    rating: Literal[-1, 1] | None
    lexical_token_count: int = Field(ge=0)
    # Predicted share of running words the learner currently knows.
    known_share: float | None = Field(default=None, ge=0.0, le=1.0)


class LessonQueueActionIn(StrictModel):
    action_id: UUID


class LessonQueueActionView(StrictModel):
    action_id: UUID
    lesson_id: int = Field(gt=0)
    skipped: bool
    occurred_at: datetime


class LessonQueueMoveIn(StrictModel):
    action_id: UUID
    direction: Literal["up", "down"]
    neighbor_lesson_id: int = Field(gt=0)


class LessonQueueMoveView(StrictModel):
    action_id: UUID
    lesson_id: int = Field(gt=0)
    direction: Literal["up", "down"]
    neighbor_lesson_id: int = Field(gt=0)
    occurred_at: datetime


class TextRequestIn(StrictModel):
    request_id: UUID
    topic: str = Field(min_length=1, max_length=200)

    @field_validator("topic")
    @classmethod
    def strip_topic(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return " ".join(value.split())


class TextRequestView(StrictModel):
    task_id: int = Field(gt=0)
    request_id: UUID
    topic: str
    state: GenerationTaskState
    created_at: datetime
    updated_at: datetime
    error: str | None
    lesson_id: int | None = Field(default=None, gt=0)


class TextPreparationView(StrictModel):
    task_id: int = Field(gt=0)
    state: Literal["pending", "running"]
    request_kind: GenerationRequestKind
    requested_topic: str | None = Field(default=None, min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_requested_topic(self) -> TextPreparationView:
        if self.request_kind == "topic_request" and self.requested_topic is None:
            raise ValueError("topic-request preparation requires requested_topic")
        if self.request_kind == "queue_fill" and self.requested_topic is not None:
            raise ValueError("queue-fill preparation must not include requested_topic")
        return self


class TextsState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    texts: list[TextView]
    preparations: list[TextPreparationView]
    requests: list[TextRequestView]


class TextDetail(StrictModel):
    """A deliberately session-free lesson payload for no-evidence previewing."""

    lesson_id: int = Field(gt=0)
    lesson: LessonDocument
    term_bands: dict[str, TermKnowledgeBand]
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    comfortable_grammar_construction_keys: list[str] = Field(default_factory=list)


class WordSurfaceForm(StrictModel):
    text: str = Field(min_length=1)
    occurrences: int = Field(ge=1)


class RelatedWordSense(StrictModel):
    """Another observed sense or grammatical role with the same written lemma."""

    key: str
    pos: str
    gloss: str
    pronunciation: str | None


class WordView(StrictModel):
    key: str
    lemma: str
    pos: str
    gloss: str
    pronunciation: str | None
    frequency_rank: int | None
    surface_forms: list[WordSurfaceForm] = Field(min_length=1)
    related_senses: list[RelatedWordSense] = Field(default_factory=list)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_reveal_count: int = Field(ge=0)
    counted_reveal_sessions: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    reveal_failures: float = Field(ge=0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_revealed_at: datetime | None
    next_due_at: datetime | None


class WordsState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    words: list[WordView]


class CharacterView(StrictModel):
    """Observed use and replayable inferred recognition of one Han character."""

    character: str = Field(min_length=1, max_length=1)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_reveal_count: int = Field(ge=0)
    counted_reveal_sessions: int = Field(ge=0)
    distinct_word_contexts: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    inferred_failure_sessions: int = Field(ge=0)
    inferred_failure_mass: float = Field(ge=0.0)
    direct_successes: int = Field(ge=0)
    direct_failures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    mastery_uncertainty: float = Field(ge=0.0, le=0.5)
    retrievability: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_evidence_at: datetime | None
    next_due_at: datetime | None


class CharactersState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    characters: list[CharacterView]


class StatisticsWordsSummary(StrictModel):
    total: int = Field(ge=0)
    learning: int = Field(ge=0)
    expected: int = Field(ge=0)
    familiar: int = Field(ge=0)
    mastered: int = Field(ge=0)
    expected_min_mastery: float = Field(ge=0.0, le=1.0)
    familiar_min_mastery: float = Field(ge=0.0, le=1.0)
    mastered_min_mastery: float = Field(ge=0.0, le=1.0)


class StatisticsReadingSummary(StrictModel):
    texts_read: int = Field(ge=0)
    completed_sessions: int = Field(ge=0)
    total_active_seconds: float = Field(ge=0.0)
    recent_average_wpm: float | None = Field(default=None, ge=0.0)
    recent_wpm_sessions: int = Field(ge=0)
    lifetime_average_wpm: float | None = Field(default=None, ge=0.0)
    lifetime_wpm_sessions: int = Field(ge=0)
    recent_window_size: int = Field(ge=1)
    minimum_wpm_active_seconds: float = Field(ge=0.0)
    minimum_wpm_completion_ratio: float = Field(ge=0.0, le=1.0)


class StatisticsLevelSummary(StrictModel):
    value: float = Field(ge=0.0, le=1.0)
    category: CefrLevel
    source: LevelSource
    status: CalibrationStatus
    lower: float | None = Field(default=None, ge=0.0, le=1.0)
    upper: float | None = Field(default=None, ge=0.0, le=1.0)
    lower_category: CefrLevel | None = None
    upper_category: CefrLevel | None = None
    qualified_attempts: int = Field(ge=0)
    usable_probes: int = Field(ge=0)
    qualified_readings: int = Field(default=0, ge=0)


class StatisticsSummary(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    words: StatisticsWordsSummary
    reading: StatisticsReadingSummary
    level: StatisticsLevelSummary


class GrammarExampleView(StrictModel):
    lesson_id: int = Field(gt=0)
    title: str
    sentence_key: str
    text: str
    translation: str | None = None
    note: str | None = None


class GrammarView(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    occurrence_count: int = Field(ge=1)
    exposed_lesson_count: int = Field(ge=1)
    raw_help_count: int = Field(ge=0)
    counted_help_sessions: int = Field(ge=0)
    inferred_difficulty_signals: int = Field(ge=0)
    qualified_exposures: int = Field(ge=0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    first_exposed_at: datetime
    last_exposed_at: datetime
    last_helped_at: datetime | None
    next_due_at: datetime | None
    examples: list[GrammarExampleView] = Field(default_factory=list)


class GrammarState(StrictModel):
    learning_language: LanguageTag
    translation_language: LanguageTag
    constructions: list[GrammarView]


class AgentTermBrief(StrictModel):
    key: str
    lemma: str
    pos: str
    gloss: str
    pronunciation: str | None
    frequency_rank: int | None
    mastery: float
    stability_days: float
    qualified_exposures: int
    reveal_failures: float
    next_due_at: datetime | None
    urgency: float
    reason: Literal["unseen", "due", "fragile"]


class AgentGrammarBrief(StrictModel):
    key: str
    label: str
    description: str
    category: str
    difficulty: float = Field(ge=0.0, le=1.0)
    mastery: float = Field(ge=0.0, le=1.0)
    stability_days: float = Field(ge=0.0)
    qualified_exposures: int = Field(ge=0)
    help_failures: int = Field(ge=0)
    inferred_difficulty_signals: int = Field(ge=0)
    next_due_at: datetime | None
    urgency: float
    reason: Literal["unseen", "due", "fragile"]


class AgentLessonBrief(StrictModel):
    id: int
    key: str
    title: str
    topic: str | None
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    imported_at: datetime
    completed: bool
    rating: Literal[-1, 1] | None
    feedback: list[FeedbackTag]
    target_term_keys: list[str]
    grammar_keys: list[str] = Field(default_factory=list)
    opening_excerpt: str = Field(max_length=180)
    ending_excerpt: str = Field(max_length=180)
    calibration_lesson: bool = False
    metadata: dict[str, Any]
    known_share: float | None = Field(default=None, ge=0.0, le=1.0)


class NewWordCandidate(StrictModel):
    lemma: str = Field(min_length=1, max_length=200)
    pronunciation: str | None = Field(default=None, max_length=300)
    gloss: str = Field(max_length=500)
    frequency_rank: int | None = Field(default=None, ge=1)
    level: int | None = Field(default=None, ge=1)
    source: str = Field(min_length=1, max_length=40)


class VocabularyPlan(StrictModel):
    """New-vocabulary guidance: list words near the frontier plus the agent's own picks."""

    list_candidates: list[NewWordCandidate] = Field(default_factory=list, max_length=32)
    list_new_words: int = Field(ge=0, le=12)
    free_new_words: int = Field(ge=0, le=12)
    list_exhausted: bool = False
    target_known_share: float = Field(ge=0.5, le=1.0)
    recent_known_share: float | None = Field(default=None, ge=0.0, le=1.0)


class AgentBrief(StrictModel):
    schema_version: Literal[1] = 1
    generated_at: datetime
    profile: ProfileView
    lesson_count: int
    completed_lesson_count: int
    interaction_count: int
    language_guidance: list[str]
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    priority_grammar: list[AgentGrammarBrief] = Field(default_factory=list)
    priority_terms: list[AgentTermBrief]
    mastered_term_keys: list[str]
    recent_lessons: list[AgentLessonBrief]
    vocabulary: VocabularyPlan | None = None


class GeneratedLessonBlock(LessonBlock):
    sentences: list[LessonSentence] = Field(min_length=1, max_length=20)


class GeneratedLessonDraft(StrictModel):
    title: str = Field(min_length=1, max_length=300)
    title_sentence: LessonSentence = Field(
        description=(
            "Fully annotated title whose text exactly equals title and includes its translation."
        )
    )
    topic: str | None = Field(max_length=200)
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    blocks: list[GeneratedLessonBlock] = Field(
        min_length=1,
        max_length=12,
        description=(
            "Fully clickable text: every lexical token is a term-bearing run; plain runs contain "
            "only whitespace or punctuation."
        ),
    )
    target_term_keys: list[str] = Field(
        max_length=16,
        description=(
            "Advisory target choices. The host reconciles these against the offered candidates "
            "and terms actually present in the lesson."
        ),
    )
    calibration: LessonCalibration | None = Field(
        default=None,
        description="Optional calibration-probe subset, independent of target_term_keys.",
    )

    @model_validator(mode="after")
    def validate_targets(self) -> GeneratedLessonDraft:
        terms: dict[str, LessonTerm] = {}
        if self.title_sentence.text != self.title:
            raise ValueError("title_sentence text must exactly equal title")
        body_sentences = [sentence for block in self.blocks for sentence in block.sentences]
        sentences = [self.title_sentence, *body_sentences]
        sentence_keys = [sentence.key for sentence in sentences]
        if len(sentence_keys) != len(set(sentence_keys)):
            raise ValueError("generated sentence keys must be unique, including title_sentence")
        _validate_unique_grammar_occurrence_keys(sentences)
        for sentence in sentences:
            for run in sentence.runs:
                _validate_generated_run_coverage(run)
                if (
                    run.term is not None
                    and run.term.frequency_rank is None
                    and not is_proper_noun_pos(run.term.pos)
                ):
                    raise ValueError(
                        "generated non-name terms require an approximate corpus frequency_rank"
                    )
            _collect_sentence_terms(sentence, terms, generated=True)
        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("target_term_keys must be unique")
        if len(sentences) > 40:
            raise ValueError("lesson draft must contain at most 40 sentences")
        if any(len(block.sentences) > 20 for block in self.blocks):
            raise ValueError("lesson draft blocks must contain at most 20 sentences")
        if any(len(sentence.runs) > 100 for sentence in sentences):
            raise ValueError("lesson draft sentences must contain at most 100 runs")
        if any(sentence.translation is None for sentence in sentences):
            raise ValueError("every generated sentence must include a translation")
        _validate_calibration_structure(self.calibration, self.blocks, self.title_sentence)
        return self


class CallbackLessonRun(StrictModel):
    """Compact callback run; canonical term metadata lives in the lesson catalog."""

    text: str = Field(min_length=1)
    term_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Key in this lesson's terms catalog, or null for whitespace/punctuation.",
    )
    pronunciation: str | None = Field(
        default=None,
        max_length=300,
        description="Optional contextual surface reading; dictionary reading belongs on the term.",
    )

    @field_validator("term_key", "pronunciation")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)
    translation: str
    grammar: list[GrammarOccurrence] = Field(default_factory=list, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[CallbackLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class CallbackLessonDraft(StrictModel):
    """Token-efficient callback contract expanded before host validation and persistence."""

    title: str = Field(min_length=1, max_length=300)
    terms: list[LessonTerm] = Field(
        min_length=1,
        max_length=4100,
        description=(
            "Lesson-local term catalog. Define each key exactly once, then reference it from runs."
        ),
    )
    title_sentence: CallbackLessonSentence = Field(
        description=(
            "Fully annotated title whose run text exactly equals title and includes its "
            "translation."
        )
    )
    topic: str | None = Field(max_length=200)
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    blocks: list[CallbackLessonBlock] = Field(min_length=1, max_length=12)
    target_term_keys: list[str] = Field(
        max_length=16,
        description=(
            "Advisory target choices. The host reconciles these against offered candidates and "
            "terms present in the body."
        ),
    )
    calibration: LessonCalibration | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def accept_expanded_python_drafts(cls, value: Any) -> Any:
        """Keep programmatic fixtures compatible while the emitted JSON schema stays compact."""

        if isinstance(value, GeneratedLessonDraft):
            value = value.model_dump(mode="python")
        if not isinstance(value, dict) or "terms" in value:
            return value
        if "title_sentence" not in value or "blocks" not in value:
            return value
        return _compact_expanded_lesson(value)

    def term_catalog(self) -> dict[str, LessonTerm]:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"callback term catalog defines key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "callback non-name catalog terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term
        return catalog

    def expand(self) -> GeneratedLessonDraft:
        catalog = self.term_catalog()

        def expand_sentence(sentence: CallbackLessonSentence) -> LessonSentence:
            runs: list[LessonRun] = []
            for run_index, run in enumerate(sentence.runs):
                term = None
                if run.term_key is not None:
                    term = catalog.get(run.term_key)
                    if term is None:
                        raise ValueError(
                            f"callback run references unknown term key {run.term_key!r} at "
                            f"sentence {sentence.key!r}, run {run_index}, surface {run.text!r}"
                        )
                runs.append(
                    LessonRun(
                        text=run.text,
                        term=term,
                        pronunciation=run.pronunciation,
                    )
                )
            return LessonSentence(
                key=sentence.key,
                runs=runs,
                translation=sentence.translation,
                grammar=sentence.grammar,
            )

        title_sentence = expand_sentence(self.title_sentence)
        blocks = [
            GeneratedLessonBlock(
                key=block.key,
                sentences=[expand_sentence(sentence) for sentence in block.sentences],
            )
            for block in self.blocks
        ]
        return GeneratedLessonDraft(
            title=self.title,
            title_sentence=title_sentence,
            topic=self.topic,
            level=self.level,
            difficulty=self.difficulty,
            blocks=blocks,
            target_term_keys=self.target_term_keys,
            calibration=self.calibration,
        )

    @model_validator(mode="after")
    def validate_expanded_lesson(self) -> CallbackLessonDraft:
        catalog = self.term_catalog()
        unresolved = any(
            run.term_key is not None and run.term_key not in catalog
            for sentence in [
                self.title_sentence,
                *(sentence for block in self.blocks for sentence in block.sentences),
            ]
            for run in sentence.runs
        )
        if not unresolved:
            self.expand()
        return self


def _compact_expanded_lesson(value: dict[str, Any]) -> dict[str, Any]:
    """Convert the old repeated-term Python shape without exposing it in callback JSON Schema."""

    compact = dict(value)
    catalog: dict[str, LessonTerm] = {}

    def compact_sentence(raw_sentence: Any) -> Any:
        if isinstance(raw_sentence, LessonSentence):
            sentence = raw_sentence.model_dump(mode="python")
        elif isinstance(raw_sentence, dict):
            sentence = dict(raw_sentence)
        else:
            return raw_sentence
        compact_runs: list[dict[str, Any]] = []
        for raw_run in sentence.get("runs", []):
            if isinstance(raw_run, LessonRun):
                run = raw_run.model_dump(mode="python")
            else:
                run = dict(raw_run)
            raw_term = run.pop("term", None)
            if raw_term is None:
                run.setdefault("term_key", None)
            else:
                term = LessonTerm.model_validate(raw_term)
                previous = catalog.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(f"term key has conflicting generated definitions: {term.key}")
                catalog[term.key] = term
                run["term_key"] = term.key
            compact_runs.append(run)
        sentence["runs"] = compact_runs
        return sentence

    compact["title_sentence"] = compact_sentence(compact["title_sentence"])
    compact_blocks: list[dict[str, Any]] = []
    for raw_block in compact["blocks"]:
        if isinstance(raw_block, LessonBlock):
            block = raw_block.model_dump(mode="python")
        else:
            block = dict(raw_block)
        block["sentences"] = [compact_sentence(sentence) for sentence in block.get("sentences", [])]
        compact_blocks.append(block)
    compact["blocks"] = compact_blocks
    compact["terms"] = list(catalog.values())
    return compact


class CalibrationGenerationBrief(StrictModel):
    sequence: int = Field(ge=1)
    target_difficulty: float = Field(ge=0.0, le=1.0)
    probe_difficulties: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        min_length=8, max_length=20
    )
    excluded_term_keys: list[str]


class GenerationTargetPolicy(StrictModel):
    preferred_count: int = Field(ge=0, le=16)
    max_count: int = Field(ge=0, le=16)
    target_text_length: int = Field(ge=1, le=10_000)
    allow_zero: bool = True
    candidate_term_keys: list[str] = Field(max_length=64)

    @model_validator(mode="after")
    def validate_policy(self) -> GenerationTargetPolicy:
        if self.preferred_count > self.max_count:
            raise ValueError("preferred_count must not exceed max_count")
        if len(self.candidate_term_keys) != len(set(self.candidate_term_keys)):
            raise ValueError("candidate_term_keys must be unique")
        return self


class GenerationGrammarPolicy(StrictModel):
    preferred_count: int = Field(default=0, ge=0, le=4)
    max_count: int = Field(default=0, ge=0, le=4)
    offered_keys: tuple[str, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def validate_policy(self) -> GenerationGrammarPolicy:
        if self.preferred_count > self.max_count:
            raise ValueError("grammar preferred_count must not exceed max_count")
        if self.max_count > len(self.offered_keys):
            raise ValueError("grammar max_count must not exceed the offered pool")
        if len(self.offered_keys) != len(set(self.offered_keys)):
            raise ValueError("grammar offered_keys must be unique")
        return self


class GenerationContentPlan(StrictModel):
    archetype: str = Field(min_length=1, max_length=100)
    discourse_form: str = Field(min_length=1, max_length=200)
    perspective: str = Field(min_length=1, max_length=200)
    concrete_seed: str = Field(min_length=1, max_length=500)
    progression: str = Field(min_length=1, max_length=500)
    ending_shape: str = Field(min_length=1, max_length=300)
    avoid_patterns: list[str] = Field(default_factory=list, max_length=8)


class GenerationFailureContext(StrictModel):
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    error: str = Field(min_length=1, max_length=4000)


class ProseLessonSentence(StrictModel):
    """Frozen learning-language text before lexical annotation."""

    key: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProseLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[ProseLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class ProseLessonDraft(StrictModel):
    """Complete prose whose text and stable keys remain immutable in downstream tasks."""

    title: str = Field(min_length=1, max_length=300)
    title_sentence: ProseLessonSentence
    topic: str | None = Field(max_length=200)
    level: CefrLevel
    difficulty: float = Field(ge=0.0, le=1.0)
    blocks: list[ProseLessonBlock] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_structure(self) -> ProseLessonDraft:
        if self.title_sentence.text != self.title:
            raise ValueError("prose title_sentence text must exactly equal title")
        sentences = [
            self.title_sentence,
            *(sentence for block in self.blocks for sentence in block.sentences),
        ]
        keys = [sentence.key for sentence in sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("prose sentence keys must be unique within a lesson")
        if len(sentences) > 41:
            raise ValueError("prose lesson must contain at most 40 body sentences")
        block_keys = [block.key for block in self.blocks]
        if len(block_keys) != len(set(block_keys)):
            raise ValueError("prose block keys must be unique within a lesson")
        return self


class GenerationProseResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)


class LexicalLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[LexicalLessonSentence] = Field(min_length=1, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalLessonDraft(StrictModel):
    """Term catalog and token boundaries without translations or grammar."""

    terms: list[LessonTerm] = Field(min_length=1, max_length=4100)
    title_sentence: LexicalLessonSentence
    blocks: list[LexicalLessonBlock] = Field(min_length=1, max_length=12)
    target_term_keys: list[str] = Field(default_factory=list, max_length=16)

    def term_catalog(self) -> dict[str, LessonTerm]:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"lexical term catalog defines key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "lexical non-name catalog terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term
        return catalog

    @model_validator(mode="after")
    def validate_structure(self) -> LexicalLessonDraft:
        catalog = self.term_catalog()
        block_keys = [block.key for block in self.blocks]
        if len(block_keys) != len(set(block_keys)):
            raise ValueError("lexical block keys must be unique within a lesson")
        sentences = [
            self.title_sentence,
            *(sentence for block in self.blocks for sentence in block.sentences),
        ]
        sentence_keys = [sentence.key for sentence in sentences]
        if len(sentence_keys) != len(set(sentence_keys)):
            raise ValueError("lexical sentence keys must be unique within a lesson")
        for sentence in sentences:
            for run_index, run in enumerate(sentence.runs):
                if run.term_key is not None and run.term_key not in catalog:
                    raise ValueError(
                        f"lexical run references unknown term key {run.term_key!r} at "
                        f"sentence {sentence.key!r}, run {run_index}, surface {run.text!r}"
                    )
        if len(self.target_term_keys) != len(set(self.target_term_keys)):
            raise ValueError("lexical target_term_keys must be unique")
        unknown_targets = set(self.target_term_keys) - catalog.keys()
        if unknown_targets:
            names = ", ".join(sorted(unknown_targets))
            raise ValueError(f"lexical target_term_keys contains unknown terms: {names}")
        return self


class GenerationLexicalResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[LexicalLessonDraft] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_stable_terms(self) -> GenerationLexicalResult:
        terms: dict[str, LessonTerm] = {}
        for lesson in self.lessons:
            for term in lesson.terms:
                previous = terms.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(
                        f"term key has conflicting lexical definitions across lessons: {term.key}"
                    )
                terms[term.key] = term
        return self


class SentenceTranslationDraft(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    translation: str = Field(min_length=1, max_length=20_000)

    @field_validator("key", "translation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class TranslationLessonDraft(StrictModel):
    sentences: list[SentenceTranslationDraft] = Field(min_length=1, max_length=41)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> TranslationLessonDraft:
        keys = [sentence.key for sentence in self.sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("translation sentence keys must be unique within a lesson")
        return self


class GenerationTranslationResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[TranslationLessonDraft] = Field(min_length=1, max_length=10)


class CallbackGrammarOccurrence(StrictModel):
    """A provider-authored range; the host assigns its stable occurrence key."""

    construction_key: str = Field(min_length=1, max_length=100)
    run_start: int = Field(ge=0)
    run_end: int = Field(ge=1)
    note: str | None = Field(default=None, max_length=500)

    @field_validator("construction_key")
    @classmethod
    def strip_construction_key(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("note")
    @classmethod
    def strip_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_range(self) -> CallbackGrammarOccurrence:
        if self.run_end <= self.run_start:
            raise ValueError("grammar occurrence run_end must be greater than run_start")
        return self


class GrammarSentenceDraft(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    occurrences: list[CallbackGrammarOccurrence] = Field(default_factory=list, max_length=20)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_unique_occurrences(self) -> GrammarSentenceDraft:
        identities = [
            (
                occurrence.construction_key,
                occurrence.run_start,
                occurrence.run_end,
            )
            for occurrence in self.occurrences
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("grammar occurrences must not duplicate a construction and range")
        return self


class GrammarLessonDraft(StrictModel):
    sentences: list[GrammarSentenceDraft] = Field(min_length=1, max_length=41)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> GrammarLessonDraft:
        keys = [sentence.key for sentence in self.sentences]
        if len(keys) != len(set(keys)):
            raise ValueError("grammar sentence keys must be unique within a lesson")
        return self


class GenerationGrammarResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[GrammarLessonDraft] = Field(min_length=1, max_length=10)


class TokenizedLessonSentence(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    runs: list[str] = Field(min_length=1, max_length=100)

    @field_validator("key")
    @classmethod
    def strip_key(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("runs")
    @classmethod
    def validate_runs(cls, value: list[str]) -> list[str]:
        if any(not run for run in value):
            raise ValueError("tokenized lesson runs must not be empty")
        return value


class TokenizedLessonBlock(StrictModel):
    key: str = Field(min_length=1, max_length=200)
    sentences: list[TokenizedLessonSentence] = Field(min_length=1, max_length=20)


class TokenizedLessonDraft(StrictModel):
    title_sentence: TokenizedLessonSentence
    blocks: list[TokenizedLessonBlock] = Field(min_length=1, max_length=12)


class GenerationStageRequestBase(StrictModel):
    schema_version: Literal[3] = 3
    stage: Literal["lexical", "translation", "grammar"]
    task: Literal["lexical", "translation", "grammar"]
    job_id: str = Field(min_length=3, max_length=100)
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    lesson_count: int = Field(ge=1, le=10)
    profile_key: str = Field(min_length=1, max_length=100)
    workspace_path: str = Field(min_length=1, max_length=1000)
    profile_fingerprint: str = Field(min_length=64, max_length=64)
    learning_language: LanguageTag
    translation_language: LanguageTag
    previous_failures: list[GenerationFailureContext] = Field(default_factory=list, max_length=4)
    instructions: str

    @model_validator(mode="after")
    def validate_stage_task(self) -> GenerationStageRequestBase:
        if self.stage != self.task:
            raise ValueError("staged generation request stage and task must match")
        return self


class GenerationLexicalUnitResult(StrictModel):
    """Sentence-local lexical output; term keys are scoped to this unit until host assembly."""

    schema_version: Literal[1]
    unit_id: str = Field(min_length=1, max_length=200)
    key: str = Field(min_length=1, max_length=200)
    terms: list[LessonTerm] = Field(max_length=200)
    runs: list[CallbackLessonRun] = Field(min_length=1, max_length=100)

    @field_validator("unit_id", "key")
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_catalog_and_runs(self) -> GenerationLexicalUnitResult:
        catalog: dict[str, LessonTerm] = {}
        for term in self.terms:
            if term.key in catalog:
                raise ValueError(f"lexical unit defines term key more than once: {term.key}")
            if term.frequency_rank is None and not is_proper_noun_pos(term.pos):
                raise ValueError(
                    "lexical unit non-name terms require an approximate corpus "
                    f"frequency_rank: {term.key}"
                )
            catalog[term.key] = term

        for run_index, run in enumerate(self.runs):
            resolved_term = catalog.get(run.term_key) if run.term_key is not None else None
            if run.term_key is not None and resolved_term is None:
                raise ValueError(
                    f"lexical unit run references unknown term key {run.term_key!r} at "
                    f"run {run_index}, surface {run.text!r}"
                )
            _validate_generated_run_coverage(
                LessonRun(
                    text=run.text,
                    term=resolved_term,
                    pronunciation=run.pronunciation,
                )
            )
        return self


class GenerationLexicalUnitRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["sentence"] = "sentence"
    unit_id: str = Field(min_length=1, max_length=200)
    lesson_index: int = Field(ge=0, le=9)
    sentence_index: int = Field(ge=0, le=40)
    unit_attempt: int = Field(default=1, ge=1, le=2)
    is_title: bool
    frozen_sentence: ProseLessonSentence
    context_sentences: list[ProseLessonSentence] = Field(default_factory=list, max_length=41)
    known_terms: list[LessonTerm] = Field(default_factory=list, max_length=256)
    language_guidance: list[str] = Field(default_factory=list, max_length=32)
    repair_result: GenerationLexicalUnitResult | None = None

    @field_validator("unit_id")
    @classmethod
    def strip_unit_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("language_guidance")
    @classmethod
    def validate_language_guidance(cls, value: list[str]) -> list[str]:
        guidance = [item.strip() for item in value]
        if any(not item for item in guidance):
            raise ValueError("language guidance must not contain blank entries")
        if any(len(item) > 4_000 for item in guidance):
            raise ValueError("language guidance entries must contain at most 4000 characters")
        return guidance

    @model_validator(mode="after")
    def validate_unit_context(self) -> GenerationLexicalUnitRequest:
        if self.lesson_index >= self.lesson_count:
            raise ValueError("lexical unit lesson_index is outside lesson_count")
        context_by_key: dict[str, ProseLessonSentence] = {}
        for sentence in self.context_sentences:
            previous = context_by_key.get(sentence.key)
            if previous is not None:
                raise ValueError(f"lexical unit repeats context sentence key: {sentence.key}")
            context_by_key[sentence.key] = sentence
        matching_context = context_by_key.get(self.frozen_sentence.key)
        if matching_context is not None and matching_context != self.frozen_sentence:
            raise ValueError("lexical unit context conflicts with its frozen sentence")

        known_keys: set[str] = set()
        for term in self.known_terms:
            if term.key in known_keys:
                raise ValueError(f"lexical unit repeats known term key: {term.key}")
            known_keys.add(term.key)

        if self.repair_result is not None:
            if self.attempt <= 1 and self.unit_attempt <= 1:
                raise ValueError("lexical unit repair_result is valid only on a task or unit retry")
            if self.repair_result.unit_id != self.unit_id:
                raise ValueError("lexical unit repair_result belongs to a different unit")
            if self.repair_result.key != self.frozen_sentence.key:
                raise ValueError("lexical unit repair_result has a different sentence key")
        return self


class LexicalBatchUnit(StrictModel):
    unit_id: str = Field(min_length=1, max_length=200)
    lesson_index: int = Field(ge=0, le=9)
    sentence_index: int = Field(ge=0, le=40)
    is_title: bool
    frozen_sentence: ProseLessonSentence


MAX_LEXICAL_BATCH_UNITS = 16


class GenerationLexicalBatchRequest(GenerationStageRequestBase):
    """Several frozen sentences tokenized in one callback; each result stays sentence-local."""

    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["sentence_batch"] = "sentence_batch"
    units: list[LexicalBatchUnit] = Field(min_length=1, max_length=MAX_LEXICAL_BATCH_UNITS)
    context_sentences: list[ProseLessonSentence] = Field(default_factory=list, max_length=41)
    known_terms: list[LessonTerm] = Field(default_factory=list, max_length=256)
    language_guidance: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_batch(self) -> GenerationLexicalBatchRequest:
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("lexical batch repeats a unit id")
        if any(unit.lesson_index >= self.lesson_count for unit in self.units):
            raise ValueError("lexical batch lesson_index is outside lesson_count")
        known_keys = [term.key for term in self.known_terms]
        if len(known_keys) != len(set(known_keys)):
            raise ValueError("lexical batch repeats a known term key")
        return self


class GenerationLexicalBatchResult(StrictModel):
    """Response shape for a batch; the host validates each unit independently."""

    schema_version: Literal[1]
    units: list[GenerationLexicalUnitResult] = Field(
        min_length=1, max_length=MAX_LEXICAL_BATCH_UNITS
    )


class LexicalConflictContext(StrictModel):
    sentence_key: str = Field(min_length=1, max_length=200)
    surface: str = Field(min_length=1, max_length=500)
    sentence_text: str = Field(min_length=1, max_length=20_000)

    @field_validator("sentence_key", "surface", "sentence_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class LexicalConflictCandidate(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    term: LessonTerm
    stored: bool
    contexts: list[LexicalConflictContext] = Field(default_factory=list, max_length=41)

    @field_validator("candidate_id")
    @classmethod
    def strip_candidate_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_contexts(self) -> LexicalConflictCandidate:
        identities = [
            (context.sentence_key, context.surface, context.sentence_text)
            for context in self.contexts
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("lexical conflict candidate contexts must be unique")
        if not self.stored and not self.contexts:
            raise ValueError("generated lexical conflict candidates require context")
        return self


class LexicalConflictGroup(StrictModel):
    group_id: str = Field(min_length=1, max_length=200)
    candidates: list[LexicalConflictCandidate] = Field(min_length=2, max_length=16)

    @field_validator("group_id")
    @classmethod
    def strip_group_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_candidate_ids(self) -> LexicalConflictGroup:
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict candidate IDs must be unique within a group")
        return self


class GenerationLexicalConflictRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    request_kind: Literal["conflicts"] = "conflicts"
    groups: list[LexicalConflictGroup] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_group_and_candidate_ids(self) -> GenerationLexicalConflictRequest:
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("lexical conflict group IDs must be unique")
        candidate_ids = [
            candidate.candidate_id for group in self.groups for candidate in group.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict candidate IDs must be globally unique")
        return self


class LexicalConflictResolution(StrictModel):
    candidate_id: str = Field(min_length=1, max_length=200)
    canonical_candidate_id: str = Field(min_length=1, max_length=200)

    @field_validator("candidate_id", "canonical_candidate_id")
    @classmethod
    def strip_candidate_ids(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class GenerationLexicalConflictResult(StrictModel):
    schema_version: Literal[1]
    resolutions: list[LexicalConflictResolution] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def validate_resolutions(self) -> GenerationLexicalConflictResult:
        candidate_ids = [resolution.candidate_id for resolution in self.resolutions]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lexical conflict result resolves a candidate more than once")
        return self


class GenerationLexicalRequest(GenerationStageRequestBase):
    stage: Literal["lexical"] = "lexical"
    task: Literal["lexical"] = "lexical"
    frozen_lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)
    known_terms: list[AgentTermBrief] = Field(default_factory=list, max_length=64)
    target_policy: GenerationTargetPolicy
    language_guidance: list[str]
    repair_lessons: list[LexicalLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationLexicalRequest:
        if len(self.frozen_lessons) != self.lesson_count:
            raise ValueError("lexical request requires exactly lesson_count frozen lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("lexical repair_lessons are valid only on a retry")
        return self


class GenerationTranslationRequest(GenerationStageRequestBase):
    stage: Literal["translation"] = "translation"
    task: Literal["translation"] = "translation"
    frozen_lessons: list[ProseLessonDraft] = Field(min_length=1, max_length=10)
    repair_lessons: list[TranslationLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationTranslationRequest:
        if len(self.frozen_lessons) != self.lesson_count:
            raise ValueError("translation request requires exactly lesson_count frozen lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("translation repair_lessons are valid only on a retry")
        return self


class GenerationGrammarRequest(GenerationStageRequestBase):
    stage: Literal["grammar"] = "grammar"
    task: Literal["grammar"] = "grammar"
    tokenized_lessons: list[TokenizedLessonDraft] = Field(min_length=1, max_length=10)
    grammar_catalog: list[AgentGrammarCatalogEntry] = Field(default_factory=list)
    repair_lessons: list[GrammarLessonDraft] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_lesson_count(self) -> GenerationGrammarRequest:
        if len(self.tokenized_lessons) != self.lesson_count:
            raise ValueError("grammar request requires exactly lesson_count tokenized lessons")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("grammar repair_lessons are valid only on a retry")
        return self


class GenerationCallbackRequest(StrictModel):
    schema_version: Literal[3] = 3
    stage: Literal["legacy", "prose"] = "legacy"
    task: Literal["complete", "prose"] = "complete"
    job_id: str = Field(min_length=3, max_length=100)
    task_id: int = Field(gt=0)
    attempt: int = Field(gt=0)
    lesson_count: int = Field(ge=1, le=10)
    lexical_coverage: Literal["all_lexical_tokens"] = "all_lexical_tokens"
    profile_key: str = Field(min_length=1, max_length=100)
    workspace_path: str = Field(min_length=1, max_length=1000)
    profile_fingerprint: str = Field(min_length=64, max_length=64)
    brief: AgentBrief
    target_policy: GenerationTargetPolicy
    grammar_policy: GenerationGrammarPolicy = Field(default_factory=GenerationGrammarPolicy)
    content_plan: GenerationContentPlan | None = None
    calibration: CalibrationGenerationBrief | None = None
    request_kind: GenerationRequestKind = "queue_fill"
    requested_topic: str | None = Field(default=None, min_length=1, max_length=200)
    latest_feedback: list[FeedbackTag] = Field(default_factory=list, max_length=8)
    previous_failures: list[GenerationFailureContext] = Field(default_factory=list, max_length=4)
    repair_lessons: list[CallbackLessonDraft] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Rejected compact drafts from this task's immediately preceding attempt. On retry, "
            "repair their annotations in place instead of replacing their content."
        ),
    )
    instructions: str

    @model_validator(mode="after")
    def validate_target_candidates(self) -> GenerationCallbackRequest:
        priority_keys = {term.key for term in self.brief.priority_terms}
        unknown = set(self.target_policy.candidate_term_keys) - priority_keys
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"target policy contains terms outside the priority pool: {names}")
        grammar_keys = {construction.key for construction in self.brief.priority_grammar}
        unknown_grammar = set(self.grammar_policy.offered_keys) - grammar_keys
        if unknown_grammar:
            names = ", ".join(sorted(unknown_grammar))
            raise ValueError(
                f"grammar policy contains constructions outside the priority pool: {names}"
            )
        if self.request_kind == "topic_request" and self.requested_topic is None:
            raise ValueError("topic requests require requested_topic")
        if self.request_kind == "queue_fill" and self.requested_topic is not None:
            raise ValueError("queue-fill requests must not include requested_topic")
        if self.calibration is None and self.content_plan is None:
            raise ValueError("ordinary generation requests require content_plan")
        if self.calibration is not None and self.content_plan is not None:
            raise ValueError("calibration generation requests must not include content_plan")
        if self.calibration is not None and (
            self.grammar_policy.preferred_count
            or self.grammar_policy.max_count
            or self.grammar_policy.offered_keys
        ):
            raise ValueError("calibration generation requests must not offer priority grammar")
        if self.repair_lessons and self.attempt <= 1:
            raise ValueError("repair_lessons are valid only on a callback retry")
        if self.calibration is not None and self.stage != "legacy":
            raise ValueError("calibration generation uses the legacy one-pass stage")
        if self.stage == "prose" and self.task != "prose":
            raise ValueError("the prose stage requires the prose task")
        if self.stage == "legacy" and self.task != "complete":
            raise ValueError("the legacy stage requires the complete task")
        return self


class GenerationCallbackResult(StrictModel):
    schema_version: Literal[1]
    lessons: list[CallbackLessonDraft] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_stable_terms(self) -> GenerationCallbackResult:
        terms: dict[str, LessonTerm] = {}
        for lesson in self.lessons:
            for term in lesson.terms:
                previous = terms.get(term.key)
                if previous is not None and previous != term:
                    raise ValueError(
                        f"term key has conflicting definitions across generated lessons: {term.key}"
                    )
                terms[term.key] = term
        return self

    def expanded_lessons(self) -> list[GeneratedLessonDraft]:
        return [lesson.expand() for lesson in self.lessons]


_INTRA_TOKEN_PUNCTUATION = {"'", "’", "-", "‐", "‑"}


def _is_plain_generated_character(character: str) -> bool:
    category = unicodedata.category(character)
    return character.isspace() or category.startswith(("P", "Z"))


def _validate_generated_run_coverage(run: LessonRun) -> None:
    if run.term is None:
        if any(not _is_plain_generated_character(character) for character in run.text):
            raise ValueError(
                "generated plain runs may contain only whitespace and punctuation; annotate every "
                "lexical token with a term"
            )
        return

    has_lexical_character = any(
        unicodedata.category(character)[0] in {"L", "M", "N", "S"} for character in run.text
    )
    if not has_lexical_character:
        raise ValueError("generated term runs must contain a lexical token")
    for index, character in enumerate(run.text):
        if character.isspace() or unicodedata.category(character).startswith("Z"):
            raise ValueError("generated term runs must isolate one token without whitespace")
        if unicodedata.category(character).startswith("P") and (
            character not in _INTRA_TOKEN_PUNCTUATION or index == 0 or index == len(run.text) - 1
        ):
            raise ValueError(
                "generated term runs must keep surrounding punctuation in separate plain runs"
            )
