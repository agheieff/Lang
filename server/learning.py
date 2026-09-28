"""Lesson import, event recording, and deterministic derived learning state."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast, get_args

from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from server.calibration import (
    INITIAL_PROBE_DIFFICULTY,
    cefr_center,
    cefr_for_difficulty,
)
from server.character_word_support import character_adjusted_offer_score
from server.clock import as_utc, utc_now
from server.derived_state import ensure_derived_states
from server.generation_tasks import (
    GenerationMode,
    generation_task_kind,
    generation_task_mode,
    requested_topic,
    topic_request_id,
)
from server.grammar import comfortable_grammar_construction_keys
from server.grammar_catalog import (
    GrammarConstruction,
    find_grammar_catalog,
    grammar_catalog_digest,
)
from server.han import character_tracking_available
from server.language_packs import language_pack, language_pack_digest
from server.learning_policy import TERM_BAND_POLICY, TermBandPolicy, frontier_frequency_rank
from server.learning_units import LearningUnitIndex, build_learning_unit_index
from server.lesson_content import body_sentences, lesson_document
from server.lesson_queue import (
    ready_lessons,
    skipped_lesson_ids,
    unread_lessons,
)
from server.lexeme_learning import (
    DEFAULT_LEXEME_LEARNING_POLICY,
    learner_frontier_rank,
    learner_level_frontier_rank,
)
from server.memory_model import recall_now
from server.models import (
    CharacterState,
    GenerationTask,
    GrammarState,
    Interaction,
    Lesson,
    LessonQueueAction,
    LessonQueueMoveAction,
    LexemeState,
    ProficiencyState,
    Profile,
)
from server.proficiency import rebuild_proficiency_state as _rebuild_proficiency_state
from server.profile_activation import (
    ACTIVATION_PREFERENCE_KEY,
    ONBOARDING_PREFERENCE_KEY,
    PROFILE_LEVEL_SOURCE_KEY,
    profile_is_active,
    questionnaire_seed,
)
from server.reading_preferences import content_history, current_preferences
from server.schemas import (
    AgentBrief,
    AgentGrammarBrief,
    AgentGrammarCatalogEntry,
    AgentLessonBrief,
    AgentTermBrief,
    CalibrationGenerationBrief,
    CefrLevel,
    EventRecordResult,
    FeedbackTag,
    GenerationFailureContext,
    GenerationTargetPolicy,
    InteractionIn,
    LessonDocument,
    LessonSentence,
    LevelSource,
    ProficiencyView,
    ProfileUpdate,
    ProfileView,
    ReaderProgress,
    ReaderState,
    TermKnowledgeBand,
    TextRequestIn,
    is_proper_noun_pos,
)
from server.vocabulary_plan import (
    DEFAULT_TARGET_KNOWN_SHARE,
    KnownShareInputs,
    known_share,
    plan_vocabulary,
    profile_known_share_inputs,
)
from server.word_lists import word_list
from server.workspaces import Workspace

GENERATION_DEDUPE_KEY = "lesson-queue"
GENERATION_CONTRACT_REVISION = 24
DEFAULT_UNREAD_QUEUE_TARGET = 3
MAX_GENERATION_ATTEMPTS = 2
MAX_STAGED_GENERATION_ATTEMPTS = 6
MAX_GENERATION_STAGE_FAILURES = 2
MAX_GENERATION_FAILURE_CONTEXT = 4
MAX_GENERATION_TRIGGER_IDS = 64
GENERATION_FAILURE_STAGES = (
    "prose",
    "lexical",
    "translation",
    "grammar",
    "assembly",
    "legacy",
)
GENERATION_TRIGGER_EVENT_TYPES = (
    "lesson.started",
    "term.revealed",
    "translation.revealed",
    "lesson.completed",
    "lesson.rated",
)
DERIVED_EVIDENCE_EVENT_TYPES = frozenset(
    {"term.revealed", "translation.revealed", "lesson.completed"}
)
PROFICIENCY_EVENT_TYPES = DERIVED_EVIDENCE_EVENT_TYPES | {"lesson.rated"}
MAX_LESSONS_PER_GENERATION_TASK = 1
CALIBRATION_QUEUE_TARGET = 1
CALIBRATION_PROBE_COUNT = 12
CALIBRATION_WORDS_PER_PROBE = 16
CALIBRATION_MIN_WORDS_PER_PROBE = 14
DEFAULT_GENERATED_TEXT_LENGTH = 300
MIN_GENERATED_TEXT_LENGTH = 50
MAX_GENERATED_TEXT_LENGTH = 2_000
WORDS_PER_DELIBERATE_TARGET = 180
MAX_PROTOCOL_TARGETS = 16
MAX_GENERATION_CANDIDATES = 64
DEFAULT_TARGET_KNOWN_RATIO = 0.85
PROFILE_LEVEL_SET_AT_KEY = "level_set_at"
PROFILE_LEVEL_EVIDENCE_CURSOR_KEY = "level_evidence_cursor"


@dataclass(frozen=True)
class _GenerationTriggers:
    event_ids: tuple[str, ...] = ()
    queue_action_ids: tuple[str, ...] = ()
    event_cursor: int = 0
    queue_action_cursor: int = 0


class LessonConflictError(ValueError):
    pass


class EventConflictError(ValueError):
    pass


class TextRequestConflictError(ValueError):
    pass


class ProfileInactiveError(ValueError):
    pass


def ensure_profile(
    db: Session,
    *,
    learning_language: str = "es-ES",
    translation_language: str = "en",
    activated: bool | None = None,
) -> Profile:
    profile = db.get(Profile, 1)
    if profile is None:
        profile = Profile(
            id=1,
            learning_language=learning_language,
            translation_language=translation_language,
            level=cefr_for_difficulty(INITIAL_PROBE_DIFFICULTY),
            difficulty=INITIAL_PROBE_DIFFICULTY,
            preferences={
                PROFILE_LEVEL_SOURCE_KEY: "unknown",
                ACTIVATION_PREFERENCE_KEY: {
                    "schema_version": 1,
                    "active": True if activated is None else activated,
                    "activated_at": utc_now().isoformat() if activated else None,
                },
            },
        )
        db.add(profile)
    else:
        preferences = dict(profile.preferences)
        if PROFILE_LEVEL_SOURCE_KEY not in preferences:
            preferences[PROFILE_LEVEL_SOURCE_KEY] = "self_reported"
        if (
            preferences[PROFILE_LEVEL_SOURCE_KEY] == "self_reported"
            and PROFILE_LEVEL_SET_AT_KEY not in preferences
        ):
            preferences[PROFILE_LEVEL_SET_AT_KEY] = as_utc(profile.updated_at).isoformat()
        if (
            preferences[PROFILE_LEVEL_SOURCE_KEY] == "self_reported"
            and PROFILE_LEVEL_EVIDENCE_CURSOR_KEY not in preferences
        ):
            anchor = _stored_level_anchor(preferences, profile.updated_at)
            preferences[PROFILE_LEVEL_EVIDENCE_CURSOR_KEY] = (
                db.scalar(select(func.max(Interaction.id)).where(Interaction.occurred_at <= anchor))
                or 0
            )
        if ACTIVATION_PREFERENCE_KEY not in preferences:
            has_lessons = bool(db.scalar(select(func.count()).select_from(Lesson)))
            preferences[ACTIVATION_PREFERENCE_KEY] = {
                "schema_version": 1,
                "active": has_lessons,
                "activated_at": None,
                "migrated": True,
            }
        if preferences != profile.preferences:
            profile.preferences = preferences
    if db.get(ProficiencyState, 1) is None:
        db.add(ProficiencyState(id=1))
    if db.new or db.dirty:
        db.commit()
        db.refresh(profile)
    return profile


def update_profile(db: Session, patch: ProfileUpdate | Mapping[str, Any]) -> Profile:
    update = patch if isinstance(patch, ProfileUpdate) else ProfileUpdate.model_validate(patch)
    profile = ensure_profile(db)
    values = update.model_dump(exclude_none=True)
    level_changed = update.level is not None or update.difficulty is not None
    reserved = {
        key: profile.preferences[key]
        for key in (ACTIVATION_PREFERENCE_KEY, ONBOARDING_PREFERENCE_KEY)
        if key in profile.preferences
    }
    if level_changed:
        values["level"] = (
            update.level
            if update.level is not None
            else cefr_for_difficulty(cast(float, update.difficulty))
        )
        values["difficulty"] = (
            update.difficulty
            if update.difficulty is not None
            else cefr_center(cast(CefrLevel, update.level))
        )
        preferences = dict(profile.preferences)
        preferences.update(update.preferences or {})
        preferences.update(reserved)
        preferences[PROFILE_LEVEL_SOURCE_KEY] = "self_reported"
        preferences[PROFILE_LEVEL_SET_AT_KEY] = utc_now().isoformat()
        preferences[PROFILE_LEVEL_EVIDENCE_CURSOR_KEY] = (
            db.scalar(select(func.max(Interaction.id))) or 0
        )
        values["preferences"] = preferences
    elif update.preferences is not None:
        preferences = dict(profile.preferences)
        preferences.update(update.preferences)
        preferences.update(reserved)
        preferences[PROFILE_LEVEL_SOURCE_KEY] = profile_level_source(profile)
        values["preferences"] = preferences
    for key, value in values.items():
        setattr(profile, key, value)
    db.commit()
    db.refresh(profile)
    if level_changed:
        rebuild_proficiency_state(db)
        db.refresh(profile)
    return profile


def profile_level_source(profile: Profile) -> LevelSource:
    value = profile.preferences.get(PROFILE_LEVEL_SOURCE_KEY)
    if value not in {"unknown", "self_reported", "estimated"}:
        raise ValueError("profile has an invalid level source")
    return cast(LevelSource, value)


def validate_lesson(payload: LessonDocument | Mapping[str, Any]) -> LessonDocument:
    if isinstance(payload, LessonDocument):
        return payload
    return LessonDocument.model_validate(payload)


def import_lesson(
    db: Session,
    payload: LessonDocument | Mapping[str, Any],
    *,
    source_path: str | None = None,
    replace: bool = False,
) -> Lesson:
    document = validate_lesson(payload)
    profile = ensure_profile(db)
    if (
        document.learning_language != profile.learning_language
        or document.translation_language != profile.translation_language
    ):
        raise ValueError("lesson languages must match the local profile")

    _validate_term_keys_across_lessons(db, document)
    serialized = document.model_dump(mode="json")
    if serialized["calibration"] is None:
        del serialized["calibration"]
    canonical = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(canonical.encode()).hexdigest()
    existing = db.scalar(select(Lesson).where(Lesson.key == document.key))

    if existing is not None:
        if existing.content_hash == content_hash:
            _ensure_lesson_audio(db, existing)
            return existing
        if not replace:
            raise LessonConflictError(
                f"lesson already exists with different content: {document.key}"
            )
        has_events = db.scalar(
            select(func.count())
            .select_from(Interaction)
            .where(Interaction.lesson_id == existing.id)
        )
        if has_events:
            raise LessonConflictError("cannot replace a lesson after interactions were recorded")
        has_queue_actions = db.scalar(
            select(func.count())
            .select_from(LessonQueueAction)
            .where(LessonQueueAction.lesson_id == existing.id)
        )
        if has_queue_actions:
            raise LessonConflictError("cannot replace a lesson after queue actions were recorded")
        has_queue_moves = db.scalar(
            select(func.count())
            .select_from(LessonQueueMoveAction)
            .where(
                or_(
                    LessonQueueMoveAction.lesson_id == existing.id,
                    LessonQueueMoveAction.neighbor_lesson_id == existing.id,
                )
            )
        )
        if has_queue_moves:
            raise LessonConflictError("cannot replace a lesson after queue moves were recorded")
        lesson = existing
    else:
        lesson = Lesson(key=document.key)
        db.add(lesson)

    lesson.schema_version = document.schema_version
    lesson.title = document.title
    lesson.learning_language = document.learning_language
    lesson.translation_language = document.translation_language
    lesson.topic = document.topic
    lesson.level = document.level
    lesson.difficulty = document.difficulty
    lesson.payload = serialized
    lesson.metadata_json = document.metadata
    lesson.content_hash = content_hash
    lesson.source_path = source_path
    lesson.updated_at = utc_now()
    db.commit()
    db.refresh(lesson)
    ensure_derived_states(db)
    if lesson.known_share_at_import is None:
        # Keep updated_at unchanged: it is part of the derived-state watermark.
        db.execute(
            update(Lesson)
            .where(Lesson.id == lesson.id)
            .values(
                known_share_at_import=known_share(document, profile_known_share_inputs(db)),
                updated_at=lesson.updated_at,
            )
        )
        db.commit()
        db.refresh(lesson)
    _ensure_lesson_audio(db, lesson)
    return lesson


def _ensure_lesson_audio(db: Session, lesson: Lesson) -> None:
    # The optional provider is isolated, but its small durable queue belongs to every import path.
    from server.tts import ensure_lesson_audio_task, model_language

    profile = ensure_profile(db)
    if profile_is_active(profile) and model_language(lesson.learning_language) is not None:
        ensure_lesson_audio_task(db, lesson)


def get_reader_state(
    db: Session, lesson_id: int | None = None, *, fresh: bool = False
) -> ReaderState:
    profile = ensure_profile(db)
    if lesson_id is None:
        lesson = next(iter(unread_lessons(db)), None)
    else:
        lesson = db.scalar(
            select(Lesson).where(
                Lesson.id == lesson_id,
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
        )
        if lesson is None:
            raise LookupError(f"lesson not found: {lesson_id}")

    profile_view = build_profile_view(db, profile)
    generation_status = _generation_status(db)
    if lesson is None:
        return ReaderState(profile=profile_view, generation_status=generation_status)

    progress = ReaderProgress()
    latest = None
    if not fresh:
        latest = db.scalar(
            select(Interaction)
            .where(Interaction.lesson_id == lesson.id)
            .order_by(Interaction.occurred_at.desc(), Interaction.id.desc())
            .limit(1)
        )
    if latest is not None:
        events = db.scalars(
            select(Interaction)
            .where(
                Interaction.lesson_id == lesson.id,
                Interaction.session_id == latest.session_id,
            )
            .order_by(Interaction.occurred_at, Interaction.id)
        ).all()
        progress = _reader_progress(latest.session_id, events)

    document = lesson_document(lesson)
    catalog = find_grammar_catalog(document.learning_language)
    used_grammar = document.grammar_construction_keys()
    return ReaderState(
        profile=profile_view,
        lesson_id=lesson.id,
        lesson=document,
        progress=progress,
        term_bands=build_reader_term_bands(db, document, profile_view),
        grammar_catalog=[
            _agent_grammar_catalog_entry(construction)
            for construction in (catalog.constructions if catalog is not None else ())
            if construction.key in used_grammar
        ],
        comfortable_grammar_construction_keys=comfortable_grammar_construction_keys(
            db,
            learning_language=document.learning_language,
            translation_language=document.translation_language,
            construction_keys=used_grammar,
        ),
        generation_status=generation_status,
    )


def _profile_learning_unit_index(
    db: Session,
    *,
    document: LessonDocument,
) -> LearningUnitIndex:
    """Build a stable profile index, replacing any stored copy of ``document``."""

    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == document.learning_language,
            Lesson.translation_language == document.translation_language,
            Lesson.key != document.key,
        )
        .order_by(Lesson.imported_at, Lesson.id)
    ).all()
    documents = [lesson_document(lesson) for lesson in lessons]
    documents.append(document)
    return build_learning_unit_index(documents)


def build_reader_term_bands(
    db: Session,
    document: LessonDocument,
    profile: ProfileView,
    *,
    policy: TermBandPolicy = TERM_BAND_POLICY,
) -> dict[str, TermKnowledgeBand]:
    """Classify current lesson terms for styling without treating a band as evidence."""

    catalog = document.term_catalog()
    units = _profile_learning_unit_index(db, document=document)
    canonical_keys = {key for key in (units.resolve(key) for key in catalog) if key is not None}
    states = {
        state.term_key: state
        for state in db.scalars(
            select(LexemeState).where(
                LexemeState.learning_language == document.learning_language,
                LexemeState.translation_language == document.translation_language,
                LexemeState.term_key.in_(canonical_keys),
            )
        )
    }
    expected_rank = _expected_frequency_rank(profile, policy)
    targets = {
        key
        for key in (units.resolve(target) for target in document.target_term_keys)
        if key is not None
    }
    bands: dict[str, TermKnowledgeBand] = {}
    for display_key, display_term in catalog.items():
        canonical = units.resolve(display_key)
        if canonical is None:
            bands[display_key] = "incidental"
            continue
        term = units.definitions.get(canonical, display_term)
        bands[display_key] = _reader_term_band(
            states.get(canonical),
            frequency_rank=term.frequency_rank,
            part_of_speech=term.pos,
            deliberate_target=canonical in targets,
            expected_rank=expected_rank,
            policy=policy,
        )
    return bands


def _reader_term_band(
    state: LexemeState | None,
    *,
    frequency_rank: int | None,
    part_of_speech: str,
    deliberate_target: bool,
    expected_rank: int,
    policy: TermBandPolicy,
) -> TermKnowledgeBand:
    if state is not None and _is_familiar(state, policy):
        return "familiar"
    if deliberate_target:
        return "focus"
    if state is not None and state.reveal_failures >= policy.repeated_reveal_threshold:
        return "uncertain"
    if state is not None and (
        state.qualified_exposures >= policy.expected_min_exposures
        and state.mastery >= policy.expected_min_mastery
    ):
        return "expected"
    if frequency_rank is None:
        if is_proper_noun_pos(part_of_speech):
            return "incidental"
        return "uncertain"
    if frequency_rank <= expected_rank:
        return "expected"
    if frequency_rank <= expected_rank * policy.uncertain_frequency_multiplier:
        return "uncertain"
    return "incidental"


def _is_familiar(state: LexemeState, policy: TermBandPolicy) -> bool:
    if (
        state.reveal_failures == 0
        and state.qualified_exposures >= policy.familiar_min_exposures
        and state.mastery >= policy.familiar_min_mastery
    ):
        return True
    return (
        state.qualified_exposures >= policy.familiar_with_failures_min_exposures
        and state.mastery >= policy.familiar_with_failures_min_mastery
    )


def _expected_frequency_rank(profile: ProfileView, policy: TermBandPolicy) -> int:
    return frontier_frequency_rank(profile.difficulty, profile.proficiency.lower, policy)


def record_events(
    db: Session,
    events: Sequence[InteractionIn | Mapping[str, Any]],
    *,
    rebuild_derived: bool = True,
) -> EventRecordResult:
    """Append events; the HTTP path defers the derived-cache replay off the request."""

    parsed = [
        event if isinstance(event, InteractionIn) else InteractionIn.model_validate(event)
        for event in events
    ]
    if not parsed:
        return EventRecordResult(accepted=0, duplicates=0, accepted_event_ids=[])

    profile = ensure_profile(db)
    lesson_ids = {event.lesson_id for event in parsed}
    lessons = {
        lesson.id: lesson
        for lesson in db.scalars(
            select(Lesson).where(
                Lesson.id.in_(lesson_ids),
                Lesson.learning_language == profile.learning_language,
                Lesson.translation_language == profile.translation_language,
            )
        ).all()
    }
    missing = lesson_ids - lessons.keys()
    if missing:
        raise LookupError(f"lessons not found: {sorted(missing)}")

    unique_events: dict[str, InteractionIn] = {}
    duplicates = 0
    for event in parsed:
        previous = unique_events.get(event.event_id)
        if previous is None:
            unique_events[event.event_id] = event
        elif _input_event_fingerprint(previous) != _input_event_fingerprint(event):
            raise EventConflictError(f"event_id reused with different content: {event.event_id}")
        else:
            duplicates += 1

    existing = {
        event.event_id: event
        for event in db.scalars(
            select(Interaction).where(Interaction.event_id.in_(unique_events))
        ).all()
    }
    for event_id, stored in existing.items():
        incoming = unique_events[event_id]
        if _stored_event_fingerprint(stored) != _input_event_fingerprint(incoming):
            raise EventConflictError(f"event_id reused with different content: {event_id}")
        duplicates += 1

    new_events = [event for event in unique_events.values() if event.event_id not in existing]
    for event in new_events:
        lesson_document = LessonDocument.model_validate(lessons[event.lesson_id].payload)
        _validate_event_references(event, lesson_document)

    accepted_ids: list[str] = []
    for event in new_events:
        statement = sqlite_insert(Interaction).values(
            event_id=event.event_id,
            session_id=event.session_id,
            lesson_id=event.lesson_id,
            event_type=event.type,
            occurred_at=event.occurred_at,
            payload=event.payload,
        )
        inserted_id = db.scalar(
            statement.on_conflict_do_nothing(index_elements=["event_id"]).returning(
                Interaction.event_id
            )
        )
        if inserted_id is not None:
            accepted_ids.append(event.event_id)
        else:
            concurrent_stored = db.scalar(
                select(Interaction).where(Interaction.event_id == event.event_id)
            )
            if concurrent_stored is None or _stored_event_fingerprint(
                concurrent_stored
            ) != _input_event_fingerprint(event):
                db.rollback()
                raise EventConflictError(
                    f"event_id reused with different content: {event.event_id}"
                )
            duplicates += 1
    db.commit()

    accepted_id_set = set(accepted_ids)
    accepted_types = {event.type for event in new_events if event.event_id in accepted_id_set}
    replay_types = accepted_types or {event.type for event in unique_events.values()}
    if rebuild_derived and replay_types & PROFICIENCY_EVENT_TYPES:
        ensure_derived_states(db)
    generation_triggers = [
        event.event_id
        for event in new_events
        if event.event_id in accepted_id_set and event.type in GENERATION_TRIGGER_EVENT_TYPES
    ]
    if generation_triggers:
        ensure_generation_task(
            db,
            trigger_event_ids=generation_triggers,
            proficiency_current=rebuild_derived,
        )
    return EventRecordResult(
        accepted=len(accepted_ids),
        duplicates=duplicates,
        accepted_event_ids=accepted_ids,
    )


def auto_generation_enabled() -> bool:
    """Enabled unless explicitly disabled, matching the documented `pnpm dev` default."""

    return os.getenv("ARC_LANG_AUTO_GENERATE", "1") != "0"


def unread_queue_target() -> int:
    raw = os.getenv("ARC_LANG_QUEUE_TARGET", str(DEFAULT_UNREAD_QUEUE_TARGET))
    try:
        target = int(raw)
    except ValueError as error:
        raise ValueError("ARC_LANG_QUEUE_TARGET must be an integer") from error
    if not 1 <= target <= 10:
        raise ValueError("ARC_LANG_QUEUE_TARGET must be between 1 and 10")
    return target


def unread_generation_lesson_count(db: Session, mode: GenerationMode) -> int:
    candidates = unread_lessons(db) if mode == "calibration" else ready_lessons(db)
    return sum(
        lesson_document(lesson).calibration is not None
        if mode == "calibration"
        else lesson_document(lesson).calibration is None
        for lesson in candidates
    )


def generation_mode(profile: Profile) -> GenerationMode:
    return "calibration" if profile_level_source(profile) == "unknown" else "lesson"


def request_topic_lesson(db: Session, request: TextRequestIn) -> GenerationTask:
    """Durably enqueue one extra lesson, independently of the automatic unread target."""

    profile = ensure_profile(db)
    if not profile_is_active(profile):
        raise ProfileInactiveError("activate this language before requesting a text")
    request_id = str(request.request_id)
    dedupe_key = f"topic-request:{request_id}"
    existing = db.scalar(
        select(GenerationTask).where(GenerationTask.dedupe_key == dedupe_key).limit(1)
    )
    if existing is not None:
        return _matching_topic_request(existing, request_id=request_id, topic=request.topic)

    key, workspace_path = _workspace_key_and_path(db, profile)
    task = GenerationTask(
        state="pending",
        dedupe_key=dedupe_key,
        payload={
            "schema_version": 1,
            "generation_contract_revision": GENERATION_CONTRACT_REVISION,
            "request_kind": "topic_request",
            "request_id": request_id,
            "requested_topic": request.topic,
            "needed_lesson_count": 1,
            "generation_mode": "lesson",
            "learning_language": profile.learning_language,
            "translation_language": profile.translation_language,
            "profile_key": key,
            "workspace_path": workspace_path,
            "profile_fingerprint": profile_fingerprint(db, profile),
            "trigger_event_ids": [],
            "latest_feedback": latest_profile_feedback(db),
        },
    )
    db.add(task)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        concurrent = db.scalar(
            select(GenerationTask).where(GenerationTask.dedupe_key == dedupe_key).limit(1)
        )
        if concurrent is None:
            raise
        return _matching_topic_request(concurrent, request_id=request_id, topic=request.topic)
    db.refresh(task)
    return task


def _matching_topic_request(task: GenerationTask, *, request_id: str, topic: str) -> GenerationTask:
    if (
        generation_task_kind(task) != "topic_request"
        or topic_request_id(task) != request_id
        or requested_topic(task) != topic
    ):
        raise TextRequestConflictError("request_id reused with different topic request data")
    return task


def generation_failure_context(task: GenerationTask) -> list[GenerationFailureContext]:
    """Return bounded, validated diagnostics from earlier attempts and legacy task rows."""

    failures: list[GenerationFailureContext] = []
    raw_failures = task.payload.get("previous_failures")
    if isinstance(raw_failures, list):
        for raw_failure in raw_failures:
            try:
                failure = GenerationFailureContext.model_validate(raw_failure)
            except ValueError:
                continue
            if failure not in failures:
                failures.append(failure)
    if task.state == "failed" and isinstance(task.error, str) and task.error:
        legacy = GenerationFailureContext(
            task_id=task.id,
            attempt=max(1, task.attempts),
            error=task.error[:4000],
        )
        if legacy not in failures:
            failures.append(legacy)
    return failures[-MAX_GENERATION_FAILURE_CONTEXT:]


def _generation_failure_buckets(error: str) -> tuple[str, ...]:
    stages = tuple(stage for stage in GENERATION_FAILURE_STAGES if f"[{stage}]" in error)
    return stages or ("task",)


def _generation_stage_failure_counts(task: GenerationTask) -> Counter[str]:
    raw_counts = task.payload.get("generation_stage_failures")
    if isinstance(raw_counts, dict):
        stored_counts = Counter(
            {
                stage: count
                for stage, count in raw_counts.items()
                if (
                    isinstance(stage, str)
                    and (stage in GENERATION_FAILURE_STAGES or stage == "task")
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count > 0
                )
            }
        )
        if stored_counts:
            return stored_counts

    derived_counts: Counter[str] = Counter()
    for failure in generation_failure_context(task):
        if failure.task_id == task.id:
            derived_counts.update(_generation_failure_buckets(failure.error))
    return derived_counts


def _generation_attempt_limit(task: GenerationTask) -> int:
    if generation_task_mode(task) == "calibration":
        return MAX_GENERATION_ATTEMPTS
    return MAX_STAGED_GENERATION_ATTEMPTS


def _generation_retry_available(task: GenerationTask) -> bool:
    if task.attempts >= _generation_attempt_limit(task):
        return False
    if task.payload.get("generation_contract_revision") != GENERATION_CONTRACT_REVISION:
        return True
    counts = _generation_stage_failure_counts(task)
    return not counts or max(counts.values()) < MAX_GENERATION_STAGE_FAILURES


def _append_generation_failure(task: GenerationTask, error: str) -> str:
    stored_error = error[:4000] or "generation failed without an error message"
    failures = generation_failure_context(task)
    current_failure = GenerationFailureContext(
        task_id=task.id,
        attempt=max(1, task.attempts),
        error=stored_error,
    )
    if current_failure not in failures:
        failures.append(current_failure)
    payload = dict(task.payload)
    counts = _generation_stage_failure_counts(task)
    if current_failure not in generation_failure_context(task):
        counts.update(_generation_failure_buckets(stored_error))
    payload["generation_stage_failures"] = dict(sorted(counts.items()))
    payload["previous_failures"] = [
        failure.model_dump(mode="json") for failure in failures[-MAX_GENERATION_FAILURE_CONTEXT:]
    ]
    task.payload = payload
    return stored_error


def _resolve_generation_triggers(
    db: Session,
    *,
    event_ids: Sequence[str] = (),
    queue_action_ids: Sequence[str] = (),
    require_all: bool = True,
) -> _GenerationTriggers:
    normalized_event_ids = _normalized_trigger_ids(event_ids)
    normalized_action_ids = _normalized_trigger_ids(queue_action_ids)
    event_rows = {
        event_id: row_id
        for event_id, row_id in db.execute(
            select(Interaction.event_id, Interaction.id).where(
                Interaction.event_id.in_(normalized_event_ids)
            )
        ).all()
    }
    action_rows = {
        action_id: row_id
        for action_id, row_id in db.execute(
            select(LessonQueueAction.action_id, LessonQueueAction.id).where(
                LessonQueueAction.action_id.in_(normalized_action_ids)
            )
        ).all()
    }
    if require_all:
        missing_events = set(normalized_event_ids) - event_rows.keys()
        missing_actions = set(normalized_action_ids) - action_rows.keys()
        if missing_events or missing_actions:
            raise ValueError("generation triggers must reference accepted events or queue actions")
    return _GenerationTriggers(
        event_ids=tuple(event_id for event_id in normalized_event_ids if event_id in event_rows),
        queue_action_ids=tuple(
            action_id for action_id in normalized_action_ids if action_id in action_rows
        ),
        event_cursor=max(event_rows.values(), default=0),
        queue_action_cursor=max(action_rows.values(), default=0),
    )


def _normalized_trigger_ids(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("generation trigger IDs must be non-empty strings")
        if value not in normalized:
            normalized.append(value)
    return normalized


def _task_generation_triggers(
    db: Session, task: GenerationTask, *, deferred: bool = False
) -> _GenerationTriggers:
    prefix = "deferred_" if deferred else ""
    event_ids = _payload_trigger_ids(task.payload.get(f"{prefix}trigger_event_ids"))
    action_ids = _payload_trigger_ids(task.payload.get(f"{prefix}trigger_queue_action_ids"))
    resolved = _resolve_generation_triggers(
        db,
        event_ids=event_ids,
        queue_action_ids=action_ids,
        require_all=False,
    )
    return _GenerationTriggers(
        event_ids=resolved.event_ids,
        queue_action_ids=resolved.queue_action_ids,
        event_cursor=max(
            resolved.event_cursor,
            _payload_trigger_cursor(task.payload.get(f"{prefix}trigger_event_cursor")),
        ),
        queue_action_cursor=max(
            resolved.queue_action_cursor,
            _payload_trigger_cursor(task.payload.get(f"{prefix}trigger_queue_action_cursor")),
        ),
    )


def _payload_trigger_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _payload_trigger_cursor(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _combine_generation_triggers(*groups: _GenerationTriggers) -> _GenerationTriggers:
    event_ids = _bounded_trigger_ids(item for group in groups for item in group.event_ids)
    action_ids = _bounded_trigger_ids(item for group in groups for item in group.queue_action_ids)
    return _GenerationTriggers(
        event_ids=event_ids,
        queue_action_ids=action_ids,
        event_cursor=max((group.event_cursor for group in groups), default=0),
        queue_action_cursor=max((group.queue_action_cursor for group in groups), default=0),
    )


def _bounded_trigger_ids(values: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        if value in unique:
            unique.remove(value)
        unique.append(value)
    return tuple(unique[-MAX_GENERATION_TRIGGER_IDS:])


def _write_task_generation_triggers(
    task: GenerationTask, triggers: _GenerationTriggers, *, deferred: bool = False
) -> bool:
    prefix = "deferred_" if deferred else ""
    payload = dict(task.payload)
    values: dict[str, object] = {
        f"{prefix}trigger_event_ids": list(triggers.event_ids),
        f"{prefix}trigger_queue_action_ids": list(triggers.queue_action_ids),
        f"{prefix}trigger_event_cursor": triggers.event_cursor,
        f"{prefix}trigger_queue_action_cursor": triggers.queue_action_cursor,
    }
    if all(payload.get(key) == value for key, value in values.items()):
        return False
    payload.update(values)
    task.payload = payload
    return True


def _merge_task_generation_triggers(
    db: Session,
    task: GenerationTask,
    incoming: _GenerationTriggers,
    *,
    deferred: bool,
) -> bool:
    current = _task_generation_triggers(db, task, deferred=deferred)
    return _write_task_generation_triggers(
        task,
        _combine_generation_triggers(current, incoming),
        deferred=deferred,
    )


def _has_fresh_generation_trigger(
    candidate: _GenerationTriggers, consumed: _GenerationTriggers
) -> bool:
    return (
        candidate.event_cursor > consumed.event_cursor
        or candidate.queue_action_cursor > consumed.queue_action_cursor
    )


def _generation_trigger_high_water(db: Session, after: _GenerationTriggers) -> _GenerationTriggers:
    """Return committed generation evidence newer than a task's durable cursors."""

    event_rows = db.execute(
        select(Interaction.event_id, Interaction.id)
        .where(
            Interaction.event_type.in_(GENERATION_TRIGGER_EVENT_TYPES),
            Interaction.id > after.event_cursor,
        )
        .order_by(Interaction.id.desc())
        .limit(MAX_GENERATION_TRIGGER_IDS)
    ).all()
    action_rows = db.execute(
        select(LessonQueueAction.action_id, LessonQueueAction.id)
        .where(
            LessonQueueAction.skipped.is_(True),
            LessonQueueAction.id > after.queue_action_cursor,
        )
        .order_by(LessonQueueAction.id.desc())
        .limit(MAX_GENERATION_TRIGGER_IDS)
    ).all()
    return _GenerationTriggers(
        event_ids=tuple(row.event_id for row in reversed(event_rows)),
        queue_action_ids=tuple(row.action_id for row in reversed(action_rows)),
        event_cursor=event_rows[0].id if event_rows else 0,
        queue_action_cursor=action_rows[0].id if action_rows else 0,
    )


def ensure_generation_task(
    db: Session,
    *,
    trigger_event_ids: Sequence[str] = (),
    trigger_queue_action_ids: Sequence[str] = (),
    require_enabled: bool = True,
    proficiency_current: bool = False,
) -> GenerationTask | None:
    if require_enabled and not auto_generation_enabled():
        return None

    profile = ensure_profile(db)
    if not profile_is_active(profile):
        return None
    if not proficiency_current and profile_level_source(profile) == "unknown":
        # Finishing calibration placement can change the generation mode itself.
        ensure_derived_states(db)
    mode = generation_mode(profile)
    target = CALIBRATION_QUEUE_TARGET if mode == "calibration" else unread_queue_target()
    unread = unread_generation_lesson_count(db, mode)
    if unread >= target:
        return None

    explicit_triggers = _resolve_generation_triggers(
        db,
        event_ids=trigger_event_ids,
        queue_action_ids=trigger_queue_action_ids,
    )

    outstanding = db.scalar(
        select(GenerationTask)
        .where(
            GenerationTask.dedupe_key == GENERATION_DEDUPE_KEY,
            GenerationTask.state.in_(("pending", "running")),
        )
        .order_by(GenerationTask.id)
        .limit(1)
    )
    if outstanding is not None:
        baseline = _combine_generation_triggers(
            _task_generation_triggers(db, outstanding),
            _task_generation_triggers(db, outstanding, deferred=True),
        )
        observed_triggers = _combine_generation_triggers(
            _generation_trigger_high_water(db, baseline),
            explicit_triggers,
        )
        changed = _merge_task_generation_triggers(
            db,
            outstanding,
            observed_triggers,
            deferred=outstanding.state == "running",
        )
        if changed:
            db.commit()
            db.refresh(outstanding)
        return outstanding
    if not proficiency_current:
        ensure_derived_states(db)
    latest = next(
        (
            task
            for task in db.scalars(select(GenerationTask).order_by(GenerationTask.id.desc())).all()
            if generation_task_kind(task) == "queue_fill"
        ),
        None,
    )
    baseline = (
        _combine_generation_triggers(
            _task_generation_triggers(db, latest),
            _task_generation_triggers(db, latest, deferred=True),
        )
        if latest is not None
        else _GenerationTriggers()
    )
    observed_triggers = _combine_generation_triggers(
        _generation_trigger_high_water(db, baseline),
        explicit_triggers,
    )
    fingerprint = profile_fingerprint(db, profile)
    task_triggers = observed_triggers
    previous_failures: list[GenerationFailureContext] = []
    if latest is not None and latest.state == "failed":
        consumed = _task_generation_triggers(db, latest)
        deferred = _task_generation_triggers(db, latest, deferred=True)
        same_fingerprint = latest.payload.get("profile_fingerprint") == fingerprint
        same_contract = (
            latest.payload.get("generation_contract_revision") == GENERATION_CONTRACT_REVISION
        )
        if same_fingerprint:
            previous_failures = generation_failure_context(latest)
        if same_fingerprint and same_contract:
            if _generation_retry_available(latest):
                if _merge_task_generation_triggers(
                    db,
                    latest,
                    observed_triggers,
                    deferred=False,
                ):
                    db.commit()
                    db.refresh(latest)
                return latest
            unconsumed = _combine_generation_triggers(deferred, observed_triggers)
            if not _has_fresh_generation_trigger(unconsumed, consumed):
                return latest
        task_triggers = _combine_generation_triggers(consumed, deferred, observed_triggers)

    key, workspace_path = _workspace_key_and_path(db, profile)
    task = GenerationTask(
        state="pending",
        dedupe_key=GENERATION_DEDUPE_KEY,
        payload={
            "schema_version": 1,
            "generation_contract_revision": GENERATION_CONTRACT_REVISION,
            "request_kind": "queue_fill",
            "queue_target": target,
            "unread_lesson_count": unread,
            "needed_lesson_count": min(MAX_LESSONS_PER_GENERATION_TASK, target - unread),
            "queue_shortfall": target - unread,
            "generation_mode": mode,
            "learning_language": profile.learning_language,
            "translation_language": profile.translation_language,
            "profile_key": key,
            "workspace_path": workspace_path,
            "profile_fingerprint": fingerprint,
            "trigger_event_ids": list(task_triggers.event_ids),
            "trigger_queue_action_ids": list(task_triggers.queue_action_ids),
            "trigger_event_cursor": task_triggers.event_cursor,
            "trigger_queue_action_cursor": task_triggers.queue_action_cursor,
            "previous_failures": [failure.model_dump(mode="json") for failure in previous_failures],
            "latest_feedback": latest_profile_feedback(db),
        },
    )
    db.add(task)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return db.scalar(
            select(GenerationTask)
            .where(
                GenerationTask.dedupe_key == GENERATION_DEDUPE_KEY,
                GenerationTask.state.in_(("pending", "running")),
            )
            .order_by(GenerationTask.id)
            .limit(1)
        )
    db.refresh(task)
    return task


def maintain_generation_task(db: Session) -> GenerationTask | None:
    """Fill the queue or retry the failed DAG stage within its bounded repair budget."""

    if not profile_is_active(ensure_profile(db)):
        return None
    task = ensure_generation_task(db, require_enabled=False)
    if task is not None and task.state == "failed" and _generation_retry_available(task):
        return retry_generation_task(db, task.id)
    return task


def maintain_topic_generation_tasks(db: Session) -> list[GenerationTask]:
    """Retry eligible failed explicit requests without serializing them with queue work."""

    if not profile_is_active(ensure_profile(db)):
        return []
    retried: list[GenerationTask] = []
    tasks = db.scalars(
        select(GenerationTask)
        .where(GenerationTask.state == "failed")
        .order_by(GenerationTask.created_at, GenerationTask.id)
    ).all()
    for task in tasks:
        if generation_task_kind(task) == "topic_request" and _generation_retry_available(task):
            retried.append(retry_generation_task(db, task.id))
    return retried


def claim_generation_task(db: Session) -> GenerationTask | None:
    if not profile_is_active(ensure_profile(db)):
        return None
    while True:
        task = db.scalar(
            select(GenerationTask)
            .where(GenerationTask.state == "pending")
            .order_by(GenerationTask.created_at, GenerationTask.id)
            .limit(1)
        )
        if task is None:
            return None
        if _generation_retry_available(task):
            break
        now = utc_now()
        task.state = "failed"
        if generation_task_kind(task) == "queue_fill":
            task.dedupe_key = None
        task.error = _append_generation_failure(
            task,
            task.error or "generation task reached the maximum attempt count before claim",
        )
        task.started_at = None
        task.finished_at = now
        task.updated_at = now
        db.commit()
    now = utc_now()
    task.state = "running"
    task.attempts += 1
    task.started_at = now
    task.finished_at = None
    task.updated_at = now
    task.error = None
    db.commit()
    db.refresh(task)
    return task


def recover_running_generation_tasks(db: Session) -> int:
    tasks = db.scalars(select(GenerationTask).where(GenerationTask.state == "running")).all()
    now = utc_now()
    for task in tasks:
        task.error = _append_generation_failure(
            task,
            "worker stopped before reporting a result",
        )
        terminal = not _generation_retry_available(task)
        task.state = "failed" if terminal else "pending"
        if terminal and generation_task_kind(task) == "queue_fill":
            task.dedupe_key = None
        task.started_at = None
        task.finished_at = now if terminal else None
        task.updated_at = now
    db.commit()
    return len(tasks)


def complete_generation_task(db: Session, task_id: int, *, log_path: str) -> GenerationTask:
    task = _running_generation_task(db, task_id)
    now = utc_now()
    task.state = "completed"
    if generation_task_kind(task) == "queue_fill":
        task.dedupe_key = None
    task.log_path = log_path
    task.error = None
    task.finished_at = now
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def fail_generation_task(db: Session, task_id: int, *, log_path: str, error: str) -> GenerationTask:
    task = _running_generation_task(db, task_id)
    now = utc_now()
    stored_error = _append_generation_failure(task, error)
    task.state = "failed"
    if generation_task_kind(task) == "queue_fill":
        task.dedupe_key = None
    task.log_path = log_path
    task.error = stored_error
    task.finished_at = now
    task.updated_at = now
    db.commit()
    db.refresh(task)
    return task


def retry_generation_task(db: Session, task_id: int) -> GenerationTask:
    if not profile_is_active(ensure_profile(db)):
        raise ProfileInactiveError("activate this language before retrying generation")
    task = db.get(GenerationTask, task_id)
    if task is None:
        raise LookupError(f"generation task not found: {task_id}")
    if task.state != "failed":
        raise ValueError("only failed generation tasks can be retried")
    if not _generation_retry_available(task):
        raise ValueError("generation task reached the maximum attempt count for this stage")
    now = utc_now()
    contract_changed = (
        task.payload.get("generation_contract_revision") != GENERATION_CONTRACT_REVISION
    )
    failures = [] if contract_changed else generation_failure_context(task)
    if contract_changed:
        payload = dict(task.payload)
        payload["generation_contract_revision"] = GENERATION_CONTRACT_REVISION
        payload["generation_stage_failures"] = {}
        payload["previous_failures"] = []
        task.payload = payload
    elif failures:
        payload = dict(task.payload)
        payload["previous_failures"] = [
            failure.model_dump(mode="json")
            for failure in failures[-MAX_GENERATION_FAILURE_CONTEXT:]
        ]
        task.payload = payload
    active_triggers = _task_generation_triggers(db, task)
    deferred_triggers = _task_generation_triggers(db, task, deferred=True)
    _write_task_generation_triggers(
        task,
        _combine_generation_triggers(active_triggers, deferred_triggers),
    )
    _write_task_generation_triggers(task, _GenerationTriggers(), deferred=True)
    task.state = "pending"
    task.dedupe_key = _generation_task_dedupe_key(task)
    task.log_path = None
    task.error = None
    task.started_at = None
    task.finished_at = None
    task.updated_at = now
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise ValueError("another generation task has the same durable identity") from error
    db.refresh(task)
    return task


def _generation_task_dedupe_key(task: GenerationTask) -> str:
    if generation_task_kind(task) == "queue_fill":
        return GENERATION_DEDUPE_KEY
    request_id = topic_request_id(task)
    if request_id is None:
        raise ValueError("topic request task has an invalid request ID")
    return f"topic-request:{request_id}"


def _running_generation_task(db: Session, task_id: int) -> GenerationTask:
    task = db.get(GenerationTask, task_id)
    if task is None:
        raise LookupError(f"generation task not found: {task_id}")
    if task.state != "running":
        raise ValueError(f"generation task {task_id} is not running")
    return task


def _generation_status(db: Session) -> Literal["pending", "running", "failed"] | None:
    latest = db.scalar(select(GenerationTask).order_by(GenerationTask.id.desc()).limit(1))
    if latest is None or latest.state == "completed":
        return None
    return latest.state


def latest_profile_feedback(db: Session) -> list[FeedbackTag]:
    profile = ensure_profile(db)
    events = db.scalars(
        select(Interaction)
        .join(Lesson, Lesson.id == Interaction.lesson_id)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
            Interaction.event_type.in_(("lesson.completed", "lesson.rated")),
        )
        .order_by(Interaction.occurred_at, Interaction.id)
    ).all()
    return _latest_completed_session_feedback(events)


def _latest_completed_session_feedback(
    events: Sequence[Interaction],
) -> list[FeedbackTag]:
    latest_completion = next(
        (event for event in reversed(events) if event.event_type == "lesson.completed"),
        None,
    )
    if latest_completion is None:
        return []
    latest_rating = next(
        (
            event
            for event in reversed(events)
            if event.event_type == "lesson.rated"
            and event.lesson_id == latest_completion.lesson_id
            and event.session_id == latest_completion.session_id
        ),
        None,
    )
    return _feedback_from_payload(latest_rating.payload) if latest_rating is not None else []


def rebuild_proficiency_state(db: Session) -> ProficiencyState:
    """Rebuild proficiency while keeping profile migration and anchoring in this facade."""

    profile = ensure_profile(db)
    source = profile_level_source(profile)
    evidence_cursor = (
        _self_reported_level_evidence_cursor(profile) if source == "self_reported" else None
    )
    return _rebuild_proficiency_state(
        db,
        profile=profile,
        source=source,
        evidence_cursor=evidence_cursor,
    )


def calibration_generation_brief(db: Session) -> CalibrationGenerationBrief | None:
    profile = ensure_profile(db)
    if generation_mode(profile) != "calibration":
        return None
    state = db.get(ProficiencyState, 1)
    target = (
        state.estimate if state is not None and state.estimate is not None else profile.difficulty
    )
    target = max(0.02, min(0.98, target))
    sequence = (state.qualified_attempts if state is not None else 0) + 1
    if sequence == 1:
        seed = questionnaire_seed(profile)
        if seed is None:
            start, stop = 0.10, 0.80
        else:
            mean, variance = seed
            radius = max(0.11, 2 * math.sqrt(variance))
            start, stop = _bounded_interval(mean, radius * 2)
        difficulties = _linspace(start, stop, CALIBRATION_PROBE_COUNT)
    else:
        difficulties = _linspace(
            max(0.02, target - 0.15),
            min(0.98, target + 0.15),
            CALIBRATION_PROBE_COUNT,
        )
    excluded = sorted(
        {
            probe.term_key
            for lesson in db.scalars(select(Lesson).order_by(Lesson.id))
            for document in [lesson_document(lesson)]
            if document.calibration is not None
            for probe in document.calibration.probes
        }
    )
    return CalibrationGenerationBrief(
        sequence=sequence,
        target_difficulty=round(target, 6),
        probe_difficulties=difficulties,
        excluded_term_keys=excluded,
    )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    step = (stop - start) / (count - 1)
    return [round(start + step * index, 6) for index in range(count)]


def _bounded_interval(center: float, width: float) -> tuple[float, float]:
    width = min(0.96, max(0.22, width))
    start = min(max(0.02, center - width / 2), 0.98 - width)
    return start, start + width


def build_agent_brief(
    db: Session,
    *,
    feedback: Sequence[FeedbackTag] | None = None,
) -> AgentBrief:
    profile = ensure_profile(db)
    profile_view = build_profile_view(db, profile)
    profile_view = profile_view.model_copy(
        update={
            "preferences": {
                key: value for key, value in profile_view.preferences.items() if key != "tts"
            }
        }
    )
    ensure_derived_states(db)
    catalog = find_grammar_catalog(profile.learning_language)
    now = utc_now()
    lessons = db.scalars(
        select(Lesson)
        .where(
            Lesson.learning_language == profile.learning_language,
            Lesson.translation_language == profile.translation_language,
        )
        .order_by(Lesson.imported_at.desc(), Lesson.id.desc())
    ).all()
    documents = {lesson.id: lesson_document(lesson) for lesson in lessons}
    units = build_learning_unit_index(documents[lesson.id] for lesson in reversed(lessons))
    lesson_ids = [lesson.id for lesson in lessons]
    skipped_ids = skipped_lesson_ids(db, lesson_ids)
    interactions = (
        db.scalars(
            select(Interaction)
            .where(Interaction.lesson_id.in_(lesson_ids))
            .order_by(Interaction.occurred_at, Interaction.id)
        ).all()
        if lesson_ids
        else []
    )
    completion_by_lesson: set[int] = set()
    rating_by_lesson: dict[int, Literal[-1, 1]] = {}
    feedback_by_lesson: dict[int, list[FeedbackTag]] = {}
    for interaction in interactions:
        if interaction.event_type == "lesson.completed":
            completion_by_lesson.add(interaction.lesson_id)
        elif interaction.event_type == "lesson.rated":
            rating = interaction.payload.get("rating")
            if rating in {-1, 1}:
                rating_by_lesson[interaction.lesson_id] = rating
            feedback_by_lesson[interaction.lesson_id] = _feedback_from_payload(interaction.payload)
    latest_feedback = (
        _latest_completed_session_feedback(interactions) if feedback is None else list(feedback)
    )

    target_appearances: Counter[str] = Counter()
    for lesson_id, document in documents.items():
        if lesson_id in skipped_ids:
            continue
        target_appearances.update(
            {
                canonical
                for term_key in document.target_term_keys
                if (canonical := units.resolve(term_key)) is not None
            }
        )
    terms = db.scalars(
        select(LexemeState).where(
            LexemeState.learning_language == profile.learning_language,
            LexemeState.translation_language == profile.translation_language,
        )
    ).all()
    terms = [term for term in terms if term.term_key in units.definitions]
    establishment = {
        term.term_key: term.qualified_exposures + term.reveal_failures for term in terms
    }
    failure_lessons: dict[str, set[int]] = defaultdict(set)
    for interaction in interactions:
        if interaction.event_type != "term.revealed":
            continue
        term_key = interaction.payload.get("term_key")
        if isinstance(term_key, str):
            canonical = units.resolve(term_key)
            if canonical is not None:
                failure_lessons[canonical].add(interaction.lesson_id)
                continue
            selected = units.select_evidence_targets(term_key, establishment)
            if len(set(selected)) >= 2:
                for selected_key in selected:
                    failure_lessons[selected_key].add(interaction.lesson_id)
    scored = [
        (
            _urgency(
                term,
                now,
                target_appearances=target_appearances[term.term_key],
                failure_lesson_count=len(failure_lessons[term.term_key]),
            ),
            term,
        )
        for term in terms
    ]
    if character_tracking_available(profile.learning_language):
        character_states = {
            state.character: state
            for state in db.scalars(
                select(CharacterState).where(
                    CharacterState.learning_language == profile.learning_language,
                    CharacterState.translation_language == profile.translation_language,
                )
            ).all()
        }
        scored.sort(
            key=lambda item: (
                -character_adjusted_offer_score(
                    item[0],
                    item[1],
                    character_states,
                    at=now,
                ),
                item[1].frequency_rank or 10**9,
                item[1].term_key,
            )
        )
    else:
        scored.sort(key=lambda item: (-item[0], item[1].frequency_rank or 10**9, item[1].term_key))
    scored = _rotate_recent_skipped_offers(scored, lessons, units)
    selected_terms = _select_generation_candidates(
        scored,
        profile_view,
        latest_feedback,
    )
    priority_terms = [_agent_term(term, urgency, now) for urgency, term in selected_terms]
    priority_grammar = _select_priority_grammar(
        db,
        catalog.constructions if catalog is not None else (),
        lessons,
        documents,
        completion_by_lesson,
        skipped_ids,
        profile_view,
        latest_feedback,
        now,
    )
    share_inputs = KnownShareInputs(
        units=units,
        states={term.term_key: term for term in terms},
        frontier_rank=learner_frontier_rank(db, DEFAULT_LEXEME_LEARNING_POLICY),
        memory=DEFAULT_LEXEME_LEARNING_POLICY.memory,
        at=now,
    )
    recent_import_shares = [
        lesson.known_share_at_import
        for lesson in lessons
        if documents[lesson.id].calibration is None and lesson.known_share_at_import is not None
    ][:3]
    vocabulary = plan_vocabulary(
        entries=word_list(profile.learning_language),
        known_lemmas=(term.lemma for term in terms),
        frontier_rank=learner_level_frontier_rank(db),
        memory=share_inputs.memory,
        text_length=_effective_text_length(profile_view, latest_feedback),
        target_known_share=_target_known_share(profile_view.preferences),
        recent_known_shares=recent_import_shares,
    )
    mastered = sorted(
        term.term_key
        for term in terms
        if term.mastery >= TERM_BAND_POLICY.mastered_min_mastery
        and term.qualified_exposures >= TERM_BAND_POLICY.mastered_min_exposures
    )
    recent: list[AgentLessonBrief] = []
    for lesson in lessons[:10]:
        document = documents[lesson.id]
        sentences = body_sentences(document)
        recent.append(
            AgentLessonBrief(
                id=lesson.id,
                key=lesson.key,
                title=document.title,
                topic=document.topic,
                level=document.level,
                difficulty=document.difficulty,
                imported_at=as_utc(lesson.imported_at),
                completed=lesson.id in completion_by_lesson,
                rating=rating_by_lesson.get(lesson.id),
                feedback=feedback_by_lesson.get(lesson.id, []),
                target_term_keys=sorted(
                    {
                        canonical
                        for term_key in document.target_term_keys
                        if (canonical := units.resolve(term_key)) is not None
                    }
                ),
                grammar_keys=sorted(document.grammar_construction_keys()),
                opening_excerpt=_lesson_excerpt(sentences, from_end=False),
                ending_excerpt=_lesson_excerpt(sentences, from_end=True),
                calibration_lesson=document.calibration is not None,
                metadata=_agent_lesson_metadata(document.metadata),
                known_share=known_share(document, share_inputs),
            )
        )
    return AgentBrief(
        generated_at=now,
        profile=profile_view,
        lesson_count=len(lessons),
        completed_lesson_count=len(completion_by_lesson),
        interaction_count=len(interactions),
        language_guidance=list(language_pack(profile.learning_language).generation_guidance),
        grammar_catalog=(
            [_agent_grammar_catalog_entry(construction) for construction in catalog.constructions]
            if catalog is not None
            else []
        ),
        priority_grammar=priority_grammar,
        priority_terms=priority_terms,
        mastered_term_keys=mastered,
        recent_lessons=recent,
        vocabulary=vocabulary,
        reading_preferences=current_preferences(db),
        content_history=content_history(db),
    )


_AGENT_LESSON_METADATA_KEYS = {
    "content_angle",
    "content_hypothesis",
    "content_plan",
    "continuity",
    "episode",
    "episode_number",
    "series",
    "series_key",
}


def _agent_lesson_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expose semantic continuity while hiding stale generation bookkeeping."""

    return {key: value for key, value in metadata.items() if key in _AGENT_LESSON_METADATA_KEYS}


def _lesson_excerpt(sentences: Sequence[LessonSentence], *, from_end: bool) -> str:
    selected = sentences[-2:] if from_end else sentences[:2]
    text = " ".join(sentence.text.strip() for sentence in selected if sentence.text.strip())
    if len(text) <= 180:
        return text
    return text[:179].rstrip() + "…"


def build_generation_target_policy(
    db: Session,
    mode: GenerationMode,
    *,
    brief: AgentBrief | None = None,
    feedback: Sequence[FeedbackTag] | None = None,
) -> GenerationTargetPolicy:
    brief = brief or build_agent_brief(db, feedback=feedback)
    latest_feedback = latest_profile_feedback(db) if feedback is None else list(feedback)
    if mode == "calibration":
        return GenerationTargetPolicy(
            preferred_count=0,
            max_count=0,
            target_text_length=_calibration_text_length(brief.profile, latest_feedback),
            candidate_term_keys=[],
        )

    text_length = _effective_text_length(brief.profile, latest_feedback)
    candidate_keys = [term.key for term in brief.priority_terms]
    if not candidate_keys:
        return GenerationTargetPolicy(
            preferred_count=0,
            max_count=0,
            target_text_length=text_length,
            candidate_term_keys=[],
        )

    confidence = _learner_confidence(brief.profile.proficiency)
    effective_learning_words = text_length * (0.5 + confidence / 2.0)
    target_capacity = (
        effective_learning_words
        / WORDS_PER_DELIBERATE_TARGET
        * _profile_target_capacity_factor(brief.profile)
    )
    max_count = min(
        len(candidate_keys),
        MAX_PROTOCOL_TARGETS,
        math.ceil(target_capacity),
    )
    preferred_count = min(max_count, math.floor(target_capacity))
    return GenerationTargetPolicy(
        preferred_count=preferred_count,
        max_count=max_count,
        target_text_length=text_length,
        candidate_term_keys=candidate_keys,
    )


def _agent_grammar_catalog_entry(
    construction: GrammarConstruction,
) -> AgentGrammarCatalogEntry:
    return AgentGrammarCatalogEntry(
        key=construction.key,
        label=construction.label,
        description=construction.description,
        category=construction.category,
        difficulty=construction.difficulty,
        generation_hint=construction.generation_hint,
    )


def _select_priority_grammar(
    db: Session,
    constructions: Sequence[GrammarConstruction],
    lessons: Sequence[Lesson],
    documents: Mapping[int, LessonDocument],
    completed_lesson_ids: set[int],
    skipped_lesson_ids: set[int],
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
    now: datetime,
) -> list[AgentGrammarBrief]:
    """Offer a small choice pool while leaving realization to the writing agent."""

    unread_keys = {
        construction_key
        for lesson in lessons
        if lesson.id not in completed_lesson_ids and lesson.id not in skipped_lesson_ids
        for construction_key in documents[lesson.id].grammar_construction_keys()
    }
    states = {
        state.construction_key: state
        for state in db.scalars(
            select(GrammarState).where(
                GrammarState.learning_language == profile.learning_language,
                GrammarState.translation_language == profile.translation_language,
            )
        )
    }
    cooled = _recent_skipped_grammar_offers(lessons)
    scored: list[tuple[float, AgentGrammarBrief]] = []
    for construction in constructions:
        if construction.key in unread_keys:
            continue
        state = states.get(construction.key)
        score, reason = _grammar_urgency(construction, state, profile.difficulty, now)
        if construction.key in cooled:
            score -= 0.18
        scored.append(
            (
                score,
                AgentGrammarBrief(
                    key=construction.key,
                    label=construction.label,
                    description=construction.description,
                    category=construction.category,
                    difficulty=construction.difficulty,
                    mastery=round(state.mastery, 6) if state is not None else 0.5,
                    stability_days=round(state.stability_days, 6) if state is not None else 0.5,
                    qualified_exposures=state.qualified_exposures if state is not None else 0,
                    help_failures=state.explicit_help_failures if state is not None else 0,
                    inferred_difficulty_signals=(
                        state.inferred_difficulty_signals if state is not None else 0
                    ),
                    next_due_at=(
                        as_utc(state.next_due_at)
                        if state is not None and state.next_due_at is not None
                        else None
                    ),
                    urgency=round(max(0.0, score), 6),
                    reason=reason,
                ),
            )
        )
    scored.sort(key=lambda item: (-item[0], item[1].difficulty, item[1].key))
    if not scored:
        return []
    confidence = _learner_confidence(profile.proficiency)
    pool_size = max(3, math.ceil(math.sqrt(len(scored)) * (1.5 + confidence / 2)))
    if "more_grammar" in feedback:
        pool_size += 2
    elif "less_grammar" in feedback:
        pool_size -= 1
    return [brief for _score, brief in scored[: min(len(scored), max(1, pool_size), 12)]]


def _grammar_urgency(
    construction: GrammarConstruction,
    state: GrammarState | None,
    learner_difficulty: float,
    now: datetime,
) -> tuple[float, Literal["unseen", "due", "fragile"]]:
    distance = abs(construction.difficulty - learner_difficulty)
    level_fit = math.exp(-5.0 * distance)
    above_frontier = max(0.0, construction.difficulty - learner_difficulty - 0.18)
    frontier_fit = max(0.0, 1.0 - above_frontier * 4.0)
    if state is None or state.first_seen_at is None:
        return 0.45 + 0.30 * level_fit + 0.20 * frontier_fit, "unseen"

    if state.next_due_at is None:
        due = 0.45
        reason: Literal["unseen", "due", "fragile"] = "fragile"
    else:
        days_overdue = (now - as_utc(state.next_due_at)).total_seconds() / 86_400
        due = max(0.0, min(1.0, 0.5 + days_overdue / 7.0))
        reason = "due" if days_overdue >= 0 else "fragile"
    uncertainty = min(1.0, 4.0 / max(1.0, state.alpha + state.beta))
    signals = min(
        1.0,
        (state.explicit_help_failures + 0.35 * state.inferred_difficulty_signals) / 3.0,
    )
    sparse_evidence = 1.0 - min(1.0, state.qualified_exposures / 4.0)
    score = (
        0.30 * (1.0 - state.mastery)
        + 0.22 * due
        + 0.14 * uncertainty
        + 0.16 * signals
        + 0.10 * level_fit
        + 0.08 * sparse_evidence
    )
    return score, reason


def _recent_skipped_grammar_offers(lessons: Sequence[Lesson]) -> set[str]:
    for lesson in lessons:
        metadata = lesson_document(lesson).metadata
        value = metadata.get("grammar")
        if not isinstance(value, Mapping):
            value = metadata.get("grammar_targets")
        if not isinstance(value, Mapping):
            continue
        offered = _metadata_term_keys(value.get("offered"))
        realized = _metadata_term_keys(value.get("realized"))
        return offered - realized
    return set()


def _rotate_recent_skipped_offers(
    scored: Sequence[tuple[float, LexemeState]],
    lessons: Sequence[Lesson],
    units: LearningUnitIndex,
) -> list[tuple[float, LexemeState]]:
    cooled: set[str] = set()
    for lesson in lessons:
        targets = lesson_document(lesson).metadata.get("targets")
        if not isinstance(targets, Mapping):
            continue
        offered = {
            canonical
            for term_key in _metadata_term_keys(targets.get("offered"))
            if (canonical := units.resolve(term_key)) is not None
        }
        realized = {
            canonical
            for term_key in _metadata_term_keys(targets.get("realized"))
            if (canonical := units.resolve(term_key)) is not None
        }
        cooled = offered - realized
        break
    if not cooled:
        return list(scored)
    return [item for item in scored if item[1].term_key not in cooled] + [
        item for item in scored if item[1].term_key in cooled
    ]


def _metadata_term_keys(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _select_generation_candidates(
    scored: Sequence[tuple[float, LexemeState]],
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> list[tuple[float, LexemeState]]:
    available = len(scored)
    if available == 0:
        return []

    confidence = _learner_confidence(profile.proficiency)
    text_length = _effective_text_length(profile, feedback)
    lesson_units = max(1.0, text_length / WORDS_PER_DELIBERATE_TARGET)
    uncertainty_breadth = 2.0 - confidence
    pool_size = math.ceil(math.sqrt(available * lesson_units) * uncertainty_breadth)
    pool_size = min(available, MAX_GENERATION_CANDIDATES, max(1, pool_size))

    review = [item for item in scored if item[1].first_seen_at is not None]
    unseen = [item for item in scored if item[1].first_seen_at is None]
    known_ratio = _target_known_ratio(profile.preferences)
    exploration_share = (1.0 - known_ratio) * confidence
    unseen_slots = min(len(unseen), math.floor(pool_size * exploration_share))
    review_slots = min(len(review), pool_size - unseen_slots)

    selected = review[:review_slots] + unseen[:unseen_slots]
    selected_keys = {item[1].term_key for item in selected}
    for item in scored:
        if len(selected) >= pool_size:
            break
        if item[1].term_key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(item[1].term_key)
    return selected[:pool_size]


def _effective_text_length(
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> int:
    configured = profile.preferences.get("text_length")
    if isinstance(configured, bool) or not isinstance(configured, int):
        requested = DEFAULT_GENERATED_TEXT_LENGTH
    else:
        requested = min(MAX_GENERATED_TEXT_LENGTH, max(MIN_GENERATED_TEXT_LENGTH, configured))
    if "longer" in feedback:
        requested = math.ceil(requested * 4 / 3)
    elif "shorter" in feedback:
        requested = math.floor(requested * 3 / 4)
    return min(MAX_GENERATED_TEXT_LENGTH, max(MIN_GENERATED_TEXT_LENGTH, requested))


def _calibration_text_length(
    profile: ProfileView,
    feedback: Sequence[FeedbackTag],
) -> int:
    target = _effective_text_length(profile, feedback)
    if "text_length" not in profile.preferences:
        target = CALIBRATION_PROBE_COUNT * CALIBRATION_WORDS_PER_PROBE
        if "longer" in feedback:
            target = math.ceil(target * 4 / 3)
        elif "shorter" in feedback:
            target = math.floor(target * 3 / 4)
    minimum = CALIBRATION_PROBE_COUNT * CALIBRATION_MIN_WORDS_PER_PROBE
    return max(minimum, target)


def _learner_confidence(proficiency: ProficiencyView) -> float:
    if proficiency.lower is not None and proficiency.upper is not None:
        return max(0.0, min(1.0, 1.0 - (proficiency.upper - proficiency.lower)))
    if proficiency.source == "self_reported":
        return 0.5
    return 0.0


def _target_known_ratio(preferences: Mapping[str, Any]) -> float:
    configured = preferences.get("target_known_ratio")
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        return DEFAULT_TARGET_KNOWN_RATIO
    return max(0.5, min(1.0, float(configured)))


def _target_known_share(preferences: Mapping[str, Any]) -> float:
    configured = preferences.get("target_known_share")
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        return DEFAULT_TARGET_KNOWN_SHARE
    return max(0.85, min(0.99, float(configured)))


def _profile_target_capacity_factor(profile: ProfileView) -> float:
    known_ratio = _target_known_ratio(profile.preferences)
    known_ratio_adjustment = (DEFAULT_TARGET_KNOWN_RATIO - known_ratio) / 2.0
    difficulty_adjustment = (profile.difficulty - 0.5) / 4.0
    return 1.0 + known_ratio_adjustment + difficulty_adjustment


DESIRED_RETENTION = 0.9


def _urgency(
    term: LexemeState,
    now: datetime,
    *,
    target_appearances: int = 0,
    failure_lesson_count: int = 0,
) -> float:
    """Rank a word for deliberate reuse by how likely it is forgotten right now.

    Reviewed words are scored by 1 - predicted recall, with a bonus once recall has fallen below
    the desired retention (due). Words never reviewed here are scored by how likely they are still
    unknown given frequency and level, so common words the learner surely knows are not drilled.
    """

    frequency = (
        1.0 / (1.0 + math.log10(max(1, term.frequency_rank)))
        if term.frequency_rank is not None
        else 0.25
    )
    memory = term.memory_state
    if memory is None:
        need = 1.0 - term.prior_known
        due = 0.5
    else:
        recall = recall_now(memory, prior=term.prior_known, at=now)
        need = 1.0 - recall
        due = 1.0 if recall < DESIRED_RETENTION else 0.0
    uncertainty = min(1.0, 4.0 / (term.alpha + term.beta))
    failure_spread = min(1.0, failure_lesson_count / 3.0)
    target_signal = min(1.0, float(target_appearances))
    rare_incidental_penalty = (
        0.12
        if term.reveal_failures == 1
        and target_appearances == 0
        and term.frequency_rank is not None
        and term.frequency_rank > 10_000
        else 0.0
    )
    score = (
        0.42 * need
        + 0.18 * due
        + 0.14 * frequency
        + 0.08 * uncertainty
        + 0.1 * failure_spread
        + 0.08 * target_signal
        - rare_incidental_penalty
    )
    return round(max(0.0, score), 6)


def _agent_term(term: LexemeState, urgency: float, now: datetime) -> AgentTermBrief:
    if term.first_seen_at is None:
        reason: Literal["unseen", "due", "fragile"] = "unseen"
    elif term.next_due_at is not None and as_utc(term.next_due_at) <= now:
        reason = "due"
    else:
        reason = "fragile"
    return AgentTermBrief(
        key=term.term_key,
        lemma=term.lemma,
        pos=term.pos,
        gloss=term.gloss,
        pronunciation=term.pronunciation,
        frequency_rank=term.frequency_rank,
        mastery=round(term.mastery, 6),
        stability_days=round(term.stability_days, 6),
        qualified_exposures=term.qualified_exposures,
        reveal_failures=term.reveal_failures,
        next_due_at=as_utc(term.next_due_at) if term.next_due_at else None,
        urgency=urgency,
        reason=reason,
    )


def _validate_event_references(event: InteractionIn, document: LessonDocument) -> None:
    if event.type == "term.revealed":
        term_key = event.payload["term_key"]
        if term_key not in document.term_catalog():
            raise ValueError(f"unknown term key for lesson: {term_key}")
    elif event.type == "translation.revealed" and event.payload["scope"] == "sentence":
        sentence_key = event.payload["sentence_key"]
        if sentence_key not in document.sentence_terms():
            raise ValueError(f"unknown sentence key for lesson: {sentence_key}")
    elif event.type == "translation.revealed" and event.payload["scope"] == "grammar":
        occurrence_key = event.payload["occurrence_key"]
        construction_key = event.payload["construction_key"]
        sentence_key = event.payload["sentence_key"]
        occurrence = document.grammar_occurrences().get(occurrence_key)
        if occurrence is None:
            raise ValueError(f"unknown grammar occurrence key for lesson: {occurrence_key}")
        expected_sentence = document.grammar_occurrence_sentences()[occurrence_key]
        if sentence_key != expected_sentence:
            raise ValueError(
                f"grammar occurrence {occurrence_key} does not belong to sentence: {sentence_key}"
            )
        if construction_key != occurrence.construction_key:
            raise ValueError(
                f"grammar occurrence {occurrence_key} does not use construction: {construction_key}"
            )


def _validate_term_keys_across_lessons(db: Session, document: LessonDocument) -> None:
    incoming = document.term_catalog()
    lessons = db.scalars(
        select(Lesson).where(Lesson.key != document.key).order_by(Lesson.imported_at, Lesson.id)
    )
    for lesson in lessons:
        existing_document = lesson_document(lesson)
        if (
            existing_document.learning_language != document.learning_language
            or existing_document.translation_language != document.translation_language
        ):
            continue
        existing = existing_document.term_catalog()
        for key in incoming.keys() & existing.keys():
            if incoming[key] != existing[key]:
                raise ValueError(f"term key has conflicting definitions across lessons: {key}")


def _reader_progress(session_id: str, events: Sequence[Interaction]) -> ReaderProgress:
    terms: set[str] = set()
    sentences: set[str] = set()
    grammar_occurrences: set[str] = set()
    full_translation = False
    started = False
    completed = False
    rating: Literal[-1, 1] | None = None
    feedback: list[FeedbackTag] = []
    for event in events:
        if event.event_type == "lesson.started":
            started = True
        elif event.event_type == "term.revealed":
            term_key = event.payload.get("term_key")
            if isinstance(term_key, str):
                terms.add(term_key)
        elif event.event_type == "translation.revealed":
            scope = event.payload.get("scope")
            if scope == "lesson":
                full_translation = True
            elif scope == "sentence" and isinstance(event.payload.get("sentence_key"), str):
                sentences.add(event.payload["sentence_key"])
            elif scope == "grammar" and isinstance(event.payload.get("occurrence_key"), str):
                grammar_occurrences.add(event.payload["occurrence_key"])
        elif event.event_type == "lesson.completed":
            completed = True
        elif event.event_type == "lesson.rated":
            if event.payload.get("rating") in {-1, 1}:
                rating = event.payload["rating"]
            feedback = _feedback_from_payload(event.payload)
    return ReaderProgress(
        session_id=session_id,
        started=started,
        completed=completed,
        rating=rating,
        feedback=feedback,
        revealed_term_keys=sorted(terms),
        revealed_sentence_keys=sorted(sentences),
        revealed_grammar_occurrence_keys=sorted(grammar_occurrences),
        full_translation_revealed=full_translation,
    )


def build_profile_view(db: Session, profile: Profile | None = None) -> ProfileView:
    profile = profile or ensure_profile(db)
    state = db.get(ProficiencyState, 1)
    difficulty = (
        state.estimate if state is not None and state.estimate is not None else profile.difficulty
    )
    proficiency = ProficiencyView(
        source=profile_level_source(profile),
        status=state.status if state is not None else "unstarted",
        lower=state.lower if state is not None else None,
        upper=state.upper if state is not None else None,
        lower_level=(
            cefr_for_difficulty(state.lower)
            if state is not None and state.lower is not None
            else None
        ),
        upper_level=(
            cefr_for_difficulty(state.upper)
            if state is not None and state.upper is not None
            else None
        ),
        qualified_attempts=state.qualified_attempts if state is not None else 0,
        usable_probes=state.usable_probes if state is not None else 0,
        qualified_readings=state.qualified_readings if state is not None else 0,
    )
    return ProfileView(
        learning_language=profile.learning_language,
        translation_language=profile.translation_language,
        level=cefr_for_difficulty(difficulty),
        difficulty=difficulty,
        interests=list(profile.interests),
        preferences=dict(profile.preferences),
        proficiency=proficiency,
    )


def profile_fingerprint(db: Session, profile: Profile | None = None) -> str:
    stored_profile = profile or ensure_profile(db)
    calibration = calibration_generation_brief(db)
    profile_payload: dict[str, Any] = {
        "learning_language": stored_profile.learning_language,
        "translation_language": stored_profile.translation_language,
        "level": stored_profile.level,
        "difficulty": stored_profile.difficulty,
        "interests": list(stored_profile.interests),
        "preferences": {
            key: value for key, value in stored_profile.preferences.items() if key != "tts"
        },
    }
    if profile_level_source(stored_profile) == "estimated":
        profile_payload.pop("difficulty")
        profile_payload.pop("level")
    serialized = {
        "profile": profile_payload,
        "calibration": calibration.model_dump(mode="json") if calibration is not None else None,
        "language_pack_digest": language_pack_digest(stored_profile.learning_language),
        "grammar_catalog_digest": (
            grammar_catalog_digest(stored_profile.learning_language)
            if find_grammar_catalog(stored_profile.learning_language) is not None
            else None
        ),
    }
    return hashlib.sha256(_canonical_json(serialized).encode()).hexdigest()


def _stored_level_anchor(preferences: Mapping[str, Any], fallback: datetime) -> datetime:
    value = preferences.get(PROFILE_LEVEL_SET_AT_KEY)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            pass
        else:
            return as_utc(parsed)
    return as_utc(fallback)


def _self_reported_level_evidence_cursor(profile: Profile) -> int:
    value = profile.preferences.get(PROFILE_LEVEL_EVIDENCE_CURSOR_KEY)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("self-reported profile has an invalid level evidence cursor")
    return value


def profile_key(profile: Profile) -> str:
    raw = f"{profile.learning_language}--{profile.translation_language}".lower()
    parts = [part for part in "".join(char if char.isalnum() else " " for char in raw).split()]
    return "-".join(parts)


def _workspace_key_and_path(db: Session, profile: Profile) -> tuple[str, str]:
    workspace = db.info.get("workspace")
    if isinstance(workspace, Workspace):
        return workspace.profile_id, workspace.relative_directory
    key = profile_key(profile)
    return key, f"profiles/{key}"


def _feedback_from_payload(payload: Mapping[str, Any]) -> list[FeedbackTag]:
    value = payload.get("feedback")
    allowed = set(get_args(FeedbackTag))
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item not in allowed for item in value
    ):
        return []
    return cast(list[FeedbackTag], list(value))


def _input_event_fingerprint(event: InteractionIn) -> str:
    return _event_fingerprint(event, event.type)


def _stored_event_fingerprint(event: Interaction) -> str:
    return _event_fingerprint(event, event.event_type)


def _event_fingerprint(event: InteractionIn | Interaction, event_type: str) -> str:
    return _canonical_json(
        {
            "event_id": event.event_id,
            "session_id": event.session_id,
            "lesson_id": event.lesson_id,
            "type": event_type,
            "occurred_at": as_utc(event.occurred_at).isoformat(),
            "payload": event.payload,
        }
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
