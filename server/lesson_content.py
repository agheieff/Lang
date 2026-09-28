"""Canonical accessors for immutable lesson content."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from weakref import WeakKeyDictionary

from server.models import Lesson
from server.schemas import LessonDocument, LessonRun, LessonSentence

_documents: WeakKeyDictionary[Lesson, tuple[str, LessonDocument]] = WeakKeyDictionary()
_documents_lock = threading.Lock()


def lesson_document(lesson: Lesson) -> LessonDocument:
    """Validate the payload that owns all reader-visible lesson content.

    Parsed documents are memoized per ORM instance so replaying several derived projections
    over the same session validates each lesson once. ``content_hash`` changes in lockstep
    with ``payload`` on the lesson-replacement path, which invalidates the cached entry.
    """

    content_hash = lesson.content_hash
    if content_hash is None:
        return LessonDocument.model_validate(lesson.payload)
    with _documents_lock:
        cached = _documents.get(lesson)
        if cached is not None and cached[0] == content_hash:
            return cached[1]
    document = LessonDocument.model_validate(lesson.payload)
    with _documents_lock:
        _documents[lesson] = (content_hash, document)
    return document


def body_sentences(document: LessonDocument) -> tuple[LessonSentence, ...]:
    return tuple(sentence for block in document.blocks for sentence in block.sentences)


def all_sentences(document: LessonDocument) -> tuple[LessonSentence, ...]:
    title = (document.title_sentence,) if document.title_sentence is not None else ()
    return (*title, *body_sentences(document))


def lexical_runs(document: LessonDocument) -> Iterator[LessonRun]:
    return (
        run for sentence in all_sentences(document) for run in sentence.runs if run.term is not None
    )


def lexical_token_count(document: LessonDocument) -> int:
    return sum(1 for _run in lexical_runs(document))
