from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from server.learning import ensure_profile
from server.models import Base

LessonFactory = Callable[..., dict[str, Any]]
EventFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def session_factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    yield factory
    engine.dispose()


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        ensure_profile(session)
        yield session


@pytest.fixture
def api_client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    from server.main import app, get_db, get_legacy_db

    def override_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_legacy_db] = override_db
    client = TestClient(app)
    yield client
    client.close()
    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_legacy_db, None)


@pytest.fixture
def lesson_factory() -> LessonFactory:
    def make_lesson(
        *,
        key: str = "lesson-one",
        learning_language: str = "es-ES",
        translation_language: str = "en",
        terms: list[tuple[str, str, str, str, str, int]] | None = None,
        targets: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        term_specs = terms or [
            ("es:manana:NOUN", "mañana🧠", "mañana", "NOUN", "tomorrow", 5),
            ("es:corazon:NOUN", "corazón", "corazón", "NOUN", "heart", 20),
            ("es:mundo:NOUN", "世界", "mundo", "NOUN", "world", 100),
        ]
        sentences: list[dict[str, Any]] = []
        for index, (term_key, surface, lemma, pos, gloss, rank) in enumerate(term_specs, 1):
            runs: list[dict[str, Any]] = []
            if index == 1:
                runs.append({"text": "<script>alert('x')</script> 🌍 "})
            runs.extend(
                [
                    {
                        "text": surface,
                        "term": {
                            "key": term_key,
                            "lemma": lemma,
                            "pos": pos,
                            "gloss": gloss,
                            "frequency_rank": rank,
                        },
                    },
                    {"text": ". "},
                ]
            )
            sentences.append(
                {
                    "key": f"sentence-{index}",
                    "runs": runs,
                    "translation": f"Translation {index} 🧠",
                }
            )

        return {
            "schema_version": 1,
            "key": key,
            "title": f"Lesson {key}",
            "learning_language": learning_language,
            "translation_language": translation_language,
            "topic": "science",
            "level": "A1",
            "difficulty": 0.15,
            "blocks": [{"key": "block-1", "sentences": sentences}],
            "target_term_keys": targets if targets is not None else [term_specs[0][0]],
            "metadata": metadata or {"series": "test-series"},
        }

    return make_lesson


@pytest.fixture
def event_factory() -> EventFactory:
    base = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    def make_event(
        lesson_id: int,
        event_type: str,
        *,
        event_id: str,
        session_id: str = "session-1",
        payload: dict[str, Any] | None = None,
        seconds: int = 0,
    ) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "session_id": session_id,
            "lesson_id": lesson_id,
            "type": event_type,
            "occurred_at": (base + timedelta(seconds=seconds)).isoformat(),
            "payload": payload or {},
        }

    return make_event
