from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.learning import import_lesson, record_events
from server.lexeme_learning import LexemeLearningPolicy
from server.models import LexemeState


@pytest.mark.parametrize(
    "policy",
    [
        {"passive_success_weight": -0.1},
        {"stability_gain_days_per_success_mass": -0.1},
        {"repeated_reveal_grades": ()},
        {"repeated_reveal_grades": (0.6, 0.25)},
        {"full_failure_due_days": 3.0, "weak_failure_due_days": 2.0},
    ],
)
def test_lexeme_policy_rejects_invalid_configuration(policy: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LexemeLearningPolicy(**policy)


def test_ten_clean_sessions_support_confident_word_recognition(
    db: Session,
    lesson_factory: Any,
    event_factory: Any,
) -> None:
    term = ("es:este:DET", "este", "este", "DET", "this", 10)
    lesson = import_lesson(
        db,
        lesson_factory(
            key="common-word",
            terms=[term],
            targets=[],
        ),
    )
    record_events(
        db,
        [
            event_factory(
                lesson.id,
                "lesson.completed",
                event_id=f"clean-{index}",
                session_id=f"session-{index}",
                payload={"active_seconds": 30, "completion_ratio": 1},
                seconds=30 + index,
            )
            for index in range(10)
        ],
    )

    state = db.scalar(select(LexemeState).where(LexemeState.term_key == term[0]))

    assert state is not None
    assert state.qualified_exposures == 10
    assert state.alpha == pytest.approx(17.0)
    assert state.mastery == pytest.approx(0.8947368421)
    assert state.stability_days > 8
