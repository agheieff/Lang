from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from server.memory_model import (
    AGAIN,
    FSRS_DEFAULT_WEIGHTS,
    GOOD,
    MemoryPolicy,
    interval_days,
    knowledge,
    prior_known,
    retrievability,
    review,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_stability_is_the_time_to_ninety_percent_recall() -> None:
    assert retrievability(7.0, 7.0) == pytest.approx(0.9)
    assert interval_days(7.0) == pytest.approx(7.0)
    assert retrievability(7.0, 0.0) == 1.0


def test_first_reviews_use_fsrs_initial_stability() -> None:
    assert review(None, AGAIN, T0).stability_days == pytest.approx(FSRS_DEFAULT_WEIGHTS[0])
    assert review(None, GOOD, T0).stability_days == pytest.approx(FSRS_DEFAULT_WEIGHTS[2])
    partial = review(None, GOOD, T0, weight=0.5).stability_days
    assert FSRS_DEFAULT_WEIGHTS[0] < partial < FSRS_DEFAULT_WEIGHTS[2]


def test_massed_success_adds_almost_nothing_but_spaced_success_grows_stability() -> None:
    state = review(None, GOOD, T0)
    massed = review(state, GOOD, T0 + timedelta(minutes=5))
    spaced = review(state, GOOD, T0 + timedelta(days=4))

    assert massed.stability_days == pytest.approx(state.stability_days, rel=0.01)
    assert spaced.stability_days > 2 * state.stability_days


def test_a_lapse_never_increases_stability() -> None:
    state = review(review(None, GOOD, T0), GOOD, T0 + timedelta(days=4))
    lapsed = review(state, AGAIN, T0 + timedelta(days=20))
    assert lapsed.stability_days < state.stability_days
    assert lapsed.difficulty > state.difficulty


def test_knowledge_projects_recall_over_the_horizon_and_falls_back_to_the_prior() -> None:
    state = review(None, GOOD, T0)
    assert knowledge(None, prior=0.3, at=T0) == 0.3
    assert knowledge(state, prior=0.3, at=T0) == pytest.approx(
        retrievability(state.stability_days, 30.0)
    )
    assert knowledge(state, prior=0.3, at=T0 + timedelta(days=10)) < knowledge(
        state, prior=0.3, at=T0
    )


def test_frequency_prior_is_centered_on_the_frontier() -> None:
    assert prior_known(1_000, 1_000) == pytest.approx(0.5)
    assert prior_known(10, 1_000) > prior_known(100, 1_000) > 0.5 > prior_known(50_000, 1_000)
    assert prior_known(None, 1_000) == 0.5


@pytest.mark.parametrize(
    "overrides",
    [{"passive_confidence": 0}, {"reread_weight": 1.5}, {"weights": (1.0,)}, {"prior_slope": 0}],
)
def test_memory_policy_rejects_invalid_configuration(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MemoryPolicy(**overrides)  # type: ignore[arg-type]
