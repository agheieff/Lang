"""Tests for text recommendation engine."""

import pytest
from server.services.recommendation import (
    TextFeatures,
    TextRequest,
    compute_similarity_score,
    compute_text_features,
)


def test_text_features_dataclass():
    """Test TextFeatures dataclass creation."""
    features = TextFeatures(
        text_id=1,
        difficulty=3.5,
        topics={"daily_life": 0.8, "culture": 0.2},
        word_count=150,
    )

    assert features.text_id == 1
    assert features.difficulty == 3.5
    assert features.topics == {"daily_life": 0.8, "culture": 0.2}
    assert features.word_count == 150


def test_text_request_dataclass():
    """Test TextRequest dataclass creation."""
    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        topics={"daily_life": 0.7},
    )

    assert request.profile_id == 1
    assert request.difficulty_target == 3.0
    assert request.difficulty_tolerance == 2.0
    assert request.topics == {"daily_life": 0.7}


def test_compute_similarity_score_perfect_match():
    """Test similarity score for a perfect match."""
    features = TextFeatures(
        text_id=1,
        difficulty=3.0,
        topics={"daily_life": 0.7},
        word_count=200,
    )

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        topics={"daily_life": 0.7},
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score = compute_similarity_score(features, request)
    assert score < 1.0  # Very low score for perfect match


def test_compute_similarity_score_difficulty_mismatch():
    """Test similarity score with difficulty mismatch."""
    features = TextFeatures(text_id=1, difficulty=8.0, word_count=200)

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=2.0,
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score = compute_similarity_score(features, request)
    assert score > 10.0  # High penalty for difficulty mismatch


def test_compute_similarity_score_topic_mismatch():
    """Test similarity score with topic mismatch."""
    features = TextFeatures(
        text_id=1, difficulty=3.0, topics={"fiction": 1.0}, word_count=200
    )

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        topics={"daily_life": 0.7},
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score = compute_similarity_score(features, request)
    assert score > 1.0  # Topic mismatch adds penalty


def test_compute_similarity_score_length_penalty():
    """Test similarity score with length mismatch."""
    features = TextFeatures(text_id=1, difficulty=3.0, word_count=50)

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score = compute_similarity_score(features, request)
    assert score > 5.0  # Too short adds penalty


def test_compute_similarity_score_quality_bonus():
    """Test quality bonus for highly-rated texts."""
    features_good = TextFeatures(
        text_id=1,
        difficulty=3.0,
        word_count=200,
        rating_avg=4.5,
    )

    features_bad = TextFeatures(
        text_id=2,
        difficulty=3.0,
        word_count=200,
        rating_avg=2.0,
    )

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        min_length=150,
        max_length=250,
        preferred_length=200,
        min_rating=0.0,
    )

    score_good = compute_similarity_score(features_good, request)
    score_bad = compute_similarity_score(features_bad, request)

    assert score_good < score_bad  # Better rating = lower score


def test_target_words_coverage():
    """Test target words coverage in similarity score."""
    features = TextFeatures(
        text_id=1,
        difficulty=3.0,
        target_words_density={"hola": 1.0, "mundo": 0.5},
        word_count=200,
    )

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        target_words={"hola", "mundo", "adios"},
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score_partial = compute_similarity_score(features, request)
    assert score_partial > 0  # Partial coverage adds some penalty

    # Full coverage
    request_full = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        target_words={"hola", "mundo"},
        min_length=150,
        max_length=250,
        preferred_length=200,
    )

    score_full = compute_similarity_score(features, request_full)
    assert score_full < score_partial  # Full coverage = better score
