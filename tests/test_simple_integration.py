"""Simple integration test to verify system works."""

import pytest


def test_simple_workflow():
    """Verify all imports work and system is functional."""
    from server.services.recommendation import (
        TextFeatures,
        TextRequest,
        compute_similarity_score,
        select_best_text,
        detect_pool_gaps,
    )
    from server.services.content import generate_text_content, generate_translations
    from server.services.background_worker import (
        background_worker,
        maintenance_cycle,
        startup_generation,
    )
    from server.utils.nlp import (
        parse_csv_word_translations,
        parse_csv_translation,
        compute_word_spans,
    )

    # All imports should work
    assert TextFeatures is not None
    assert TextRequest is not None
    assert compute_similarity_score is not None
    assert select_best_text is not None
    assert detect_pool_gaps is not None
    assert generate_text_content is not None
    assert generate_translations is not None
    assert background_worker is not None
    assert maintenance_cycle is not None
    assert startup_generation is not None
    assert parse_csv_word_translations is not None
    assert parse_csv_translation is not None
    assert compute_word_spans is not None

    # Test CSV parsing
    csv_text = """word|translation|pos|lemma
hola|hello|INTJ|hola
mundo|world|NOUN|mundo
"""
    words = parse_csv_word_translations(csv_text)
    assert len(words) == 2
    assert words[0]["surface"] == "hola"

    # Test translation CSV parsing
    trans_text = """source|translation
Hola mundo|Hello world
"""
    translations = parse_csv_translation(trans_text)
    assert len(translations) == 1
    assert translations[0]["source"] == "Hola mundo"

    # Test span computation
    text = "Ich rufe meinen Freund an."
    word_data = [{"surface": "Ich", "translation": "I"}]
    spans = compute_word_spans(text, word_data)
    assert len(spans) == 1
    assert spans[0] == [(0, 3)]

    # Test non-continuous spans
    word_data_nc = [{"surface": "ruf...an", "translation": "call"}]
    spans_nc = compute_word_spans(text, word_data_nc)
    assert len(spans_nc) == 1
    assert len(spans_nc[0]) == 2  # Two spans for non-continuous

    # Test similarity scoring
    features = TextFeatures(
        text_id=1,
        difficulty=3.0,
        topics={"daily_life": 0.8},
        word_count=150,
    )

    request = TextRequest(
        profile_id=1,
        lang="es",
        target_lang="en",
        difficulty_target=3.0,
        topics={"daily_life": 0.7},
        min_length=100,
        max_length=200,
    )

    score = compute_similarity_score(features, request)
    # Perfect match should have low score
    assert score < 2.0

    # Mismatch should have high score
    features_bad = TextFeatures(
        text_id=2,
        difficulty=8.0,  # Very different
        topics={"culture": 1.0},  # Wrong topic
        word_count=150,
    )

    score_bad = compute_similarity_score(features_bad, request)
    # Bad match should have high score
    assert score_bad > 5.0
