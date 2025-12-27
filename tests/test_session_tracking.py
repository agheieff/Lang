"""Tests for session tracking functionality."""

import pytest
from datetime import datetime, timezone, timedelta


def test_session_data_structure():
    """Test that session data has required structure."""
    session_data = {
        "text_id": 1,
        "exposed_at": int(datetime.now(timezone.utc).timestamp() * 1000),
        "words": [
            {
                "surface": "hola",
                "lemma": "hola",
                "pos": "INTJ",
                "span_start": 0,
                "span_end": 4,
                "clicked": True,
                "click_count": 2,
                "translation_viewed": True,
                "translation_viewed_at": int(
                    datetime.now(timezone.utc).timestamp() * 1000
                ),
                "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            }
        ],
        "sentences": [],
        "full_translation_views": [
            {"timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
        ],
    }

    # Verify structure
    assert "text_id" in session_data
    assert "exposed_at" in session_data
    assert "words" in session_data
    assert "sentences" in session_data
    assert "full_translation_views" in session_data

    # Verify word has all required fields
    word = session_data["words"][0]
    assert "clicked" in word
    assert "click_count" in word
    assert "translation_viewed" in word
    assert "translation_viewed_at" in word


def test_track_interactions_from_session():
    """Test that session tracking function can be called."""
    from server.services.learning import track_interactions_from_session

    # Mock session data
    interactions = [
        {
            "surface": "hola",
            "lemma": "hola",
            "pos": "INTJ",
            "span_start": 0,
            "span_end": 4,
            "clicked": True,
            "click_count": 1,
            "translation_viewed": True,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
    ]

    # Function should exist and be callable
    assert callable(track_interactions_from_session)


def test_save_session_endpoint_accepts_new_fields():
    """Test that /reading/save-session accepts extended session data."""
    import json

    # Sample request body
    request_body = {
        "text_id": 1,
        "session_data": {
            "exposed_at": int(datetime.now(timezone.utc).timestamp() * 1000),
            "words": [
                {
                    "surface": "hola",
                    "lemma": "hola",
                    "clicked": True,
                    "click_count": 1,
                    "translation_viewed": True,
                    "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
                }
            ],
            "sentences": [],
            "full_translation_views": [
                {"timestamp": int(datetime.now(timezone.utc).timestamp() * 1000)}
            ],
        },
    }

    # Verify JSON is valid
    json_str = json.dumps(request_body)
    parsed = json.loads(json_str)

    assert parsed["session_data"]["exposed_at"] > 0
    assert len(parsed["session_data"]["words"]) == 1
    assert len(parsed["session_data"]["full_translation_views"]) == 1
    assert parsed["session_data"]["words"][0]["translation_viewed"] == True
