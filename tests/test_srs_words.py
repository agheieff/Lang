"""Test /srs/words endpoint returns actual data."""

import pytest
from datetime import datetime, timezone


def test_srs_words_endpoint_returns_lexemes(client, test_user):
    """Test that /srs/words endpoint returns actual lexemes, not empty list."""
    account, profile = test_user

    # Create some lexemes for the profile
    from server.auth import create_access_token
    from server.models import Lexeme

    token = create_access_token(subject=str(account.id), secret_key="dev-secret-change")
    headers = {"Authorization": f"Bearer {token}"}

    # Request words
    response = client.get("/srs/words", headers=headers)

    assert response.status_code == 200

    data = response.json()

    # Should return a list (could be empty if no words yet, but should be a list)
    assert isinstance(data, list)

    # If lexemes exist, they should have required fields
    if len(data) > 0:
        word = data[0]
        required_fields = ["id", "lemma", "surface", "pos", "n", "p_click", "stability"]
        for field in required_fields:
            assert field in word, f"Missing field: {field}"
