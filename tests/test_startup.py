"""Smoke tests to catch startup issues and basic functionality."""

import pytest
from unittest.mock import patch


def test_app_startup(client):
    """Smoke test: Does the app boot and serve the health check?"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data.get("status") == "ok"


def test_static_files_available(client):
    """Ensure static files routing works (404 or 200, not 500)."""
    # Test that static directory routing exists (may return 200 or 404)
    response = client.get("/static/non-existent.js")
    # Should return 404 or 200, not 500 (server error would indicate mount issue)
    assert response.status_code in [200, 404, 403]


def test_database_connection(db):
    """Test database connection and basic functionality."""
    # Test that we can create and query records
    from server.models import Profile

    # Create a test record
    profile = Profile(account_id=1, lang="es", target_lang="en", level_value=2.0)
    db.add(profile)
    db.commit()

    # Query it back
    retrieved = db.query(Profile).filter(Profile.lang == "es").first()
    assert retrieved is not None
    assert retrieved.target_lang == "en"
    assert retrieved.level_value == 2.0


def test_foreign_key_constraints(db):
    """Test that foreign key constraints are enforced in test DB."""
    from server.models import Profile, ReadingText

    # Try to create a ReadingText with non-existent account_id
    with pytest.raises(Exception):  # Should raise an integrity error
        text = ReadingText(
            account_id=999,  # Non-existent account
            lang="es",
            target_lang="en",
            content="test content",
        )
        db.add(text)
        db.commit()


def test_jinja2_templates_mounted(client):
    """Ensure Jinja2 is configured and templates can be rendered."""
    # Test a basic page load - this will fail if templates are misconfigured
    response = client.get("/")

    # Should either redirect, show login, or show the main page
    assert response.status_code in [200, 302]

    # If it returns HTML, check it's not an error page
    if response.status_code == 200:
        assert "server error" not in response.text.lower()


def test_auth_dependencies_available():
    """Test that auth system components are importable."""
    from server.auth import Account, create_access_token
    from server.models import Profile

    # Create a test account
    token = create_access_token(subject="1", secret_key="dev-secret-change")
    assert isinstance(token, str)
    assert len(token) > 20  # JWT tokens are substantial strings


def test_llm_client_imports():
    """Test that LLM client can be imported without configuration errors."""
    from server.llm.client import chat_complete, get_llm_config

    # Test configuration loading
    config = get_llm_config()
    assert "models" in config
    assert isinstance(config["models"], list)
    assert len(config["models"]) > 0
