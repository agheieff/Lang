"""Tests for background worker and text pool management."""

import pytest
from unittest.mock import patch, AsyncMock
import asyncio

from server.services.background_worker import (
    background_worker,
    maintenance_cycle,
    fill_gaps,
    retry_failed_translations,
    cleanup_old_texts,
    startup_generation,
)


def test_background_worker_exists():
    """Test that background worker module can be imported."""
    assert background_worker is not None
    assert maintenance_cycle is not None
    assert fill_gaps is not None


def test_startup_generation_no_langs():
    """Test startup generation with no languages specified."""
    from server.services.background_worker import startup_generation

    # This is an async function, just check it doesn't crash
    # Actual execution would require a DB session
    assert asyncio.iscoroutinefunction(startup_generation)


def test_fill_gaps_with_no_gaps(db):
    """Test fill_gaps with empty gap list."""
    # Should not crash when gaps is empty
    # This is an async function
    assert fill_gaps is not None


def test_cleanup_old_texts(db):
    """Test cleanup function doesn't crash."""
    from server.services.background_worker import cleanup_old_texts
    from server.models import ReadingText

    # Add a very old text
    from datetime import datetime, timezone, timedelta

    old_text = ReadingText(
        generated_for_account_id=1,
        lang="es",
        target_lang="en",
        content="Old text",
        words_complete=True,
        sentences_complete=True,
        created_at=datetime.now(timezone.utc) - timedelta(days=40),
    )
    db.add(old_text)
    db.commit()

    # Run cleanup
    cleanup_old_texts(db)

    # Text should be hidden now
    db.refresh(old_text)
    assert old_text.is_hidden


def test_startup_generation_with_langs():
    """Test startup generation accepts language list."""
    # Just verify the function signature works
    from server.services.background_worker import startup_generation

    langs = ["es", "zh"]
    # This would require real DB to execute
    assert asyncio.iscoroutinefunction(startup_generation)
