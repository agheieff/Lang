"""Integration tests for critical database flows.

These tests verify that database operations work correctly with real schema constraints.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.db import Base
from server.models import (
    Profile, 
    ReadingText, 
    ReadingLookup, 
    ReadingWordGloss,
    ReadingTextTranslation,
    Lexeme,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    from sqlalchemy import event
    
    engine = create_engine("sqlite:///:memory:", echo=False)
    
    # Enable FK enforcement in SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


class TestReadingLookupCreation:
    """Test ReadingLookup model constraints."""
    
    def test_lookup_requires_lang_and_target_lang(self, db_session):
        """Verify that ReadingLookup requires lang and target_lang fields."""
        # First create a ReadingText (required for FK)
        text = ReadingText(
            account_id=1,
            lang="zh",
            content="Test content",
        )
        db_session.add(text)
        db_session.flush()
        
        # This should work - all required fields present
        lookup = ReadingLookup(
            account_id=1,
            text_id=text.id,
            lang="zh",
            target_lang="en",
            surface="你好",
            span_start=0,
            span_end=2,
        )
        db_session.add(lookup)
        db_session.commit()
        
        # Verify it was saved
        saved = db_session.query(ReadingLookup).first()
        assert saved is not None
        assert saved.lang == "zh"
        assert saved.target_lang == "en"
    
    def test_lookup_fk_constraint(self, db_session):
        """Verify that ReadingLookup enforces FK to ReadingText."""
        # Try to create lookup with non-existent text_id
        lookup = ReadingLookup(
            account_id=1,
            text_id=99999,  # Doesn't exist
            lang="zh",
            target_lang="en",
            surface="你好",
            span_start=0,
            span_end=2,
        )
        db_session.add(lookup)
        
        # Should fail on commit due to FK constraint
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()


class TestTextStateTransitions:
    """Test ReadingText state transitions."""
    
    def test_text_lifecycle(self, db_session):
        """Test complete text lifecycle: create -> open -> read."""
        # Create placeholder
        text = ReadingText(
            account_id=1,
            lang="zh",
            content=None,  # Placeholder
            pooled=True,
        )
        db_session.add(text)
        db_session.commit()
        
        assert text.content is None
        assert text.opened_at is None
        assert text.read_at is None
        assert text.pooled is True
        
        # Simulate generation complete
        text.content = "Generated content"
        text.generated_at = datetime.now(timezone.utc)
        db_session.commit()
        
        assert text.content == "Generated content"
        assert text.generated_at is not None
        
        # Simulate user opens text
        text.opened_at = datetime.now(timezone.utc)
        text.pooled = False
        db_session.commit()
        
        assert text.opened_at is not None
        assert text.pooled is False
        
        # Simulate user finishes text
        text.is_read = True
        text.read_at = datetime.now(timezone.utc)
        db_session.commit()
        
        assert text.is_read is True
        assert text.read_at is not None
    
    def test_pool_query_filters(self, db_session):
        """Test that pool queries correctly filter texts."""
        # Create texts in different states
        pooled_ready = ReadingText(
            account_id=1, lang="zh", content="Ready", pooled=True, opened_at=None
        )
        pooled_generating = ReadingText(
            account_id=1, lang="zh", content=None, pooled=True, opened_at=None
        )
        opened = ReadingText(
            account_id=1, lang="zh", content="Opened", pooled=False, 
            opened_at=datetime.now(timezone.utc)
        )
        
        db_session.add_all([pooled_ready, pooled_generating, opened])
        db_session.commit()
        
        # Query for pool (ready texts only)
        pool = db_session.query(ReadingText).filter(
            ReadingText.account_id == 1,
            ReadingText.lang == "zh",
            ReadingText.pooled == True,
            ReadingText.content.isnot(None),
            ReadingText.opened_at.is_(None),
        ).all()
        
        assert len(pool) == 1
        assert pool[0].content == "Ready"


class TestProfilePreferences:
    """Test Profile model with new fields."""
    
    def test_profile_defaults(self, db_session):
        """Test that Profile has correct default values."""
        profile = Profile(
            account_id=1,
            lang="zh",
            target_lang="en",
        )
        db_session.add(profile)
        db_session.commit()
        
        # Check defaults
        assert profile.ci_preference == 0.92
        assert profile.topic_weights is not None
        assert isinstance(profile.topic_weights, dict)
        # Default is empty dict - services use config.DEFAULT_TOPIC_WEIGHTS when empty
        assert profile.topic_weights == {}
    
    def test_profile_ci_preference_update(self, db_session):
        """Test updating CI preference."""
        profile = Profile(
            account_id=1,
            lang="zh", 
            target_lang="en",
        )
        db_session.add(profile)
        db_session.commit()
        
        # Update CI preference
        profile.ci_preference = 0.88
        db_session.commit()
        
        # Verify persistence
        db_session.expire(profile)
        assert profile.ci_preference == 0.88
    
    def test_profile_topic_weights_update(self, db_session):
        """Test updating topic weights."""
        profile = Profile(
            account_id=1,
            lang="zh",
            target_lang="en",
        )
        db_session.add(profile)
        db_session.commit()
        
        # Update topic weights
        weights = profile.topic_weights.copy()
        weights["fiction"] = 1.5
        weights["news"] = 0.8
        profile.topic_weights = weights
        db_session.commit()
        
        # Verify persistence
        db_session.expire(profile)
        assert profile.topic_weights["fiction"] == 1.5
        assert profile.topic_weights["news"] == 0.8


class TestLexemeSRS:
    """Test Lexeme SRS fields."""
    
    def test_lexeme_srs_defaults(self, db_session):
        """Test Lexeme SRS field defaults."""
        # Need a profile first
        profile = Profile(account_id=1, lang="zh", target_lang="en")
        db_session.add(profile)
        db_session.flush()
        
        lexeme = Lexeme(
            account_id=1,
            profile_id=profile.id,
            lang="zh",
            lemma="你好",
        )
        db_session.add(lexeme)
        db_session.commit()
        
        # Check SRS defaults
        assert lexeme.alpha == 1.0
        assert lexeme.beta == 9.0
        assert lexeme.stability == 0.2
        assert lexeme.exposures == 0
        assert lexeme.clicks == 0
    
    def test_lexeme_srs_update(self, db_session):
        """Test updating Lexeme SRS fields."""
        profile = Profile(account_id=1, lang="zh", target_lang="en")
        db_session.add(profile)
        db_session.flush()
        
        lexeme = Lexeme(
            account_id=1,
            profile_id=profile.id,
            lang="zh",
            lemma="你好",
        )
        db_session.add(lexeme)
        db_session.commit()
        
        # Simulate click event
        lexeme.alpha += 1.0
        lexeme.clicks += 1
        lexeme.stability = 0.5
        db_session.commit()
        
        # Verify
        db_session.expire(lexeme)
        assert lexeme.alpha == 2.0
        assert lexeme.clicks == 1
        assert lexeme.stability == 0.5


class TestReadingWordGloss:
    """Test ReadingWordGloss constraints."""
    
    def test_word_gloss_creation(self, db_session):
        """Test creating word glosses."""
        text = ReadingText(account_id=1, lang="zh", content="你好世界")
        db_session.add(text)
        db_session.flush()
        
        gloss = ReadingWordGloss(
            account_id=1,
            text_id=text.id,
            lang="zh",
            surface="你好",
            translation="hello",
            span_start=0,
            span_end=2,
        )
        db_session.add(gloss)
        db_session.commit()
        
        saved = db_session.query(ReadingWordGloss).first()
        assert saved.surface == "你好"
        assert saved.translation == "hello"
    
    def test_word_gloss_unique_constraint(self, db_session):
        """Test that duplicate spans are rejected."""
        text = ReadingText(account_id=1, lang="zh", content="你好世界")
        db_session.add(text)
        db_session.flush()
        
        gloss1 = ReadingWordGloss(
            account_id=1, text_id=text.id, lang="zh",
            surface="你好", span_start=0, span_end=2,
        )
        db_session.add(gloss1)
        db_session.commit()
        
        # Try to add duplicate span
        gloss2 = ReadingWordGloss(
            account_id=1, text_id=text.id, lang="zh",
            surface="你好", span_start=0, span_end=2,  # Same span
        )
        db_session.add(gloss2)
        
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()
