#!/usr/bin/env python3
"""
Clean, Pareto efficient testing conftest.py - demonstrates the 80/20 approach.

Key principles:
- Use in-memory SQLite for fast, realistic testing with real constraints
- Focus on API endpoint testing (the real "unit") not internal functions  
- Only mock external dependencies (LLM calls)
- Test the happy path and error conditions that matter
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from server.models import Base, Account, Profile, ReadingText, Lexeme
from server.auth import create_access_token


@pytest.fixture(scope="session")
def db_engine():
    """In-memory SQLite with foreign key enforcement for realistic constraint testing."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")  # Enforce real constraints
        cursor.close()
    
    return engine


@pytest.fixture(scope="function") 
def db(db_engine):
    """Fresh database for each test function."""
    Base.metadata.create_all(bind=db_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=db_engine)


@pytest.fixture
def client(db):
    """FastAPI test client that uses our test database."""
    # Import here to avoid circular imports during setup
    from server.main import app  
    from server.db import SessionLocal
    
    # Override the app's database dependency
    def override_get_db():
        yield db
    
    # Replace the dependency temporarily
    original_get_db = app.dependency_overrides.get(SessionLocal)
    app.dependency_overrides[SessionLocal] = override_get_db
    
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Restore original dependency
        if original_get_db:
            app.dependency_overrides[SessionLocal] = original_get_db
        elif SessionLocal in app.dependency_overrides:
            del app.dependency_overrides[SessionLocal]


@pytest.fixture
def test_user(db):
    """Create a test user with account and language profile."""
    # Create account
    account = Account(
        email="test@example.com",
        password_hash="hashed_password",
        subscription_tier="Standard",
        is_active=True,
        is_verified=True
    )
    db.add(account)
    db.flush()
    
    # Create language learning profile  
    profile = Profile(
        account_id=account.id,
        lang="es",
        target_lang="en", 
        level_value=3.0,
        level_code="B1"
    )
    db.add(profile)
    db.commit()
    
    return account, profile


@pytest.fixture  
def auth_headers(test_user):
    """JWT auth headers for API requests."""
    account, profile = test_user
    token = create_access_token(data={"sub": str(account.id)}, secret="dev-secret-change")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_llm_response():
    """Sample LLM response for testing."""
    return {
        "text": "Hola mundo. Este es un texto de ejemplo.",
        "content": "Hola mundo. Este es un texto de ejemplo para aprender español.",
        "translations": [
            {"surface": "Hola", "translation": "Hello"},
            {"surface": "mundo", "translation": "world"}
        ]
    }


# Helper functions for common test patterns
def create_test_text(db, account_id, lang="es", target_lang="en"):
    """Create a test reading text."""
    text = ReadingText(
        account_id=account_id,
        lang=lang,
        target_lang=target_lang,
        content="Test text for learning Spanish.",
        source="test",
        words_complete=True,
        sentences_complete=True
    )
    db.add(text) 
    db.commit()
    db.refresh(text)
    return text


def create_test_lexeme(db, profile_id, surface="hola", lemma="hola"):
    """Create a test vocabulary entry."""
    lexeme = Lexeme(
        account_id=1,  # Would be profile.account_id in real usage
        profile_id=profile_id,
        lang="es",
        surface=surface,
        lemma=lemma,
        pos="INTERJECTION",
        exposures=1,
        clicks=1
    )
    db.add(lexeme)
    db.commit()
    db.refresh(lexeme)
    return lexeme


# Test markers for selecting test subsets
def test_smoke():
    """Mark smoke tests (startup, basic functionality)."""
    return pytest.mark.smoke


def test_flow():
    """Mark integration flow tests (user journey)."""  
    return pytest.mark.flow


def test_parsing():
    """Mark parsing logic tests (LLM input handling)."""
    return pytest.mark.parsing


# Pytest configuration
pytest_plugins = []

# Marker definitions for pytest -m syntax
pytest_marks = [
    "smoke: Startup and basic functionality tests", 
    "flow: User journey integration tests",
    "parsing: LLM parsing and data extraction tests"
]
