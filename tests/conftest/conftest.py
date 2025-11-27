"""Pytest configuration for Arcadia Lang tests."""

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment
os.environ["ARC_LANG_ENVIRONMENT"] = "test"
os.environ["ARC_LANG_JWT_SECRET"] = "test-secret"

from server.main import app
from server.db import get_global_db, Base
from server.auth.models import Account
from server.models import Profile, Lexeme, ReadingText


@pytest.fixture(scope="session")
def temp_db() -> Generator[str, None, None]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_db_path = f.name
    
    yield temp_db_path
    
    # Cleanup
    if os.path.exists(temp_db_path):
        os.unlink(temp_db_path)


@pytest.fixture(scope="session")
def engine(temp_db: str):
    """Create test database engine."""
    engine = create_engine(f"sqlite:///{temp_db}", echo=False)
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(engine):
    """Create a fresh database session for each test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db_session) -> Generator[TestClient, None, None]:
    """Create a test client with database dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_global_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def async_client(db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_global_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as async_client:
        yield async_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def sample_account(db_session) -> Account:
    """Create a sample account for testing."""
    account = Account(
        email="test@example.com",
        password_hash="hashed_password",
        is_active=True,
        subscription_tier="free"
    )
    db_session.add(account)
    db_session.commit()
    db_session.refresh(account)
    return account


@pytest.fixture
def sample_profile(db_session, sample_account) -> Profile:
    """Create a sample profile for testing."""
    profile = Profile(
        account_id=sample_account.id,
        lang="es",
        target_lang="en",
    )
    db_session.add(profile)
    db_session.commit()
    db_session.refresh(profile)
    return profile


@pytest.fixture
def sample_lexeme(db_session, sample_account, sample_profile) -> Lexeme:
    """Create a sample lexeme for testing."""
    lexeme = Lexeme(
        account_id=sample_account.id,
        profile_id=sample_profile.id,
        lang="es",
        lemma="casa",
        pos="noun",
        familiarity=0.5
    )
    db_session.add(lexeme)
    db_session.commit()
    db_session.refresh(lexeme)
    return lexeme


@pytest.fixture
def sample_reading_text(db_session, sample_account) -> ReadingText:
    """Create a sample reading text for testing."""
    reading_text = ReadingText(
        account_id=sample_account.id,
        lang="es",
        content="Esta es una casa. La casa es grande.",
    )
    db_session.add(reading_text)
    db_session.commit()
    db_session.refresh(reading_text)
    return reading_text


@pytest.fixture
def sample_lexeme_with_data(db_session, sample_account, sample_profile) -> Lexeme:
    """Create a sample lexeme with SRS data for testing."""
    lexeme = Lexeme(
        account_id=sample_account.id,
        profile_id=sample_profile.id,
        lang="es",
        lemma="gato",
        pos="noun",
        familiarity=0.6,
        clicks=5,
        exposures=10
    )
    db_session.add(lexeme)
    db_session.commit()
    db_session.refresh(lexeme)
    return lexeme


@pytest.fixture
def auth_headers(sample_account) -> dict:
    """Create authorization headers for authenticated requests."""
    from jose import jwt
    from datetime import datetime, timedelta, timezone
    
    payload = {
        "sub": str(sample_account.id),
        "email": sample_account.email,
        "tier": sample_account.subscription_tier,
        "exp": datetime.now(timezone.utc) + timedelta(hours=1)
    }
    
    token = jwt.encode(payload, "test-secret", algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "srs: Spaced Repetition System tests")
    config.addinivalue_line("markers", "llm: LLM service tests")
    config.addinivalue_line("markers", "auth: Authentication tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Add markers to test items based on their location."""
    for item in items:
        # Add marker based on directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add specific markers based on test content
        if "srs" in item.name.lower():
            item.add_marker(pytest.mark.srs)
        if "llm" in item.name.lower():
            item.add_marker(pytest.mark.llm)
        if "auth" in item.name.lower():
            item.add_marker(pytest.mark.auth)
