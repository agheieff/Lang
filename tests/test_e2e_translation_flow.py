import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server.db import Base
from server.auth.models import Account
from server.auth.models import Base as AuthBase
from server.models import Profile, ReadingText, ReadingTextTranslation
from server.services.translation_service import TranslationService
from server.routes.reading import get_translations

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    AuthBase.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def test_user(db_session):
    account = Account(id=1, email="e2e_test@example.com", password_hash="hash", subscription_tier="Free")
    db_session.add(account)
    db_session.flush()
    profile = Profile(account_id=1, lang="es", target_lang="en")
    db_session.add(profile)
    db_session.commit()
    return account

@pytest.mark.asyncio
async def test_e2e_full_text_translation_synthesis(db_session, test_user):
    """
    E2E Test verifying the gap: Generation -> DB Persistence -> API Retrieval.
    Ensures that generating sentence translations ALSO synthesizes and persists
    a full-text translation that the API can serve (global model).
    """
    # 1. Setup: Existing text waiting for translation (global model - no account_id)
    rt = ReadingText(lang="es", target_lang="en", content="Hola mundo. Esto es una prueba.")
    db_session.add(rt)
    db_session.commit()
    
    service = TranslationService()
    
    # 2. Execution: Trigger Translation Generation (Mocking LLM)
    structured_json = """
    {
        "paragraphs": [
            {
                "sentences": [
                    {"text": "Hola mundo.", "translation": "Hello world."},
                    {"text": "Esto es una prueba.", "translation": "This is a test."}
                ]
            }
        ]
    }
    """
    
    with patch("server.services.translation_service.llm_call_and_log_to_file") as mock_llm:
        mock_llm.side_effect = [
            ("{}", {}, "mock", "mock"),  # words call 1
            ("{}", {}, "mock", "mock"),  # words call 2
            (structured_json, {}, "mock", "mock"),  # sentences (CRITICAL)
            ("{}", {}, "mock", "mock"),  # title
            ("{}", {}, "mock", "mock")   # extra safety
        ]
        
        # Pass same session for both account_db and global_db in test
        service.generate_translations(
            db_session, db_session, test_user.id, "es", rt.id, rt.content, "Test Title",
            MagicMock(), [{"role": "user", "content": "dummy"}, {"role": "user", "content": "dummy"}]
        )
        
    # 3. Verification Point 1: Database State (global model)
    full_text_db = db_session.query(ReadingTextTranslation).filter(
        ReadingTextTranslation.text_id == rt.id,
        ReadingTextTranslation.target_lang == "en",
        ReadingTextTranslation.unit == "text",
        ReadingTextTranslation.segment_index == 1
    ).first()
    
    assert full_text_db is not None, "Full text translation was not synthesized/persisted in DB"
    assert full_text_db.translated_text == "Hello world. This is a test."
    
    # 4. Verification Point 2: API Access
    api_response = await get_translations(
        text_id=rt.id,
        unit="text",
        target_lang="en",
        db=db_session,
        account=test_user
    )
    
    assert api_response["unit"] == "text"
    assert len(api_response["items"]) >= 1
    
    matched_item = next(
        (item for item in api_response["items"] if "Hello world" in item["translation"]), 
        None
    )
    
    assert matched_item is not None, "API did not return the full text translation"
    assert matched_item["translation"] == "Hello world. This is a test."

    print("\n✅ E2E Test Passed: Generation -> Synthesis -> DB -> API Flow is intact.")
