import pytest
import json
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server.db import Base
from server.auth.models import Account
from server.auth.models import Base as AuthBase
from server.models import ReadingText, ReadingTextTranslation, ReadingWordGloss, Profile
from server.enums import TextUnit
from server.services.reading_view_service import ReadingViewService

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
    account = Account(id=1, email="nextbtn_test@example.com", password_hash="hash", subscription_tier="Free")
    db_session.add(account)
    db_session.flush()
    profile = Profile(account_id=1, lang="es", target_lang="en")
    db_session.add(profile)
    db_session.commit()
    return account

def test_reading_context_is_fully_ready(db_session, test_user):
    """
    Verify that ReadingViewService correctly flags a text as fully ready
    when it has content, words, and sentence translations (global model).
    """
    # 1. Setup Data (global model - no account_id)
    rt = ReadingText(lang="es", target_lang="en", content="Hola mundo.")
    db_session.add(rt)
    db_session.flush()
    
    # Add Sentence Translation (global model)
    trans = ReadingTextTranslation(
        text_id=rt.id,
        target_lang="en",
        unit=TextUnit.SENTENCE,
        segment_index=0,
        source_text="Hola mundo.",
        translated_text="Hello world."
    )
    db_session.add(trans)
    
    # Add Word Gloss (global model)
    word = ReadingWordGloss(
        text_id=rt.id,
        target_lang="en",
        lang="es",
        surface="mundo",
        span_start=5,
        span_end=10
    )
    db_session.add(word)
    db_session.commit()
    
    service = ReadingViewService()
    
    # Mock selection to return our text
    with patch.object(service.selection_service, 'pick_current_or_new', return_value=rt):
        with patch.object(service.title_service, 'get_title', return_value=("Title", "Trans")):
            with patch.object(service.title_service, 'get_title_words', return_value=[]):
                
                # Action - pass same session as both account_db and global_db for test
                context = service.get_current_reading_context(db_session, db_session, test_user.id)
                
                # Assertion
                assert context.text_id == rt.id
                assert context.is_fully_ready is True

def test_readiness_evaluation_logic(db_session, test_user):
    """
    Directly test the ReadinessService evaluate logic (global model).
    """
    from server.services.text_state_service import TextStateService, ReadinessStatus
    rs = TextStateService()
    
    # Global model - no account_id
    rt = ReadingText(lang="es", target_lang="en", content="Hola.")
    db_session.add(rt)
    db_session.flush()
    db_session.commit()
    
    # No translations yet
    ready, reason = rs.evaluate(db_session, rt, target_lang="en")
    assert ready is False
    
    # Add Word (global model)
    db_session.add(ReadingWordGloss(
        text_id=rt.id, target_lang="en", lang="es", 
        surface="Hola", span_start=0, span_end=4
    ))
    db_session.commit()
    
    ready, reason = rs.evaluate(db_session, rt, target_lang="en")
    assert ready is False  # Needs sentences too
    
    # Add Sentence (global model)
    db_session.add(ReadingTextTranslation(
        text_id=rt.id, 
        target_lang="en",
        unit=TextUnit.SENTENCE, 
        source_text="Hola.",
        translated_text="Hello."
    ))
    db_session.commit()
    
    ready, reason = rs.evaluate(db_session, rt, target_lang="en")
    assert ready is True
    assert reason == "both"
