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
    when it has content, words, and sentence translations.
    """
    # 1. Setup Data
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola mundo.")
    db_session.add(rt)
    db_session.flush()
    
    # Add Sentence Translation
    trans = ReadingTextTranslation(
        account_id=test_user.id,
        text_id=rt.id,
        unit=TextUnit.SENTENCE,
        target_lang="en",
        segment_index=0,
        source_text="Hola mundo.",
        translated_text="Hello world."
    )
    db_session.add(trans)
    
    # Add Word Gloss
    word = ReadingWordGloss(
        account_id=test_user.id,
        text_id=rt.id,
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
                
                # Action
                context = service.get_current_reading_context(db_session, test_user.id)
                
                # Assertion
                assert context.text_id == rt.id
                assert context.is_fully_ready is True

def test_readiness_evaluation_logic(db_session, test_user):
    """
    Directly test the ReadinessService evaluate logic.
    """
    from server.services.readiness_service import ReadinessService
    rs = ReadinessService()
    
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola.")
    db_session.add(rt)
    db_session.flush()
    
    # No translations yet
    # Note: We need to commit before evaluating because evaluate() runs new queries against DB
    db_session.commit()
    
    ready, reason = rs.evaluate(db_session, rt, test_user.id)
    assert ready is False
    
    # Add Word
    db_session.add(ReadingWordGloss(account_id=test_user.id, text_id=rt.id, lang="es", surface="Hola", span_start=0, span_end=4))
    db_session.commit()
    
    ready, reason = rs.evaluate(db_session, rt, test_user.id)
    assert ready is False # Needs sentences too
    
    # Add Sentence
    db_session.add(ReadingTextTranslation(
        account_id=test_user.id, 
        text_id=rt.id, 
        unit=TextUnit.SENTENCE, 
        target_lang="en",
        source_text="Hola.",
        translated_text="Hello."
    ))
    db_session.commit()
    
    ready, reason = rs.evaluate(db_session, rt, test_user.id)
    assert ready is True
    assert reason == "both"
