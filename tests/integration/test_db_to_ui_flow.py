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
from server.views.reading_renderer import render_loading_block
from server.routes.reading import get_translations, sync_session_state
from server.schemas.session import TextSessionState

@pytest.fixture
def db_session():
    # Setup in-memory SQLite db
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
    account = Account(id=1001, email="ui_test@example.com", password_hash="hash", subscription_tier="Free")
    db_session.add(account)
    db_session.flush()
    profile = Profile(account_id=1001, lang="es", target_lang="en", current_text_id=None)
    db_session.add(profile)
    db_session.commit()
    return account

def test_reading_view_html_structure(db_session, test_user):
    """
    Verify that data from DB is correctly embedded into the HTML structure.
    """
    # 1. Setup Data
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola mundo.")
    db_session.add(rt)
    db_session.flush()
    
    # Add Profile reference
    prof = db_session.query(Profile).filter_by(account_id=test_user.id).first()
    prof.current_text_id = rt.id
    
    # Add Title
    # We'll mock TitleExtractionService for simplicity, or add data if we want to test that too.
    # Let's patch the service method to return known title data
    
    # Add Words
    word = ReadingWordGloss(
        account_id=test_user.id,
        text_id=rt.id,
        lang="es",
        surface="mundo",
        translation="world",
        span_start=5,
        span_end=10
    )
    db_session.add(word)
    db_session.commit()

    service = ReadingViewService()
    
    # Mock dependencies that aren't under test here
    with patch.object(service.title_service, 'get_title', return_value=("Hola Title", "Hello Title")):
        with patch.object(service.title_service, 'get_title_words', return_value=[]):
            # Mock ReadinessService to avoid "no sentences" retry logic affecting status
            with patch.object(service.readiness_service, 'evaluate', return_value=(True, "both")):
                # Force selection service to pick our text
                with patch.object(service.selection_service, 'pick_current_or_new', return_value=rt):
                    
                    # 2. Action
                    context = service.get_current_reading_context(db_session, test_user.id)
                    
                    # 3. Render (call the renderer logic directly or via context)
                    from server.views.reading_renderer import render_reading_block
                    html = render_reading_block(
                        context.text_id,
                        context.content,
                        context.words,
                        title=context.title,
                        title_words=context.title_words,
                        title_translation=context.title_translation
                    )

    # 4. Assertions
    assert "Hola mundo." in html
    assert "Hola Title" in html
    
    # Check for embedded JSON data (the "DB -> UI" handoff)
    assert 'id="reading-words-json"' in html
    
    # Parse the embedded JSON to verify structure
    import re
    match = re.search(r'<script id="reading-words-json" type="application/json">(.*?)</script>', html, re.DOTALL)
    assert match is not None
    json_data = json.loads(match.group(1))
    
    assert len(json_data) == 1
    assert json_data[0]['surface'] == "mundo"
    assert json_data[0]['translation'] == "world"
    assert json_data[0]['span_start'] == 5
    
    # Check title translation embed
    assert 'id="reading-title-translation"' in html
    assert "Hello Title" in html

@pytest.mark.asyncio
async def test_sentence_translations_api(db_session, test_user):
    """
    Verify that the sentence translation API returns DB data in correct format.
    """
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola. Mundo.")
    db_session.add(rt)
    db_session.flush()
    
    trans = ReadingTextTranslation(
        account_id=test_user.id,
        text_id=rt.id,
        unit=TextUnit.SENTENCE,
        target_lang="en",
        segment_index=0,
        span_start=0,
        span_end=4,
        source_text="Hola.",
        translated_text="Hello."
    )
    db_session.add(trans)
    db_session.commit()
    
    # Call the API function directly (simulating route)
    result = await get_translations(
        text_id=rt.id,
        unit="sentence",
        target_lang="en",
        db=db_session,
        account=test_user
    )
    
    assert result['unit'] == "sentence"
    assert result['target_lang'] == "en"
    assert len(result['items']) == 1
    
    item = result['items'][0]
    assert item['source'] == "Hola."
    assert item['translation'] == "Hello."
    assert item['start'] == 0
    assert item['end'] == 4

def test_renderer_loading_state():
    """
    Verify the loading state HTML structure.
    """
    html = render_loading_block("generating")
    assert "Generating text" in html
    assert 'id="reading-block"' in html
    assert "animate-pulse" in html

@pytest.mark.asyncio
async def test_see_translation_data_availability(db_session, test_user):
    """
    Verify full text translation availability for the 'See Translation' button.
    """
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola mundo.")
    db_session.add(rt)
    db_session.flush()
    
    # Translation for the whole text
    trans = ReadingTextTranslation(
        account_id=test_user.id,
        text_id=rt.id,
        unit="text",  # Note: DB stores string enum usually
        target_lang="en",
        source_text="Hola mundo.",
        translated_text="Hello world.",
        segment_index=0
    )
    db_session.add(trans)
    db_session.commit()
    
    # The "See Translation" button fetches unit='text' (or paragraph)
    result = await get_translations(
        text_id=rt.id,
        unit="text",
        target_lang="en",
        db=db_session,
        account=test_user
    )
    
    assert result['items'][0]['translation'] == "Hello world."

@pytest.mark.asyncio
async def test_sync_session_nested_schema(db_session, test_user):
    """
    Verify that the sync endpoint accepts the nested 'Big JSON' structure.
    """
    rt = ReadingText(account_id=test_user.id, lang="es", content="Hola world")
    db_session.add(rt)
    db_session.commit()
    
    # Mock payload matching home.html structure
    payload = {
        "session_id": "test_sess_123",
        "text_id": rt.id,
        "lang": "es",
        "target_lang": "en",
        "opened_at": 1234567890,
        "paragraphs": [
            {
                "text": "Hola world",
                "sentences": [
                    {
                        "text": "Hola world",
                        "words": [
                            {"surface": "Hola", "looked_up_at": 12345},
                            {"surface": "world", "looked_up_at": None}
                        ]
                    }
                ]
            }
        ]
    }
    
    # Validate via Pydantic model first
    state = TextSessionState(**payload)
    assert len(state.paragraphs) == 1
    assert len(state.paragraphs[0].sentences[0].words) == 2
    assert state.paragraphs[0].sentences[0].words[0].looked_up_at == 12345
    
    # Call the endpoint function
    result = await sync_session_state(
        state=state,
        db=db_session,
        account=test_user
    )
    assert result["ok"] is True
    
    # Verify persistence
    from server.models import ProfilePref
    pref = db_session.query(ProfilePref).first()
    assert pref is not None
    data = pref.data.get("current_session")
    assert data["session_id"] == "test_sess_123"
    assert data["paragraphs"][0]["sentences"][0]["words"][0]["looked_up_at"] == 12345
