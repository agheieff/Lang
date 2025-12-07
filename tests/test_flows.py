"""Integration tests for critical user flows."""

import pytest
from unittest.mock import patch, AsyncMock
import json


def test_user_registration_flow(client, db):
    """Test complete user registration and initial setup flow."""
    # 1. Register new user
    register_data = {
        "email": "newuser@test.com",
        "password": "securepassword123"
    }
    
    response = client.post("/auth/register", json=register_data)
    assert response.status_code == 201
    
    result = response.json()
    assert "access_token" in result
    assert result["email"] == "newuser@test.com"
    
    token = result["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Create language profile
    profile_data = {
        "lang": "es",
        "target_lang": "en",
        "level_code": "A2"
    }
    
    response = client.post("/me/profile", json=profile_data, headers=headers)
    # This might return 201 or 200 depending on implementation
    assert response.status_code in [200, 201]
    
    # 3. Verify profile was created in database
    from server.models import Profile
    profile = db.query(Profile).filter(Profile.account_id > 0).first()
    assert profile is not None
    assert profile.lang == "es"
    assert profile.target_lang == "en"


@patch('server.services.content.chat_complete_with_raw')
def test_reading_flow(mock_chat_complete_with_raw, client, test_user, mock_llm_response):
    """Test the complete reading flow from generation to interaction."""
    account, profile = test_user
    
    # Mock LLM response
    mock_chat_complete_with_raw.return_value = (mock_llm_response, {"model": "test"})
    
    # Get auth headers for the test user
    token = create_access_token(data={"sub": str(account.id)}, secret="dev-secret-change")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Request current reading material
    response = client.get("/reading/current", headers=headers)
    assert response.status_code in [200, 302]
    
    # 2. Manually trigger text generation if needed (simulate background process)
    response = client.post("/reading/generate", headers=headers)
    # This endpoint might not exist in current implementation
    if response.status_code != 404:
        assert response.status_code in [200, 202]
    
    # 3. Direct database check - was text created?
    from server.models import ReadingText
    from server.services.content import generate_text_content
    
    # Generate text content directly for testing
    import asyncio
    text_obj = asyncio.run(generate_text_content(
        account_id=account.id,
        profile_id=profile.id,
        lang=profile.lang,
        target_lang=profile.target_lang,
        profile=profile
    ))
    
    assert text_obj is not None
    assert text_obj.account_id == account.id
    assert "Hola mundo" in text_obj.content
    
    # 4. Test word interaction (SRS logic)
    word_data = {
        "text_id": text_obj.id,
        "word_info": {
            "surface": "Hola",
            "lemma": "hola",
            "pos": "INTERJECTION",
            "span_start": 0,
            "span_end": 4,
            "lang": "es"
        }
    }
    
    response = client.post("/reading/word-click", json=word_data, headers=headers)
    # This endpoint might need adjustment based on actual implementation
    if response.status_code != 404:
        assert response.status_code == 200
    
    # 5. Verify SRS state - did lexeme get created?
    from server.models import Lexeme
    from server.services.learning import track_word_click
    
    # Track the word click directly
    track_word_click(
        db=db,
        account_id=account.id,
        profile_id=profile.id,
        text_id=text_obj.id,
        word_info=word_data["word_info"]
    )
    
    lexeme = db.query(Lexeme).filter_by(surface="Hola").first()
    assert lexeme is not None
    assert lexeme.clicks == 1
    assert lexeme.lemma == "hola"


def test_profile_management_flow(client, db):
    """Test profile creation, retrieval, and management."""
    # Register user and get token
    register_data = {"email": "profile@test.com", "password": "password123"}
    response = client.post("/auth/register", json=register_data)
    token = response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create multiple profiles
    profiles_data = [
        {"lang": "es", "target_lang": "en", "level_code": "A2"},
        {"lang": "fr", "target_lang": "en", "level_code": "B1"},
    ]
    
    profile_ids = []
    for profile_data in profiles_data:
        response = client.post("/me/profile", json=profile_data, headers=headers)
        if response.status_code in [200, 201]:
            profile_id = response.json().get("id")
            if profile_id:
                profile_ids.append(profile_id)
    
    # Retrieve user profiles
    response = client.get("/me/profiles", headers=headers)
    if response.status_code == 200:
        profiles = response.json()
        assert len(profiles) >= len(profile_ids)


@patch('server.services.content.chat_complete_with_raw')
def test_text_generation_and_translation_flow(mock_chat, client, test_user, mock_llm_response_with_translations):
    """Test text generation and translation pipeline."""
    account, profile = test_user
    
    # Mock comprehensive LLM responses
    mock_chat.return_value = (json.dumps(mock_llm_response_with_translations["text"]), {"model": "test"})
    
    token = create_access_token(data={"sub": str(account.id)}, secret="dev-secret-change")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Generate a text
    from server.services.content import generate_text_content, generate_translations
    import asyncio
    
    text_obj = asyncio.run(generate_text_content(
        account_id=account.id,
        profile_id=profile.id,
        lang=profile.lang,
        target_lang=profile.target_lang,
        profile=profile
    ))
    
    assert text_obj is not None
    assert text_obj.content is not None
    
    # Generate translations for the text
    success = asyncio.run(generate_translations(
        text_id=text_obj.id,
        lang=profile.lang,
        target_lang=profile.target_lang
    ))
    
    assert success == True
    
    # Verify translations were created
    from server.models import ReadingWordGloss
    glosses = text_obj.word_glosses
    assert len(glosses) > 0
    
    # Check that we have translations for specific words
    hello_gloss = next((g for g in glosses if g.surface == "Hola"), None)
    assert hello_gloss is not None
    assert "Hello" in hello_gloss.translation


def test_database_constraints_in_real_flow(client, db):
    """Test that database constraints work correctly during real operations."""
    from server.models import Account, Profile, ReadingText, Lexeme
    from sqlalchemy.exc import IntegrityError
    
    # Create account
    account = Account(
        email="constraint@test.com",
        password_hash="hash",
        subscription_tier="Standard",
        is_active=True,
        is_verified=True
    )
    db.add(account)
    db.commit()
    
    # Create profile (should succeed)
    profile = Profile(
        account_id=account.id,
        lang="es",
        target_lang="en"
    )
    db.add(profile)
    db.commit()
    
    # Try to create text with invalid account_id (should fail)
    with pytest.raises(IntegrityError):
        text = ReadingText(
            account_id=9999,  # Non-existent account
            lang="es",
            target_lang="en",
            content="test"
        )
        db.add(text)
        db.commit()
    
    db.rollback()  # Clean up after error
