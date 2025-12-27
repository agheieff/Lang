"""End-to-end integration test for the reading system."""

import pytest
from unittest.mock import patch
from server.auth import create_access_token


@pytest.mark.integration
def test_end_to_end_reading_workflow(client, db, test_user, mock_llm_response):
    """Test complete reading workflow: registration → profile → reading."""
    account, profile = test_user

    # Mock LLM responses
    with patch("server.services.content.chat_complete_with_raw") as mock_llm:
        # Mock CSV responses
        mock_llm.return_value = (
            "title|text\nUn paseo en el parque|Hoy hace buen tiempo. Vamos juntos a pasear por el parque.",
            "word|translation|pos|lemma|pinyin\nHola|Hello|INTJ|hola|\nparque|park|NOUN|parque|",
            "source|translation\nHoy hace buen tiempo.|The weather is good today.\nVamos juntos a pasear por el parque.|Let's go for a walk in the park together.",
        )

        # 1. Generate text (simulated by background worker)
        from server.services.content import generate_text_content
        import asyncio

        text_obj = asyncio.run(
            generate_text_content(
                account_id=account.id,
                profile_id=profile.id,
                lang=profile.lang,
                target_lang=profile.target_lang,
                profile=profile,
            )
        )

        assert text_obj is not None
        assert text_obj.lang == "es"
        assert text_obj.title == "Un paseo en el parque"

        # 2. Mark text as ready (simulating completion)
        text_obj.words_complete = True
        text_obj.sentences_complete = True
        db.commit()

        # 3. Request reading page
        token = create_access_token(
            subject=str(account.id), secret_key="dev-secret-change"
        )
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get("/reading", headers=headers)
        assert response.status_code == 200

        # 4. Verify response contains text
        html = response.text
        assert "Un paseo en el parque" in html or text_obj.content in html

        # 5. Test word glosses are available
        gloss_response = client.get(
            f"/reading/{text_obj.id}/translations", headers=headers
        )
        assert gloss_response.status_code == 200
        gloss_data = gloss_response.json()
        assert "items" in gloss_data


@pytest.mark.integration
def test_recommendation_engine_integration(client, db, test_user):
    """Test recommendation engine with real texts."""
    account, profile = test_user

    # Create multiple texts with different properties
    from server.models import ReadingText
    from datetime import datetime, timezone

    texts = [
        ReadingText(
            generated_for_account_id=account.id,
            lang=profile.lang,
            target_lang=profile.target_lang,
            title="Easy Text",
            content="Texto fácil para aprender.",
            words_complete=True,
            sentences_complete=True,
            difficulty_estimate=2.0,
            topic="daily_life",
            word_count=50,
            created_at=datetime.now(timezone.utc),
        ),
        ReadingText(
            generated_for_account_id=account.id,
            lang=profile.lang,
            target_lang=profile.target_lang,
            title="Medium Text",
            content="Texto de nivel medio con más palabras y desafíos.",
            words_complete=True,
            sentences_complete=True,
            difficulty_estimate=4.0,
            topic="culture",
            word_count=100,
            created_at=datetime.now(timezone.utc),
        ),
    ]

    for text in texts:
        db.add(text)
    db.commit()

    # Test recommendation
    from server.services.recommendation import select_best_text

    best_text = select_best_text(db, profile)
    assert best_text is not None

    # Since profile level is 3.0, medium text (4.0) should be closer
    # Both are within tolerance, so check that it picks one of them
    assert best_text.id in [t.id for t in texts]

    # Request reading page
    token = create_access_token(subject=str(account.id), secret_key="dev-secret-change")
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/reading", headers=headers)
    assert response.status_code == 200

    html = response.text
    # Should contain one of our texts
    assert "Easy Text" in html or "Medium Text" in html
