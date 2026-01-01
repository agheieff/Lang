import pytest
import pytest_asyncio
from server.db import SessionLocal
from server.models import (
    Account,
    Profile,
    ReadingText,
    ReadingWordGloss,
    ReadingTextTranslation,
)
from server.services.content import generate_text_content, generate_translations


@pytest.fixture
def test_account():
    """Create a test account."""
    with SessionLocal() as db:
        account = Account(
            email="test@example.com",
            password_hash="hashed_password",
            is_active=True,
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        yield account
        db.delete(account)
        db.commit()


@pytest.fixture
def test_profile(test_account):
    """Create a test profile."""
    with SessionLocal() as db:
        profile = Profile(
            account_id=test_account.id,
            lang="zh-CN",
            target_lang="en",
            level_value=3.0,
            level_var=1.0,
            text_length=200,
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        yield profile
        db.delete(profile)
        db.commit()


@pytest.mark.asyncio
async def test_generate_text_content(test_account, test_profile):
    """Test text generation using generate_text_content function."""
    result = await generate_text_content(
        account_id=test_account.id,
        profile_id=test_profile.id,
        lang="zh-CN",
        target_lang="en",
        profile=test_profile,
    )

    assert result is not None
    assert isinstance(result, ReadingText)
    assert result.id is not None
    assert result.content is not None
    assert len(result.content) > 0
    assert result.lang == "zh-CN"
    assert result.target_lang == "en"


@pytest.mark.asyncio
async def test_generate_word_translations(test_account, test_profile):
    """Test word translations generation for a text."""
    with SessionLocal() as db:
        generated_text = await generate_text_content(
            account_id=test_account.id,
            profile_id=test_profile.id,
            lang="zh-CN",
            target_lang="en",
            profile=test_profile,
        )

        assert generated_text is not None
        text_id = generated_text.id

        success = await generate_translations(
            text_id=text_id,
            lang="zh-CN",
            target_lang="en",
        )

        assert success is True

        word_glosses = (
            db.query(ReadingWordGloss).filter(ReadingWordGloss.text_id == text_id).all()
        )

        assert len(word_glosses) > 0
        for gloss in word_glosses:
            assert gloss.text_id == text_id
            assert gloss.lang == "zh-CN"
            assert gloss.target_lang == "en"
            assert gloss.surface is not None
            assert len(gloss.surface) > 0
            assert gloss.translation is not None
            assert len(gloss.translation) > 0
            assert gloss.pos is not None
            assert gloss.lemma is not None


@pytest.mark.asyncio
async def test_generate_sentence_translations(test_account, test_profile):
    """Test sentence translations generation for a text."""
    with SessionLocal() as db:
        generated_text = await generate_text_content(
            account_id=test_account.id,
            profile_id=test_profile.id,
            lang="zh-CN",
            target_lang="en",
            profile=test_profile,
        )

        assert generated_text is not None
        text_id = generated_text.id

        success = await generate_translations(
            text_id=text_id,
            lang="zh",
            target_lang="en",
        )

        assert success is True

        sentence_translations = (
            db.query(ReadingTextTranslation)
            .filter(ReadingTextTranslation.text_id == text_id)
            .all()
        )

        assert len(sentence_translations) > 0
        for trans in sentence_translations:
            assert trans.text_id == text_id
            assert trans.target_lang == "en"
            assert trans.unit == "sentence"
            assert trans.source_text is not None
            assert len(trans.source_text) > 0
            assert trans.translated_text is not None
            assert len(trans.translated_text) > 0
            assert trans.segment_index is not None
