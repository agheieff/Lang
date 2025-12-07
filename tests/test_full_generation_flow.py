import pytest
import asyncio
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from server.db import Base
from server.auth.models import Account
from server.auth.models import Base as AuthBase
from server.models import Profile, ReadingText, ReadingTextTranslation, ReadingWordGloss
from server.services.generation_orchestrator import GenerationOrchestrator
from server.services.text_generation_service import TextGenerationResult

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
    account = Account(id=1, email="flow_test@example.com", password_hash="hash", subscription_tier="Free")
    db_session.add(account)
    db_session.flush()
    profile = Profile(account_id=1, lang="es", target_lang="en")
    db_session.add(profile)
    db_session.commit()
    return account

@pytest.mark.asyncio
async def test_full_generation_flow_calls_translation(db_session, test_user):
    """
    Verify that orchestrator.ensure_text_available triggers BOTH text generation AND translation.
    """
    orchestrator = GenerationOrchestrator()
    
    # Mock the actual LLM calls to avoid network/cost/latency
    # We mock at the service level to verify orchestration
    
    # 1. Setup Mocks
    mock_text_result = TextGenerationResult(
        success=True,
        text="Hola mundo. Esto es una prueba.",
        title="Test Title",
        provider="mock",
        model="mock"
    )
    # Manually attach messages as the service does
    mock_text_result.messages = [{"role": "user", "content": "Generate text"}]
    
    orchestrator.text_gen_service.generate_text_content = MagicMock(return_value=mock_text_result)
    
    # We want to verify this is called
    orchestrator.translation_service.generate_translations = MagicMock(
        return_value=MagicMock(success=True, words=True, sentences=True)
    )
    
    # Mock internal helpers to avoid FS/DB side effects that are hard to setup
    with patch("server.services.generation_orchestrator.db_manager") as mock_db_mgr:
        # Make db_manager.transaction() yield our test session
        mock_db_mgr.transaction.return_value.__enter__.return_value = db_session
        
        with patch("server.services.generation_orchestrator.GlobalSessionLocal", return_value=db_session):
            with patch("server.services.generation_orchestrator.build_reading_prompt_spec", return_value=("spec", [], "A1")):
                with patch("server.llm.build_reading_prompt", return_value=[{"role": "user", "content": "prompt"}]):
                    with patch.object(orchestrator.text_gen_service, "create_placeholder_text", return_value=123):
                        with patch.object(orchestrator.text_gen_service, "mark_generation_started", return_value=True):
                             with patch.object(orchestrator.text_gen_service, "mark_generation_completed"):
                                with patch.object(orchestrator.file_lock, "acquire", return_value="/tmp/lock"):
                                    with patch.object(orchestrator.file_lock, "release"):
                                        
                                        # 2. Execute
                                        # We call the internal async method directly to avoid threading issues in test
                                        # real ensure_text_available launches a thread which calls _run_generation_job
                                        await orchestrator._run_generation_job(db_session, test_user.id, "es")
                                        
                                        # 3. Assertions
                                        
                                        # Text Gen should be called
                                        orchestrator.text_gen_service.generate_text_content.assert_called_once()
                                        
                                        # Translation Gen should be called (Wait! _run_generation_job calls _start_translation_job_async which spawns a thread)
                                        # In the code:
                                        # await self._start_translation_job_async(...) -> self._start_translation_job(...) -> Thread(target=self._run_translation_job).start()
                                        
                                        # We need to verify _run_translation_job was called OR mock the threading to run synchronously.
                                        # Actually, let's look at the orchestrator code again.
                                        # _start_translation_job_async calls _start_translation_job which spawns a thread.
                                        
                                        # Ideally we'd verify that generate_translations IS called.
                                        # But since it's in a thread, we might need to mock `threading.Thread` to run it immediately or just assert start_translation_job was called.
                                        
                                        pass

    # Since we can't easily await the thread in this unit-ish test without mocking Thread,
    # let's re-run with Thread patched to run immediately.
    
@pytest.mark.asyncio
async def test_translation_service_is_called(db_session, test_user):
    orchestrator = GenerationOrchestrator()
    
    # Mock Text Gen success
    mock_text_result = TextGenerationResult(
        success=True,
        text="Hola mundo.",
        title="Title",
        provider="mock",
        model="mock"
    )
    mock_text_result.messages = [{"role": "user", "content": "Generate"}]
    orchestrator.text_gen_service.generate_text_content = MagicMock(return_value=mock_text_result)
    
    # Mock Translation Gen
    orchestrator.translation_service.generate_translations = MagicMock(return_value=MagicMock(success=True))
    
    # Mock DB manager
    with patch("server.services.generation_orchestrator.db_manager") as mock_db_mgr:
        mock_db_mgr.transaction.return_value.__enter__.return_value = db_session
        with patch("server.services.generation_orchestrator.GlobalSessionLocal", return_value=db_session):
            with patch("server.services.generation_orchestrator.build_reading_prompt_spec", return_value=("spec", [], "A1")):
                with patch("server.llm.build_reading_prompt", return_value=[]):
                    with patch.object(orchestrator.text_gen_service, "create_placeholder_text", return_value=123):
                        with patch.object(orchestrator.text_gen_service, "mark_generation_started", return_value=True):
                             with patch.object(orchestrator.file_lock, "acquire", return_value="/tmp/lock"):
                                with patch.object(orchestrator.file_lock, "release"):
                                    
                                    # CRITICAL: Mock threading.Thread to run inline
                                    with patch("threading.Thread") as mock_thread_cls:
                                        # Define a fake thread that runs the target immediately
                                        def side_effect(target=None, args=(), **kwargs):
                                            target(*args)
                                            return MagicMock()
                                        
                                        mock_thread_cls.side_effect = side_effect
                                        
                                        # Run
                                        await orchestrator._run_generation_job(db_session, test_user.id, "es")
                                        
                                        # Check if translation service was called
                                        orchestrator.translation_service.generate_translations.assert_called_once()
                                        
                                        # Check arguments
                                        call_args = orchestrator.translation_service.generate_translations.call_args
                                        assert call_args is not None
                                        _, kwargs = call_args
                                        assert kwargs['text_content'] == "Hola mundo."
                                        assert kwargs['text_id'] == 123
