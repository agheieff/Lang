"""Unit tests for Generation Orchestrator."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from sqlalchemy.orm import Session

from server.services.generation_orchestrator import GenerationOrchestrator
from server.services.text_generation_service import TextGenerationResult


class TestGenerationOrchestrator:
    """Test cases for GenerationOrchestrator methods."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a GenerationOrchestrator instance for testing."""
        return GenerationOrchestrator()
    
    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return MagicMock(spec=Session)
    
    @pytest.fixture
    def mock_global_db(self):
        """Mock global database session."""
        return MagicMock(spec=Session)
    
    @pytest.fixture
    def sample_text_generation_result(self):
        """Sample successful TextGenerationResult."""
        result = TextGenerationResult(
            success=True,
            text="Test generated text content",
            title="Test Title",
            provider="openrouter",
            model="test-model"
        )
        return result
    
    @pytest.fixture
    def sample_failed_text_generation_result(self):
        """Sample failed TextGenerationResult."""
        return TextGenerationResult(
            success=False,
            text="",
            error="Generation failed due to network error"
        )

    @pytest.mark.asyncio
    async def test_generate_main_text_content_success(
        self, orchestrator, mock_db, sample_text_generation_result
    ):
        """Test successful text content generation."""
        account_id = 123
        lang = "es"
        text_id = 456
        job_dir = Path("/tmp/test_job_dir")
        
        # Mock dependencies
        with patch('server.services.generation_orchestrator.db_manager') as mock_db_manager, \
             patch('server.services.generation_orchestrator.GlobalSessionLocal') as mock_global_session_local, \
             patch('server.services.generation_orchestrator.build_reading_prompt_spec') as mock_build_spec, \
             patch('server.llm.build_reading_prompt') as mock_build_prompt:  # Note: imported inside function
            
            # Setup mocks
            mock_db_manager.transaction.return_value.__enter__.return_value = mock_db
            mock_global_session = MagicMock()
            mock_global_session_local.return_value = mock_global_session
            mock_build_spec.return_value = ("spec", ["word1", "word2"], "intermediate")
            mock_build_prompt.return_value = [{"role": "system", "content": "test"}]
            
            # Mock text_gen_service.generate_text_content (NOT async)
            orchestrator.text_gen_service.generate_text_content = MagicMock(
                return_value=sample_text_generation_result
            )
            
            # Execute the method
            result = await orchestrator._generate_main_text_content(
                account_id, lang, text_id, job_dir
            )
            
            # Verify the result
            assert result is sample_text_generation_result
            assert result.success is True
            assert result.text == "Test generated text content"
            assert result.title == "Test Title"
            assert hasattr(result, 'messages')
            assert result.messages == [{"role": "system", "content": "test"}]
            
            # Verify method calls
            mock_db_manager.transaction.assert_called_once_with(account_id)
            mock_global_session.close.assert_called_once()
            mock_build_spec.assert_called_once_with(mock_db, account_id=account_id, lang=lang)
            mock_build_prompt.assert_called_once()
            orchestrator.text_gen_service.generate_text_content.assert_called_once_with(
                mock_db, mock_global_session, account_id, lang, text_id, job_dir, [{"role": "system", "content": "test"}]
            )

    @pytest.mark.asyncio
    async def test_generate_main_text_content_failure(
        self, orchestrator, mock_db, sample_failed_text_generation_result
    ):
        """Test text content generation failure."""
        account_id = 123
        lang = "es"
        text_id = 456
        job_dir = Path("/tmp/test_job_dir")
        
        # Mock dependencies
        with patch('server.services.generation_orchestrator.db_manager') as mock_db_manager, \
             patch('server.services.generation_orchestrator.GlobalSessionLocal') as mock_global_session_local, \
             patch('server.services.generation_orchestrator.build_reading_prompt_spec') as mock_build_spec, \
             patch('server.llm.build_reading_prompt') as mock_build_prompt:
            
            # Setup mocks
            mock_db_manager.transaction.return_value.__enter__.return_value = mock_db
            mock_global_session = MagicMock()
            mock_global_session_local.return_value = mock_global_session
            mock_build_spec.return_value = ("spec", ["word1", "word2"], "intermediate")
            mock_build_prompt.return_value = [{"role": "system", "content": "test"}]
            
            # Mock text_gen_service.generate_text_content with failure (NOT async)
            orchestrator.text_gen_service.generate_text_content = MagicMock(
                return_value=sample_failed_text_generation_result
            )
            
            # Execute the method
            result = await orchestrator._generate_main_text_content(
                account_id, lang, text_id, job_dir
            )
            
            # Verify the result
            assert result is sample_failed_text_generation_result
            assert result.success is False
            assert result.error == "Generation failed due to network error"
            
            # Verify global DB is still closed even on failure
            mock_global_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_main_text_content_handles_exceptions(
        self, orchestrator, mock_db
    ):
        """Test that exceptions during text generation are properly handled."""
        account_id = 123
        lang = "es"
        text_id = 456
        job_dir = Path("/tmp/test_job_dir")
        
        # Mock dependencies
        with patch('server.services.generation_orchestrator.db_manager') as mock_db_manager, \
             patch('server.services.generation_orchestrator.GlobalSessionLocal') as mock_global_session_local, \
             patch('server.services.generation_orchestrator.build_reading_prompt_spec') as mock_build_spec, \
             patch('server.llm.build_reading_prompt') as mock_build_prompt:
            
            # Setup mocks
            mock_db_manager.transaction.return_value.__enter__.return_value = mock_db
            mock_global_session = MagicMock()
            mock_global_session_local.return_value = mock_global_session
            mock_build_spec.return_value = ("spec", ["word1", "word2"], "intermediate")
            mock_build_prompt.return_value = [{"role": "system", "content": "test"}]
            
            # Mock text_gen_service.generate_text_content to raise exception (NOT async)
            orchestrator.text_gen_service.generate_text_content = MagicMock(
                side_effect=Exception("Database connection failed")
            )
            
            # Execute and verify exception is raised
            with pytest.raises(Exception, match="Database connection failed"):
                await orchestrator._generate_main_text_content(
                    account_id, lang, text_id, job_dir
                )
            
            # Verify global DB is still closed even when exception occurs
            mock_global_session.close.assert_called_once()
