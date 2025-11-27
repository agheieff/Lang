"""
Tests for GenerationOrchestrator behavior.

These tests verify observable behavior (events emitted, services called)
rather than implementation details.
"""
from pathlib import Path

import pytest

from server.services.generation_orchestrator import GenerationOrchestrator
from server.services.text_generation_service import TextGenerationResult


# Fake implementations for isolation
class FakeNotifications:
    """Records notification events for verification."""
    def __init__(self):
        self.events = []

    def send_content_ready(self, a, l, t):
        self.events.append(("content_ready", t))

    def send_translations_ready(self, a, l, t):
        self.events.append(("translations_ready", t))

    def send_generation_started(self, a, l, t):
        self.events.append(("generation_started", t))

    def send_generation_failed(self, a, l, t, err):
        self.events.append(("generation_failed", t, err))


class FakeTextGen:
    """Simulates text generation service."""
    def __init__(self):
        self.generated = False
        self.result = TextGenerationResult(
            success=True,
            text="Generated test content.",
            title="Test Title",
            provider="test",
            model="test-model",
        )

    def is_generation_in_progress(self, a, l):
        return False

    def create_placeholder_text(self, db, a, l, ci_target=None, topic=None, pooled=False):
        return 1

    def generate_text_content(self, db, gdb, a, l, tid, job_dir, msgs):
        self.generated = True
        self.result.messages = msgs
        return self.result


class FakeTranslations:
    """Simulates translation service."""
    def __init__(self):
        self.translated = False

    class _Result:
        success = True

    def generate_translations(self, adb, gdb, **kw):
        self.translated = True
        return self._Result()


@pytest.mark.asyncio
async def test_generation_emits_content_ready_event(tmp_path: Path, monkeypatch):
    """Test that successful generation emits content_ready notification."""
    orch = GenerationOrchestrator()
    orch.notification_service = FakeNotifications()
    orch.text_gen_service = FakeTextGen()
    orch.translation_service = FakeTranslations()
    orch._get_job_dir = lambda a, l: tmp_path / "job"
    
    # Mock file lock
    class FakeLock:
        def acquire(self, a, l): return tmp_path
        def release(self, p): pass
    orch.file_lock = FakeLock()
    
    # Mock external dependencies
    _setup_mocks(monkeypatch)
    
    # Make translation synchronous for testing
    def inline_translation(a, l, tid, text, title, job_dir, msgs):
        res = orch.translation_service.generate_translations(None, None, text_id=tid)
        if res.success:
            orch.notification_service.send_translations_ready(a, l, tid)
    orch._start_translation_job = inline_translation

    await orch._run_generation_job(None, account_id=1, lang="es")

    # Verify behavior: events were emitted
    event_types = [e[0] for e in orch.notification_service.events]
    assert "content_ready" in event_types, "Should emit content_ready on success"
    assert "translations_ready" in event_types, "Should emit translations_ready after translation"
    
    # Verify behavior: services were called
    assert orch.text_gen_service.generated, "Text generation should have been called"
    assert orch.translation_service.translated, "Translation should have been called"


@pytest.mark.asyncio
async def test_generation_failure_emits_failed_event(tmp_path: Path, monkeypatch):
    """Test that generation failure emits generation_failed notification."""
    orch = GenerationOrchestrator()
    orch.notification_service = FakeNotifications()
    orch.translation_service = FakeTranslations()
    orch._get_job_dir = lambda a, l: tmp_path / "job"
    
    # Text gen that fails
    class FailingTextGen(FakeTextGen):
        def generate_text_content(self, *args, **kwargs):
            self.generated = True
            return TextGenerationResult(success=False, text="", error="Test error")
    
    orch.text_gen_service = FailingTextGen()
    
    class FakeLock:
        def acquire(self, a, l): return tmp_path
        def release(self, p): pass
    orch.file_lock = FakeLock()
    
    _setup_mocks(monkeypatch)

    await orch._run_generation_job(None, account_id=1, lang="es")

    event_types = [e[0] for e in orch.notification_service.events]
    assert "generation_failed" in event_types, "Should emit generation_failed on error"
    assert "content_ready" not in event_types, "Should not emit content_ready on failure"


def _setup_mocks(monkeypatch):
    """Setup common mocks for orchestrator tests."""
    from contextlib import contextmanager
    from server.llm.prompts import PromptSpec
    
    # Mock pool service
    class FakePoolService:
        def get_generation_params(self, profile, vary=True):
            return 0.92, "fiction"
    monkeypatch.setattr(
        "server.services.generation_orchestrator.get_pool_selection_service",
        lambda: FakePoolService()
    )
    
    # Mock DB operations
    class FakeProfile:
        ci_preference = 0.92
        topic_weights = {}
    
    class FakeQuery:
        def filter(self, *args, **kwargs): return self
        def first(self): return FakeProfile()
    
    class FakeDB:
        def query(self, model): return FakeQuery()
        def add(self, obj): pass
        def flush(self): pass
        def commit(self): pass
    
    @contextmanager
    def fake_transaction(account_id):
        yield FakeDB()
    monkeypatch.setattr("server.services.generation_orchestrator.db_manager.transaction", fake_transaction)
    
    # Mock prompt building
    def fake_build_spec(db, *, account_id, lang, **kwargs):
        spec = PromptSpec(
            lang=lang, unit="sentences", approx_len=150,
            user_level_hint="intermediate", include_words=[]
        )
        return spec, [], "intermediate"
    monkeypatch.setattr("server.services.generation_orchestrator.build_reading_prompt_spec", fake_build_spec)
    
    # Mock global session
    class FakeGlobalSession:
        def close(self): pass
    monkeypatch.setattr("server.services.generation_orchestrator.GlobalSessionLocal", lambda: FakeGlobalSession())
