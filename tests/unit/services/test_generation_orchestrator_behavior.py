import asyncio
from pathlib import Path

import pytest

from server.services.generation_orchestrator import GenerationOrchestrator
from server.services.text_generation_service import TextGenerationResult


class FakeFileLock:
    def __init__(self, path: Path):
        self._path = path

    def acquire(self, account_id, lang):
        return self._path

    def release(self, _):
        pass


class FakeNotifications:
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
    def __init__(self, text: str = "Hello world.", title: str = "Title"):
        self.called = []
        self.result = TextGenerationResult(
            success=True,
            text=text,
            title=title,
            provider="test",
            model="m",
        )

    def is_generation_in_progress(self, a, l):
        return False

    def mark_generation_started(self, a, l):
        return True

    def mark_generation_completed(self, a, l):
        pass

    def create_placeholder_text(self, db, a, l):
        return 1

    def generate_text_content(self, db, gdb, a, l, tid, job_dir, msgs):
        # record minimal call info without asserting internals
        self.called.append((tid, str(job_dir)))
        # attach messages to mimic real service behavior
        self.result.messages = msgs
        return self.result


class FakeTranslations:
    def __init__(self):
        self.calls = []

    class _R:
        success = True
        words = True
        sentences = True

    def generate_translations(self, adb, gdb, **kw):
        self.calls.append(kw)
        return FakeTranslations._R()


@pytest.mark.asyncio
async def test_orchestrator_emits_events_and_runs_translation(tmp_path: Path):
    orch = GenerationOrchestrator()
    orch.file_lock = FakeFileLock(tmp_path)
    orch.notification_service = FakeNotifications()
    orch.text_gen_service = FakeTextGen()
    orch.translation_service = FakeTranslations()

    # Avoid real filesystem structure
    orch._get_job_dir = lambda a, l: tmp_path / "job"

    # Make translation run inline (avoid thread timing)
    def _inline_start_translation(a, l, tid, text, title, job_dir, msgs):
        # call synchronously and emit event on success
        res = orch.translation_service.generate_translations(
            None, None,
            account_id=a,
            lang=l,
            text_id=tid,
            text_content=text,
            text_title=title,
            job_dir=job_dir,
            reading_messages=msgs,
            provider="test",
            model_id="m",
            base_url="http://test",
        )
        if getattr(res, "success", False):
            orch.notification_service.send_translations_ready(a, l, tid)

    orch._start_translation_job = _inline_start_translation  # type: ignore

    await orch._run_generation_job(None, account_id=1, lang="es")

    event_types = [e[0] for e in orch.notification_service.events]
    assert "content_ready" in event_types
    assert "translations_ready" in event_types
    # Confirm services were invoked
    assert orch.text_gen_service.called
    assert orch.translation_service.calls
