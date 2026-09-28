from __future__ import annotations

import io
import wave
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from server.clock import utc_now
from server.db import init_db, session_scope
from server.learning import import_lesson
from server.models import AudioTask
from server.profile_activation import ProfileActivationUpdate, activate_profile_settings
from server.remote_tts import claim_remote_audio, complete_remote_audio, fail_remote_audio
from server.tts import audio_status
from server.workspaces import Workspace, WorkspaceRegistry


def _wave_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x01" * 2_400)
    return buffer.getvalue()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lesson_factory: Any) -> Workspace:
    registry = WorkspaceRegistry(tmp_path)
    monkeypatch.setattr("server.remote_tts.registry", registry)
    resolved = registry.resolve("es-es")
    init_db(resolved)
    with session_scope(resolved) as db:
        activate_profile_settings(db, ProfileActivationUpdate())
        import_lesson(db, lesson_factory(key="remote-one"))
    return resolved


def _task(workspace: Workspace) -> AudioTask:
    with session_scope(workspace) as db:
        task = db.scalar(select(AudioTask))
        assert task is not None
        db.expunge(task)
        return task


def test_remote_worker_claims_uploads_and_completes_audio(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    assert claim["profile_id"] == "es-es"
    assert claim["request"]["chunks"] and "output_path" not in claim["request"]
    assert claim_remote_audio() is None  # one running task, nothing else pending

    relative = complete_remote_audio(workspace, claim["task_id"], _wave_bytes())

    task = _task(workspace)
    assert task.state == "completed" and task.relative_path == relative
    assert (workspace.directory / relative).is_file()


def test_invalid_audio_is_rejected_and_reported_failures_use_attempts(
    workspace: Workspace,
) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    with pytest.raises(ValueError):
        complete_remote_audio(workspace, claim["task_id"], b"not audio")
    assert _task(workspace).state == "running"

    assert fail_remote_audio(workspace, claim["task_id"], "synthesis crashed") == "pending"
    assert _task(workspace).attempts == 1


def test_expired_lease_requeues_without_charging_an_attempt(workspace: Workspace) -> None:
    claim = claim_remote_audio()
    assert claim is not None
    with session_scope(workspace) as db:
        task = db.get(AudioTask, claim["task_id"])
        assert task is not None
        task.started_at = utc_now() - timedelta(hours=2)
        db.commit()

    again = claim_remote_audio(lease_minutes=45)

    assert again is not None and again["task_id"] == claim["task_id"]
    assert _task(workspace).attempts == 1


def test_remote_mode_reports_queued_instead_of_unavailable(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("server.tts.provider_command", lambda: None)
    with session_scope(workspace) as db:
        lesson_id = db.scalar(select(AudioTask.lesson_id))
        assert lesson_id is not None
        assert audio_status(db, workspace, lesson_id).state == "unavailable"
        monkeypatch.setenv("ARC_LANG_TTS_REMOTE", "1")
        status = audio_status(db, workspace, lesson_id)
    assert status.reason == "queued"
    assert "PC" in (status.message or "")
