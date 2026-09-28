"""Owner operations that let a worker on another machine synthesize lesson audio.

The PC claims one task at a time over SSH, synthesizes locally, and returns the WAV bytes. A claim
is a lease: a task left running longer than the lease (the PC went off mid-task) is requeued
without charging an attempt. Synthesis failures reported by the worker do use the attempt budget.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from server.clock import as_utc, utc_now
from server.db import session_scope
from server.models import AudioTask
from server.tts import (
    audio_output_paths,
    claim_audio_task,
    complete_audio_task,
    fail_audio_task,
    lesson_for_audio_task,
    provider_request,
    release_audio_task,
    retry_audio_task,
    validate_wave,
)
from server.workspaces import Workspace, registry

DEFAULT_LEASE_MINUTES = 45.0
MAX_AUDIO_BYTES = 200_000_000


def claim_remote_audio(lease_minutes: float = DEFAULT_LEASE_MINUTES) -> dict[str, Any] | None:
    selected = registry.selected_id()
    workspaces = sorted(registry.list(), key=lambda item: item.profile_id != selected)
    for workspace in workspaces:
        with session_scope(workspace) as db:
            cutoff = utc_now() - timedelta(minutes=lease_minutes)
            for stale in db.scalars(select(AudioTask).where(AudioTask.state == "running")):
                if stale.started_at is None or as_utc(stale.started_at) < cutoff:
                    # The PC went away mid-task; that is not the task's fault.
                    release_audio_task(db, stale.id, reason="remote audio lease expired")
            task = claim_audio_task(db)
            if task is None:
                continue
            lesson = lesson_for_audio_task(db, task)
            request = provider_request(task, lesson, Path("remote.wav"))
            return {
                "profile_id": workspace.profile_id,
                "task_id": task.id,
                "request": request.model_dump(mode="json", exclude={"output_path"}),
            }
    return None


def complete_remote_audio(workspace: Workspace, task_id: int, audio: bytes) -> str:
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("remote audio is too large")
    with session_scope(workspace) as db:
        task = db.get(AudioTask, task_id)
        if task is None or task.state != "running":
            raise LookupError(f"running audio task not found: {task_id}")
        temporary, final_path, relative_path = audio_output_paths(workspace, task)
    try:
        temporary.write_bytes(audio)
        temporary.chmod(0o600)
        validate_wave(temporary)
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    with session_scope(workspace) as db:
        complete_audio_task(db, task_id, relative_path=relative_path)
    return relative_path


def fail_remote_audio(workspace: Workspace, task_id: int, error: str) -> str:
    with session_scope(workspace) as db:
        return fail_audio_task(db, task_id, error=f"remote: {error}").state


def release_remote_audio(workspace: Workspace, task_id: int, reason: str) -> str:
    with session_scope(workspace) as db:
        return release_audio_task(db, task_id, reason=f"remote: {reason}").state


def retry_failed_audio(workspace: Workspace, task_id: int) -> str:
    with session_scope(workspace) as db:
        return retry_audio_task(db, task_id).state
