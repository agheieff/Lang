"""Single-process durable worker for optional local lesson audio."""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import time
import wave
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from server.db import init_all_databases, session_scope
from server.tts import (
    AudioProviderRequest,
    audio_output_paths,
    claim_audio_task,
    complete_audio_task,
    ensure_audio_backfill,
    fail_audio_task,
    lesson_for_audio_task,
    project_environment,
    provider_command,
    provider_request,
    recover_audio_tasks,
)
from server.workspaces import PROJECT_ROOT, Workspace, registry

POLL_SECONDS = 2.0
DEFAULT_TIMEOUT_SECONDS = 3600.0


class AudioProvider(Protocol):
    def generate(self, request: AudioProviderRequest) -> dict[str, Any]: ...


class ProviderProcess:
    def __init__(self, command: Sequence[str], *, timeout_seconds: float) -> None:
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None

    def generate(self, request: AudioProviderRequest) -> dict[str, Any]:
        process = self._running_process()
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(request.model_dump_json() + "\n")
        process.stdin.flush()
        ready, _, _ = select.select([process.stdout], [], [], self.timeout_seconds)
        if not ready:
            self.close()
            raise TimeoutError("Qwen audio generation timed out")
        line = process.stdout.readline()
        if not line:
            return_code = process.poll()
            self.close()
            raise RuntimeError(f"Qwen provider stopped unexpectedly ({return_code})")
        try:
            result: Any = json.loads(line)
        except json.JSONDecodeError as error:
            self.close()
            raise RuntimeError("Qwen provider returned an invalid response") from error
        if not isinstance(result, dict) or result.get("request_id") != request.request_id:
            self.close()
            raise RuntimeError("Qwen provider returned a mismatched response")
        if result.get("ok") is not True:
            message = result.get("error")
            self.close()
            raise RuntimeError(str(message) if message else "Qwen provider failed")
        return result

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                process.kill()
            process.wait()

    def _running_process(self) -> subprocess.Popen[str]:
        if self.process is not None and self.process.poll() is None:
            return self.process
        self.close()
        self.process = subprocess.Popen(
            self.command,
            cwd=PROJECT_ROOT,
            env=project_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self.process


def process_audio_task(workspace: Workspace, task_id: int, provider: AudioProvider) -> Path:
    temporary: Path | None = None
    try:
        with session_scope(workspace) as db:
            task = db.get(_audio_task_model(), task_id)
            if task is None or task.state != "running":
                raise LookupError(f"running audio task not found: {task_id}")
            lesson = lesson_for_audio_task(db, task)
            temporary, final_path, relative_path = audio_output_paths(workspace, task)
            request = provider_request(task, lesson, temporary)
        temporary.unlink(missing_ok=True)
        provider.generate(request)
        _validate_wave(temporary)
        os.replace(temporary, final_path)
        final_path.chmod(0o600)
        with session_scope(workspace) as db:
            complete_audio_task(db, task_id, relative_path=relative_path)
        return final_path
    except Exception as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        with session_scope(workspace) as db, suppress(LookupError, ValueError):
            fail_audio_task(db, task_id, error=f"{type(error).__name__}: {error}")
        raise


def _audio_task_model() -> type[Any]:
    # Local import keeps this worker's public protocol easy to fake in tests.
    from server.models import AudioTask

    return AudioTask


def _validate_wave(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 44:
        raise ValueError("Qwen provider did not create usable audio")
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getnframes() <= 0 or audio.getframerate() < 8_000:
                raise ValueError("Qwen provider created an empty or invalid WAV")
            if audio.getnchannels() not in {1, 2} or audio.getsampwidth() not in {2, 3, 4}:
                raise ValueError("Qwen provider created an unsupported WAV format")
    except wave.Error as error:
        raise ValueError("Qwen provider created an invalid WAV") from error


class Worker:
    def __init__(self, provider: ProviderProcess) -> None:
        self.provider = provider
        self.stopping = False

    def stop(self, _signal: int, _frame: FrameType | None) -> None:
        self.stopping = True

    def run(self, *, once: bool = False) -> None:
        init_all_databases()
        for workspace in registry.list():
            with session_scope(workspace) as db:
                recover_audio_tasks(db)
                ensure_audio_backfill(db)
        while not self.stopping:
            worked = self._step()
            if once:
                return
            if not worked:
                time.sleep(POLL_SECONDS)

    def _step(self) -> bool:
        selected = registry.selected_id()
        workspaces = sorted(registry.list(), key=lambda workspace: workspace.profile_id != selected)
        for workspace in workspaces:
            with session_scope(workspace) as db:
                task = claim_audio_task(db)
            if task is None:
                continue
            try:
                process_audio_task(workspace, task.id, self.provider)
            except Exception as error:
                print(f"Audio task {workspace.profile_id}/{task.id} failed: {error}", flush=True)
            return True
        return False


def _timeout_seconds() -> float:
    raw = os.getenv("ARC_LANG_TTS_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    value = float(raw)
    if not 60 <= value <= 7_200:
        raise ValueError("ARC_LANG_TTS_TIMEOUT_SECONDS must be between 60 and 7200")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare cached local audio for Arcadia lessons")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    command = provider_command()
    if command is None:
        print("Local Qwen runtime is not installed; audio worker is idle.", flush=True)
        return
    provider = ProviderProcess(command, timeout_seconds=_timeout_seconds())
    worker = Worker(provider)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    try:
        worker.run(once=args.once)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
