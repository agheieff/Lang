"""PC-side audio worker: claims lesson audio on the Pi over SSH and synthesizes it locally.

The Pi owns the queue and the finished files; this process only needs SSH access and the local
Qwen runtime. It runs whenever the PC is on and backs off quietly while the Pi is unreachable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from types import FrameType
from typing import Any

from server.tts import AudioProviderRequest, provider_command, validate_wave
from server.tts_worker import ProviderProcess

DEFAULT_HOST = "rpi"
DEFAULT_REMOTE_CLI = (
    "cd ~/data/lang/app && ARC_LANG_DATA_DIR=~/data/lang/state .venv/bin/python -m server.cli"
)
IDLE_SECONDS = 120.0
UNREACHABLE_SECONDS = 300.0
RUNTIME_BROKEN_SECONDS = 1800.0
# Failures of this machine's runtime rather than of the task: release without using an attempt.
ENVIRONMENT_ERROR_MARKERS = (
    "LocalEntryNotFoundError",
    "GPU runtime is unavailable",
    "No module named",
    "Qwen provider stopped unexpectedly",
)
SSH_TIMEOUT_SECONDS = 120.0


class RemoteUnavailable(RuntimeError):
    pass


class RuntimeBroken(RuntimeError):
    pass


class Remote:
    def __init__(self, host: str, cli: str) -> None:
        self.host = host
        self.cli = cli

    def run(self, arguments: list[str], *, stdin: bytes | None = None) -> Any:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            self.host,
            f"{self.cli} {shlex.join(arguments)}",
        ]
        try:
            completed = subprocess.run(
                command, input=stdin, capture_output=True, timeout=SSH_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired as error:
            raise RemoteUnavailable("ssh timed out") from error
        if completed.returncode == 255:
            raise RemoteUnavailable(completed.stderr.decode(errors="replace").strip())
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode(errors="replace").strip())
        return json.loads(completed.stdout)


class Worker:
    def __init__(self, remote: Remote, provider: ProviderProcess) -> None:
        self.remote = remote
        self.provider = provider
        self.stopping = False

    def stop(self, _signal: int, _frame: FrameType | None) -> None:
        self.stopping = True

    def step(self) -> bool:
        claim = self.remote.run(["tts", "claim"])
        if claim is None:
            return False
        profile, task_id = claim["profile_id"], int(claim["task_id"])
        task = ["--profile", profile, "tts"]
        with tempfile.TemporaryDirectory(prefix="lang-tts-") as directory:
            output = Path(directory) / "audio.tmp.wav"  # the provider requires this suffix
            try:
                request = AudioProviderRequest(**claim["request"], output_path=str(output))
                self.provider.generate(request)
                validate_wave(output)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if any(marker in message for marker in ENVIRONMENT_ERROR_MARKERS):
                    self.remote.run([*task, "release", "--task", str(task_id), "--reason", message])
                    raise RuntimeBroken(message) from error
                self.remote.run([*task, "fail", "--task", str(task_id), "--error", message])
                print(f"audio {profile}/{task_id} failed: {message}", flush=True)
                return True
            self.remote.run([*task, "complete", "--task", str(task_id)], stdin=output.read_bytes())
        print(f"audio {profile}/{task_id} completed", flush=True)
        return True

    def run(self) -> None:
        while not self.stopping:
            try:
                worked = self.step()
            except RemoteUnavailable as error:
                print(f"Pi unreachable, retrying later: {error}", flush=True)
                self._sleep(UNREACHABLE_SECONDS)
                continue
            except RuntimeBroken as error:
                print(f"local audio runtime is broken, retrying later: {error}", flush=True)
                self.provider.close()
                self._sleep(RUNTIME_BROKEN_SECONDS)
                continue
            if not worked:
                self._sleep(IDLE_SECONDS)

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stopping and time.monotonic() < deadline:
            time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize Pi lesson audio on this machine")
    parser.add_argument("--host", default=os.getenv("ARC_LANG_REMOTE_HOST", DEFAULT_HOST))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    command = provider_command()
    if command is None:
        raise SystemExit("Local Qwen runtime is not installed (scripts/install_qwen_tts.sh).")
    provider = ProviderProcess(command, timeout_seconds=3600.0)
    worker = Worker(
        Remote(args.host, os.getenv("ARC_LANG_REMOTE_CLI", DEFAULT_REMOTE_CLI)), provider
    )
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    try:
        if args.once:
            try:
                worker.step()
            except RuntimeBroken as error:
                raise SystemExit(f"local audio runtime is broken: {error}") from error
        else:
            worker.run()
    finally:
        provider.close()


if __name__ == "__main__":
    main()
