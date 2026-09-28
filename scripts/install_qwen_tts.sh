#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
command -v rocminfo >/dev/null || {
  echo "This installer currently targets AMD ROCm; browser speech remains available." >&2
  exit 1
}
ROC_INFO="$(rocminfo 2>/dev/null)"
[[ "$ROC_INFO" =~ Name:[[:space:]]+gfx ]] || {
  echo "No ROCm GPU was found; refusing to install slow Qwen CPU inference." >&2
  exit 1
}

umask 077
RUNTIME="data/tts/qwen"
PYTHON="$RUNTIME/.venv/bin/python"
mkdir -p "$RUNTIME"
uv venv --python 3.12 "$RUNTIME/.venv"

TORCH_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl"
TORCHAUDIO_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl"
TRITON_URL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl"

uv pip install --python "$PYTHON" "numpy==1.26.4" "$TORCH_URL" "$TORCHAUDIO_URL" "$TRITON_URL"
uv pip install --python "$PYTHON" "numpy==1.26.4" "qwen-tts==0.1.1"

rm -f "$RUNTIME/.ready"
"$PYTHON" -m server.qwen_tts_provider --warmup
touch "$RUNTIME/.ready"
chmod 600 "$RUNTIME/.ready"

echo "Qwen3-TTS is installed and its pinned model is ready. Restart pnpm dev to start audio work."
