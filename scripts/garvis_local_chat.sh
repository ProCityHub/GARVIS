#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE="${GARVIS_LLAMA_CHAT:-$HOME/llama.cpp/build/bin/llama-simple-chat}"
MODEL="${GARVIS_LOCAL_MODEL:-$ROOT/models/Qwen3-4B-Q4_K_M.gguf}"
exec "$ENGINE" -m "$MODEL" -c "${GARVIS_CONTEXT_SIZE:-4096}" -ngl "${GARVIS_GPU_LAYERS:-0}"
