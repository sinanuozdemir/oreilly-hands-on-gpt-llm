#!/bin/bash
# ──────────────────────────────────────────────────────────
# vLLM Inference Server Launch Script
# Model: Qwen/Qwen3-4B-AWQ (INT4 quantized)
# ──────────────────────────────────────────────────────────

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-AWQ}"
PORT="${PORT:-8080}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

echo "🚀 Starting vLLM server"
echo "   Model:    ${MODEL_NAME}"
echo "   Port:     ${PORT}"
echo "   GPU Mem:  ${GPU_MEMORY_UTILIZATION}"
echo "   Max Ctx:  ${MAX_MODEL_LEN}"
echo "   Max Seqs: ${MAX_NUM_SEQS}"

exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --quantization awq \
    --dtype half \
    --trust-remote-code \
    --enable-auto-tool-choice
