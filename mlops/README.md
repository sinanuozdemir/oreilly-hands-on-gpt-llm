# LLM Deploy Stack

Two deployment patterns for serving open-source / fine-tuned LLMs.

## Option 1: Railway (CPU) — `railway/`

**For:** Prototyping, low-traffic APIs, demos, CPU-only environments  
**Model:** Qwen3-4B GGUF (Q4_K_M quantized, ~2.5GB RAM)  
**Stack:** FastAPI + llama-cpp-python  
**Deploy:** Railway (no GPU needed)

```
┌──────────────────────────────────────────┐
│              Railway Service              │
│                                          │
│   FastAPI + llama-cpp-python             │
│   ┌────────────────────────────────┐     │
│   │  /v1/chat/completions          │     │
│   │  /v1/completions               │     │
│   │  /health                       │     │
│   │  Streaming (SSE)               │     │
│   │  API Key Auth                  │     │
│   │  Rate Limiting                 │     │
│   └────────────────────────────────┘     │
│   Model: Qwen3-4B-Q4_K_M.gguf          │
│   Runs on CPU · ~2.5 GB RAM             │
└──────────────────────────────────────────┘
```

## Option 2: GPU (vLLM) — `gpu-vllm/`

**For:** Production, high throughput, concurrent users  
**Model:** Qwen/Qwen3-4B-AWQ (INT4, runs on single GPU)  
**Stack:** FastAPI gateway + vLLM inference server  
**Deploy:** RunPod, Lambda Labs, Vultr, OVH, or any NVIDIA GPU box  
**Kubernetes:** Full K8s manifests included in `gpu-vllm/k8s/`

```
┌────────────────────┐     ┌──────────────────────────────┐
│   FastAPI Gateway   │────▶│     vLLM Inference Server    │
│                     │     │                              │
│  • API key auth     │     │  • Qwen3-4B-AWQ              │
│  • Rate limiting    │     │  • Continuous batching        │
│  • Streaming proxy  │     │  • Paged attention            │
│  • Health checks    │     │  • OpenAI-compatible API      │
│  • Request logging  │     │  • ~3GB VRAM                  │
│                     │     │                              │
│  Can run anywhere   │     │  Needs NVIDIA GPU             │
│  (Railway, VPS)     │     │  (RunPod, Lambda, etc.)       │
│  PORT: 8000         │     │  PORT: 8080                   │
└────────────────────┘     └──────────────────────────────┘
```

---

## Deployment Guides

- **Railway (CPU):** See [`railway/DEPLOY.md`](railway/DEPLOY.md)
- **GPU (Docker Compose):** See [`gpu-vllm/DEPLOY.md`](gpu-vllm/DEPLOY.md)
- **GPU (DigitalOcean Kubernetes):** See [`gpu-vllm/DIGITAL_OCEAN.md`](gpu-vllm/DIGITAL_OCEAN.md)

---

## Streaming

Both options support OpenAI-compatible SSE streaming:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<YOUR_ENDPOINT>/v1", api_key="your-key")

stream = client.chat.completions.create(
    model="Qwen/Qwen3-14B-AWQ",
    messages=[{"role": "user", "content": "Tell me a quick story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Qwen3 Thinking Mode

Qwen3 models default to "thinking" mode, generating a `<think>` reasoning block before answering. This uses many tokens and slows down simple requests. Disable it with `/no_think` in the system message:

```python
messages=[
    {"role": "system", "content": "You are a concise assistant. /no_think"},
    {"role": "user", "content": "What is 1+1?"},
]
```

| Mode | "what is 1+1?" | Tokens | Time |
|---|---|---|---|
| Thinking (default) | 256 tokens of reasoning, never answers | 256 | ~26s |
| `/no_think` | "2." | 7 | ~1s |

---

## Quick Comparison

| | Railway (CPU) | GPU (vLLM) |
|---|---|---|
| Throughput | ~5-15 tok/s | ~80-200+ tok/s |
| Concurrency | 1-3 users | 50+ users |
| Cost | ~$5-20/mo | ~$0.20-0.80/hr GPU |
| Setup time | 5 min | 15 min |
| GPU required | No | Yes |
| Best for | Dev, demos, low traffic | Production, high traffic |

## GPU Provider Recommendations (non-AWS, non-GCP)

| Provider | GPU | Price/hr | Notes |
|----------|-----|----------|-------|
| **DigitalOcean** | RTX 6000 Ada (48GB) | ~$1.57 | K8s native. See [`DIGITAL_OCEAN.md`](gpu-vllm/DIGITAL_OCEAN.md) |
| **RunPod** | A40 (48GB) | ~$0.39 | Easiest. Docker deploy. |
| **Lambda Labs** | A10 (24GB) | ~$0.60 | SSH access. Great DX. |
| **Vultr** | A100 (80GB) | ~$2.06 | Bare metal. Full control. |
| **OVHcloud** | L40S (48GB) | ~$1.50 | EU-based. Good pricing. |
| **Vast.ai** | Various | ~$0.15+ | Cheapest. Spot-like. |
| **Together.ai** | — | Per-token | If you just want an API |
