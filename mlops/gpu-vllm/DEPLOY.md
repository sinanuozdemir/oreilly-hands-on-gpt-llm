# GPU Deployment Guide (vLLM + FastAPI)

## Architecture

```
Your users → FastAPI Gateway (anywhere) → vLLM Server (GPU box)
```

The gateway is lightweight (no GPU needed). The vLLM server needs an NVIDIA GPU.

---

## Option A: RunPod (Easiest)

RunPod lets you deploy Docker containers directly onto GPU instances.

### 1. Deploy vLLM on RunPod

```bash
# Go to https://www.runpod.io/console/pods
# Click "Deploy" → "GPU Pod"
# Select: A40 (48GB) or RTX A5000 (24GB) — both overkill for 4B model
# Template: "RunPod PyTorch" or custom Docker

# Under "Docker Image", use:
vllm/vllm-openai:latest

# Under "Docker Command / Args", set:
--model Qwen/Qwen3-4B-AWQ --host 0.0.0.0 --port 8080 --gpu-memory-utilization 0.9 --max-model-len 4096 --quantization awq --dtype half --trust-remote-code

# Expose port 8080
# Deploy!
```

Your vLLM server will be at: `https://{pod-id}-8080.proxy.runpod.net`

### 2. Deploy Gateway (Railway, VPS, or same RunPod pod)

```bash
# Set VLLM_BASE_URL to your RunPod endpoint
export VLLM_BASE_URL=https://{pod-id}-8080.proxy.runpod.net

# Deploy gateway on Railway
cd gateway
railway init
railway variables set VLLM_BASE_URL=$VLLM_BASE_URL
railway variables set API_KEY=your-secret-key
railway up
```

### RunPod Serverless (Scale-to-Zero)

For pay-per-request pricing (no idle costs):

```bash
# Go to RunPod Console → Serverless → New Endpoint
# Docker Image: vllm/vllm-openai:latest
# GPU: 24GB+
# Handler: built-in OpenAI compatible
# Set env vars: MODEL_NAME=Qwen/Qwen3-4B-AWQ
```

---

## Option B: Lambda Labs

Lambda gives you full SSH access to GPU instances.

### 1. Launch an instance

```bash
# Go to https://cloud.lambda.ai/instances
# Select: 1x A10 (24GB VRAM) — $0.60/hr
# Launch and SSH in
```

### 2. Run vLLM on the instance

```bash
# SSH into your Lambda instance
ssh ubuntu@<your-lambda-ip>

# Pull and run vLLM
docker run -d \
  --gpus all \
  --name vllm \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B-AWQ \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --quantization awq \
  --dtype half \
  --trust-remote-code

# Verify it's running
curl http://localhost:8080/health
```

### 3. Deploy gateway pointing at Lambda

```bash
cd gateway
railway variables set VLLM_BASE_URL=http://<lambda-ip>:8080
railway variables set API_KEY=your-secret-key
railway up
```

---

## Option C: Any NVIDIA GPU Box (Vultr, OVH, Hetzner, your own)

### 1. Prerequisites

```bash
# Ensure NVIDIA drivers + Docker + nvidia-container-toolkit are installed
nvidia-smi  # should show your GPU

# If not installed:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 2. Run with Docker Compose

```bash
git clone <your-repo>
cd gpu-vllm

# Set your API key
export API_KEY=your-secret-key

# Launch both services
docker compose up -d

# Check logs
docker compose logs -f vllm
docker compose logs -f gateway
```

### 3. Or run vLLM standalone

```bash
# Just the vLLM server (if gateway is elsewhere)
docker run -d \
  --gpus all \
  --name vllm \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B-AWQ \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --quantization awq \
  --dtype half \
  --trust-remote-code
```

---

## Test It

```bash
# Direct to vLLM (no auth)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-AWQ",
    "messages": [{"role": "user", "content": "What is vLLM?"}],
    "max_tokens": 256
  }'

# Through gateway (with auth)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B-AWQ",
    "messages": [{"role": "user", "content": "What is vLLM?"}],
    "max_tokens": 256,
    "stream": true
  }'
```

## OpenAI SDK (drop-in)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # or your Railway URL
    api_key="your-secret-key",
)

# Non-streaming
response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain LoRA fine-tuning in 3 sentences."},
    ],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen/Qwen3-4B-AWQ",
    messages=[{"role": "user", "content": "Write a poem about GPUs"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Serve Your Own Fine-Tuned Model

### If you fine-tuned with LoRA adapters:

```bash
# vLLM can hot-swap LoRA adapters on top of a base model!
docker run -d --gpus all \
  -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/your/lora:/lora \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B \
  --enable-lora \
  --lora-modules my-adapter=/lora \
  --host 0.0.0.0 --port 8080

# Then request with the adapter name:
# "model": "my-adapter"
```

### If you merged weights (full model):

```bash
# Push to HuggingFace (private repo works too)
huggingface-cli upload your-username/your-model ./merged-model

# Quantize with AutoAWQ for best vLLM perf
pip install autoawq
# (see AutoAWQ docs for quantization script)

# Then deploy:
docker run -d --gpus all \
  -e HF_TOKEN=your-hf-token \
  -p 8080:8080 \
  vllm/vllm-openai:latest \
  --model your-username/your-model-AWQ \
  --quantization awq \
  --host 0.0.0.0 --port 8080
```

---

## GPU Memory Reference (Qwen3 family)

| Model | Quantization | VRAM Needed | Recommended GPU |
|-------|-------------|-------------|-----------------|
| Qwen3-1.7B | AWQ (INT4) | ~2 GB | Any (T4, RTX 3060) |
| Qwen3-4B | AWQ (INT4) | ~3 GB | Any (T4, RTX 3060) |
| Qwen3-8B | AWQ (INT4) | ~5 GB | T4, RTX 3090, A10 |
| Qwen3-14B | AWQ (INT4) | ~9 GB | RTX 3090, A10, L4 |
| Qwen3-32B | AWQ (INT4) | ~18 GB | A10, L40S, A100 |
| Qwen3-30B-A3B | AWQ (MoE) | ~18 GB | A10, L40S, A100 |
