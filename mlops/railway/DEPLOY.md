# Railway CPU Deployment Guide

## Deploy to Railway

### Option A: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project
railway init

# Set environment variables
railway variables set API_KEY=your-secret-key-here
railway variables set MODEL_REPO=Qwen/Qwen3-4B-GGUF
railway variables set MODEL_FILE=qwen3-4b-q4_k_m.gguf
railway variables set N_CTX=4096
railway variables set N_THREADS=4

# Deploy
railway up
```

### Option B: GitHub (auto-deploy)
1. Push this `railway-cpu/` folder to a GitHub repo
2. Go to https://railway.com/new
3. Select "Deploy from GitHub repo"
4. Pick the repo → Railway auto-detects the Dockerfile
5. Add environment variables in the Railway dashboard
6. Deploy 🚀

## Railway Plan Recommendations

- **Pro Plan ($20/mo)**: 32 GB RAM, 32 vCPUs — more than enough
- Set **N_THREADS=4** to match allocated vCPUs (adjust based on your plan)
- The Q4_K_M model needs ~2.5 GB RAM at rest, ~4 GB under load
- **Add a volume** at `/app/models` so the model persists across deploys:
  ```bash
  railway volume create --mount /app/models
  ```

## Test It

```bash
# Health check
curl https://your-app.railway.app/health

# Chat completion
curl https://your-app.railway.app/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Explain transformers in 2 sentences"}],
    "max_tokens": 256
  }'

# Streaming
curl https://your-app.railway.app/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "Write a haiku about LLMs"}],
    "stream": true
  }'
```

## Use with OpenAI SDK (drop-in replacement)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-app.railway.app/v1",
    api_key="your-secret-key-here",
)

response = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "What is agentic AI?"}],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Swap Models

Change `MODEL_REPO` and `MODEL_FILE` to any GGUF model on HuggingFace:

```bash
# Smaller/faster
railway variables set MODEL_REPO=Qwen/Qwen3-1.7B-GGUF
railway variables set MODEL_FILE=qwen3-1.7b-q4_k_m.gguf

# Or use a fine-tuned model you uploaded to HF
railway variables set MODEL_REPO=your-username/your-finetuned-model-GGUF
railway variables set MODEL_FILE=model-q4_k_m.gguf
```

## Performance Notes

- CPU inference is ~5-15 tokens/sec depending on Railway plan
- Good enough for demos, internal tools, low-traffic APIs
- For production/high-throughput, use the GPU version (gpu-vllm/)
