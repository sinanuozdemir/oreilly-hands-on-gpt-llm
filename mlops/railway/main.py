"""
LLM Inference Server — CPU Edition (Railway-compatible)
Stack: FastAPI + llama-cpp-python
Model: Qwen3-4B-Q4_K_M (GGUF quantized)

Provides OpenAI-compatible API endpoints.
"""

import os
import time
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from llama_cpp import Llama

# ──────────────────────────── Config ────────────────────────────

MODEL_REPO = os.getenv("MODEL_REPO", "Qwen/Qwen3-4B-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "qwen3-4b-q4_k_m.gguf")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "4"))  # Railway gives ~4 vCPUs on Pro
API_KEY = os.getenv("API_KEY", "changeme")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))

# ──────────────────────────── Rate Limiter ────────────────────────────

class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.requests: list[float] = []

    def check(self) -> bool:
        now = time.time()
        self.requests = [t for t in self.requests if now - t < 60]
        if len(self.requests) >= self.rpm:
            return False
        self.requests.append(now)
        return True

rate_limiter = RateLimiter(RATE_LIMIT_RPM)

# ──────────────────────────── Model Loading ────────────────────────────

llm: Optional[Llama] = None


def download_model():
    """Download GGUF model from Hugging Face if not already present."""
    model_path = Path(MODEL_DIR) / MODEL_FILE
    if model_path.exists():
        print(f"✅ Model already exists at {model_path}")
        return str(model_path)

    print(f"⬇️  Downloading {MODEL_REPO}/{MODEL_FILE}...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(model_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192 * 16):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = (downloaded / total) * 100
                    print(f"\r  {pct:.1f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)", end="", flush=True)
        print(f"\n✅ Downloaded to {model_path}")
    return str(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    model_path = download_model()
    print(f"🔄 Loading model with {N_THREADS} threads, {N_CTX} context...")
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS,
        verbose=False,
    )
    print("✅ Model loaded and ready!")
    yield
    del llm

# ──────────────────────────── FastAPI App ────────────────────────────

app = FastAPI(
    title="LLM Inference API (CPU)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials


def check_rate_limit():
    if not rate_limiter.check():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# ──────────────────────────── Request/Response Models ────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-4b"
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    # Qwen3 thinking mode — set enable_thinking=False to disable /think tags
    enable_thinking: Optional[bool] = None


class CompletionRequest(BaseModel):
    model: str = "qwen3-4b"
    prompt: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False

# ──────────────────────────── Helpers ────────────────────────────

def format_chat_prompt(messages: list[Message], enable_thinking: Optional[bool] = None) -> str:
    """Format messages into Qwen3 chat template."""
    prompt_parts = []
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

    # Add thinking mode toggle if specified
    if enable_thinking is False:
        # Append /no_think to disable thinking for this request
        prompt_parts.append("<|im_start|>assistant\n/no_think\n")
    else:
        prompt_parts.append("<|im_start|>assistant\n")

    return "\n".join(prompt_parts)


def make_completion_response(text: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0):
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

# ──────────────────────────── Endpoints ────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": f"{MODEL_REPO}/{MODEL_FILE}",
        "backend": "llama-cpp-python (CPU)",
    }


@app.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-4b",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _=Depends(verify_api_key),
    __=Depends(check_rate_limit),
):
    prompt = format_chat_prompt(request.messages, request.enable_thinking)

    if request.stream:
        return StreamingResponse(
            _stream_chat(prompt, request),
            media_type="text/event-stream",
        )

    # Synchronous generation (run in thread to not block event loop)
    result = await asyncio.to_thread(
        llm,
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=["<|im_end|>", "<|im_start|>"],
    )

    text = result["choices"][0]["text"]
    return make_completion_response(
        text=text,
        model=request.model,
        prompt_tokens=result["usage"]["prompt_tokens"],
        completion_tokens=result["usage"]["completion_tokens"],
    )


async def _stream_chat(prompt: str, request: ChatCompletionRequest):
    """Stream tokens as SSE events (OpenAI-compatible)."""
    stream = llm(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=["<|im_end|>", "<|im_start|>"],
        stream=True,
    )
    for output in stream:
        token = output["choices"][0]["text"]
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0)  # yield control

    # Final chunk
    final = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    _=Depends(verify_api_key),
    __=Depends(check_rate_limit),
):
    result = await asyncio.to_thread(
        llm,
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "text": result["choices"][0]["text"],
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": result["usage"],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
