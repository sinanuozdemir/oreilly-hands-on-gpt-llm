"""
FastAPI Gateway for vLLM Inference Server
Proxies OpenAI-compatible requests to a vLLM backend.

Deploy this anywhere (Railway, VPS, etc.)
Point VLLM_BASE_URL at your GPU-hosted vLLM server.
"""

import os
import time
import json
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ──────────────────────────── Config ────────────────────────────

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080")
API_KEY = os.getenv("API_KEY", "changeme")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

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

# ──────────────────────────── App ────────────────────────────

app = FastAPI(title="LLM Gateway", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()
http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def check_rate_limit():
    if not rate_limiter.check():
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

# ──────────────────────────── Endpoints ────────────────────────────

@app.get("/health")
async def health():
    """Check gateway + vLLM backend health."""
    try:
        resp = await http_client.get(f"{VLLM_BASE_URL}/health", timeout=5)
        vllm_ok = resp.status_code == 200
    except Exception:
        vllm_ok = False

    return {
        "gateway": "healthy",
        "vllm_backend": "healthy" if vllm_ok else "unreachable",
        "vllm_url": VLLM_BASE_URL,
    }


@app.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):
    """Proxy model list from vLLM."""
    resp = await http_client.get(f"{VLLM_BASE_URL}/v1/models")
    return resp.json()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    _=Depends(verify_api_key),
    __=Depends(check_rate_limit),
):
    """Proxy chat completions to vLLM. Supports streaming."""
    body = await request.json()
    is_stream = body.get("stream", False)

    if is_stream:
        return StreamingResponse(
            _proxy_stream(body),
            media_type="text/event-stream",
        )

    # Non-streaming: forward and return
    resp = await http_client.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=body,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()


async def _proxy_stream(body: dict):
    """Stream SSE events from vLLM to client."""
    async with http_client.stream(
        "POST",
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=body,
        timeout=REQUEST_TIMEOUT,
    ) as resp:
        async for line in resp.aiter_lines():
            if line:
                yield f"{line}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(
    request: Request,
    _=Depends(verify_api_key),
    __=Depends(check_rate_limit),
):
    """Proxy text completions to vLLM."""
    body = await request.json()
    resp = await http_client.post(
        f"{VLLM_BASE_URL}/v1/completions",
        json=body,
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
