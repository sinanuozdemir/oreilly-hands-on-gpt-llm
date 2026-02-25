#!/usr/bin/env python3
"""
Test client for both Railway (CPU) and GPU (vLLM) versions.
Works with any OpenAI-compatible endpoint.

Usage:
    python test_client.py                          # defaults to localhost:8000
    python test_client.py https://your.railway.app # custom URL
"""

import sys
import os

# pip install openai
from openai import OpenAI

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "changeme")

client = OpenAI(
    base_url=f"{BASE_URL}/v1",
    api_key=API_KEY,
)

def test_models():
    print("=" * 60)
    print("📋 Available Models")
    print("=" * 60)
    models = client.models.list()
    for m in models.data:
        print(f"  • {m.id}")
    print()


def test_chat():
    print("=" * 60)
    print("💬 Chat Completion (non-streaming)")
    print("=" * 60)
    response = client.chat.completions.create(
        model="qwen3-4b",  # or "Qwen/Qwen3-4B-AWQ" for GPU version
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is LoRA fine-tuning? 2 sentences max."},
        ],
        max_tokens=256,
        temperature=0.7,
    )
    print(f"  Model: {response.model}")
    print(f"  Tokens: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion")
    print(f"  Response: {response.choices[0].message.content}")
    print()


def test_stream():
    print("=" * 60)
    print("🌊 Chat Completion (streaming)")
    print("=" * 60)
    stream = client.chat.completions.create(
        model="qwen3-4b",
        messages=[{"role": "user", "content": "Explain agentic AI in 3 bullet points."}],
        max_tokens=512,
        stream=True,
    )
    print("  ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def test_multi_turn():
    print("=" * 60)
    print("🔄 Multi-turn Conversation")
    print("=" * 60)
    messages = [
        {"role": "system", "content": "You are a helpful AI tutor."},
        {"role": "user", "content": "What is a transformer model?"},
    ]

    # Turn 1
    r1 = client.chat.completions.create(model="qwen3-4b", messages=messages, max_tokens=256)
    print(f"  User: What is a transformer model?")
    print(f"  Assistant: {r1.choices[0].message.content[:200]}...")

    # Turn 2
    messages.append({"role": "assistant", "content": r1.choices[0].message.content})
    messages.append({"role": "user", "content": "How does self-attention work in one sentence?"})

    r2 = client.chat.completions.create(model="qwen3-4b", messages=messages, max_tokens=256)
    print(f"  User: How does self-attention work in one sentence?")
    print(f"  Assistant: {r2.choices[0].message.content}")
    print()


if __name__ == "__main__":
    print(f"\n🎯 Testing endpoint: {BASE_URL}")
    print(f"🔑 API Key: {API_KEY[:8]}...\n")

    try:
        test_models()
        test_chat()
        test_stream()
        test_multi_turn()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
