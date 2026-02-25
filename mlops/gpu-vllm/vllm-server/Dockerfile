# Use the official vLLM image — comes with CUDA, PyTorch, everything
FROM vllm/vllm-openai:latest

WORKDIR /app

COPY start.sh .
RUN chmod +x start.sh

# Defaults (override via env vars)
ENV MODEL_NAME=Qwen/Qwen3-4B-AWQ
ENV PORT=8080
ENV GPU_MEMORY_UTILIZATION=0.90
ENV MAX_MODEL_LEN=4096
ENV MAX_NUM_SEQS=64

EXPOSE 8080

CMD ["./start.sh"]
