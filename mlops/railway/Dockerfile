FROM python:3.11-slim

WORKDIR /app

# Build deps for llama-cpp-python (compiles from source for CPU optimizations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Build llama-cpp-python with CPU optimizations
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Railway sets PORT env var automatically
ENV PORT=8000
EXPOSE 8000

# Pre-create model directory (model downloads on first boot)
RUN mkdir -p /app/models

CMD ["python", "main.py"]
