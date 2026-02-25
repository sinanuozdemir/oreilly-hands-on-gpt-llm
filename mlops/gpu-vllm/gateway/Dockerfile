FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic

COPY main.py .

ENV PORT=8000
EXPOSE 8000

CMD ["python", "main.py"]
