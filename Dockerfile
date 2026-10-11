FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ models/
COPY scripts/ scripts/
RUN python scripts/fetch_models.py
COPY src/ src/

RUN useradd -m -u 1000 app
USER app
EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fs http://localhost:8100/health || exit 1
# One process: the pool inside it runs FACE_POOL_SIZE requests at once.
CMD ["uvicorn", "face_service.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
