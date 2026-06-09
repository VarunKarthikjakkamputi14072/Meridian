FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Torch CPU wheels are large; install deps first for layer caching.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
 && pip install -r requirements.txt

COPY src ./src

# Default command is overridden per-service in docker-compose.
CMD ["uvicorn", "meridian.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
