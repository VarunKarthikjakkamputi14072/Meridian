# Lightweight image for the Streamlit "god-mode" drift dashboard.
# Deliberately does NOT pull the training stack (torch/mlflow/evidently) — the
# dashboard only needs streamlit + httpx + pandas and talks to serving over HTTP.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# app.py does `import nim`, so both live side-by-side on Streamlit's script path.
COPY app.py nim.py ./

EXPOSE 8501

# Healthcheck against Streamlit's built-in endpoint.
HEALTHCHECK --interval=10s --timeout=3s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
