FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/opt/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding + reranker models into the image (no download at startup).
COPY config.py .
RUN python -c "import config; \
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding(config.EMBED_MODEL, cache_dir=config.MODEL_CACHE_DIR); \
TextCrossEncoder(config.RERANK_MODEL, cache_dir=config.MODEL_CACHE_DIR)"

COPY . .
RUN useradd --create-home appuser && mkdir -p /app/data /app/logs && chown -R appuser /app /opt/models
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
