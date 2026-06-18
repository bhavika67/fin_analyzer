# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (faiss, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG BUILD_DATE
ARG VCS_REF
LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="FinAnalyzer" \
      org.opencontainers.image.description="Agentic RAG for financial documents"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Install wheels from builder stage (no compiler needed here)
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
 && rm -rf /wheels

# Copy application source
COPY --chown=app:app . .

# Create runtime directories
RUN mkdir -p data/raw data/processed data/embeddings/faiss_index \
             ui/charts reports/output \
 && chown -R app:app data ui reports

USER app

EXPOSE 8000 7860

# Health check for the FastAPI service
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Default: start the API (override in docker-compose for the UI)
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
