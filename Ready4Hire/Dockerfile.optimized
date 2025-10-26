# ==============================================================================
# Multi-stage Dockerfile for Ready4Hire - CI/CD Optimized
# ==============================================================================
# Features:
# - Multi-stage build for minimal image size
# - Layer caching optimization
# - Security best practices
# - Non-root user
# - Health checks
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base Dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    portaudio19-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ------------------------------------------------------------------------------
# Stage 2: Builder
# ------------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt && \
    pip install --no-cache-dir --no-deps --find-links /build/wheels -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 3: Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bullseye AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.local/bin:$PATH" \
    OLLAMA_URL=http://localhost:11434 \
    LOG_LEVEL=INFO

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    portaudio19-dev \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user early
RUN useradd --create-home --uid 1000 --shell /bin/bash appuser && \
    mkdir -p /app /app/logs /app/embeddings/cache /tmp/ready4hire_audio && \
    chown -R appuser:appuser /app /tmp/ready4hire_audio

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=appuser:appuser /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser pyproject.toml ./

# Copy datasets (if needed)
COPY --chown=appuser:appuser app/datasets/ ./app/datasets/

# Add labels for better traceability
LABEL maintainer="Ready4Hire Team" \
      version="3.3" \
      description="AI-powered interview system with FastAPI" \
      org.opencontainers.image.source="https://github.com/yourusername/Ready4Hire"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/api/v2/health || exit 1

# Expose port
EXPOSE 8000

# Start application with optimized settings
CMD ["uvicorn", "app.main_v2_improved:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

