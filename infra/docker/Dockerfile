# ============================================================================
# AtlasRAG Multi-Stage Dockerfile
# ============================================================================
# This Dockerfile creates optimized images for development and production.
#
# Stages:
# - base: Common dependencies and setup
# - development: Development with hot-reload, full dev tools
# - production: Optimized production image with minimal size
#
# Build:
#   docker build -t atlasrag:dev --target development .
#   docker build -t atlasrag:prod --target production .
#
# Run:
#   docker run -p 8000:8000 atlasrag:dev
# ============================================================================

# ============================================================================
# Base Stage - Common dependencies and setup
# ============================================================================
FROM python:3.11-slim as base

# Metadata
LABEL maintainer="AtlasRAG Team"
LABEL description="AtlasRAG Enterprise RAG Platform"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/atlasrag/scripts:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency specification
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .

# Copy atlasrag source code
COPY ./atlasrag ./atlasrag

# ============================================================================
# Development Stage - Hot-reload enabled for active development
# ============================================================================
FROM base as development

# Install additional development dependencies
RUN pip install -e ".[dev]"

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with auto-reload for development
CMD ["uvicorn", \
     "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--reload", \
     "--reload-dir", "/app/atlasrag/src"]

# ============================================================================
# Production Stage - Optimized minimal image
# ============================================================================
FROM base as production

# Create non-root user for security
RUN useradd -m -u 1000 atlasrag && \
    chown -R atlasrag:atlasrag /app

# Switch to non-root user
USER atlasrag

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with production settings (multiple workers, no reload)
CMD ["uvicorn", \
     "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools"]
