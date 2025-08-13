# Multi-stage Dockerfile for PDE-Fluid-Φ production deployment
# Optimized for performance, security, and minimal attack surface

# Build stage
FROM python:3.11-slim-bullseye AS builder

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt -r requirements-prod.txt

# Copy source code
COPY src/ ./src/
COPY setup.py pyproject.toml README.md ./

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim-bullseye AS production

# Create non-root user for security
RUN groupadd -r pde && useradd -r -g pde -s /bin/false pde

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libblas3 \
    liblapack3 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src ./src
COPY --chown=pde:pde config/ ./config/
COPY --chown=pde:pde scripts/ ./scripts/

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/models /app/logs /app/temp && \
    chown -R pde:pde /app && \
    chmod -R 755 /app && \
    chmod -R 770 /app/data /app/models /app/logs /app/temp

# Security hardening
RUN chmod 600 /app/config/* || true

# Set environment variables
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="" \
    PDE_FLUID_PHI_CONFIG_DIR=/app/config \
    PDE_FLUID_PHI_DATA_DIR=/app/data \
    PDE_FLUID_PHI_LOG_DIR=/app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.pde_fluid_phi; print('OK')" || exit 1

# Switch to non-root user
USER pde

# Expose port (if running as web service)
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.pde_fluid_phi.applications.inference_server"]

# Development stage (optional)
FROM production AS development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort mypy jupyter

# Install additional debugging tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

USER pde

# Override command for development
CMD ["python", "-c", "print('Development container ready. Use docker exec to run commands.')"]