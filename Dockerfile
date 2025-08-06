# Multi-stage production-ready Dockerfile for PDE-Fluid-Φ
# Optimized for neural operator training and inference
# Supports both CPU and GPU deployments with security hardening

# Build arguments for metadata
ARG VERSION=latest
ARG BUILD_DATE
ARG VCS_REF

# Base stage with common dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    libopenmpi-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash pde_user
WORKDIR /app
RUN chown pde_user:pde_user /app

# CPU-only stage
FROM base as cpu

USER pde_user

# Install CPU PyTorch
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install package
COPY --chown=pde_user:pde_user . .
RUN pip install --user -e ".[viz,docs]"

# Set path
ENV PATH="/home/pde_user/.local/bin:${PATH}"

ENTRYPOINT ["pde-fluid-phi"]
CMD ["--help"]

# GPU stage
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    libopenmpi-dev \
    pkg-config \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash pde_user
WORKDIR /app
RUN chown pde_user:pde_user /app

FROM gpu-base as gpu

USER pde_user

# Install GPU PyTorch and CUDA dependencies
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install --user cupy-cuda11x triton

# Copy and install package
COPY --chown=pde_user:pde_user . .
RUN pip install --user -e ".[cuda,viz,docs]"

# Set path
ENV PATH="/home/pde_user/.local/bin:${PATH}"

ENTRYPOINT ["pde-fluid-phi"]
CMD ["--help"]

# Development stage with all tools
FROM gpu as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    jupyter \
    && rm -rf /var/lib/apt/lists/*

USER pde_user

# Install development dependencies
RUN pip install --user -e ".[all]"

# Install Jupyter extensions
RUN pip install --user jupyterlab ipywidgets

# Expose Jupyter port
EXPOSE 8888

# Override entrypoint for development
ENTRYPOINT ["bash"]

# Production stage (minimal and secure)
FROM cpu as production

# Security hardening
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER pde_user

# Create required directories
RUN mkdir -p /app/data /app/logs /app/models \
    && chmod 755 /app/data /app/logs /app/models

# Health check with proper endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import pde_fluid_phi; print('OK')" || exit 1

# Expose application port
EXPOSE 8000

# Production entrypoint with proper signal handling
ENTRYPOINT ["python", "-m", "pde_fluid_phi.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]

# Enhanced labels for metadata
LABEL org.opencontainers.image.title="PDE-Fluid-Φ" \
      org.opencontainers.image.description="Neural Operators for High-Reynolds Number Turbulent Flows" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/pde-fluid-phi" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.documentation="https://pde-fluid-phi.readthedocs.io" \
      maintainer="research@terragonlabs.com"

# Default to production stage
FROM production