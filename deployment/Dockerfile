# Production Dockerfile for PDE-Fluid-Φ Neural Operators
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y     python3.10     python3-pip     python3-dev     build-essential     libopenmpi-dev     openmpi-bin     git     wget     && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY *.py /app/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose ports for distributed training
EXPOSE 29500 29501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3     CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"

# Default command
CMD ["python3", "-m", "src.pde_fluid_phi.main"]
