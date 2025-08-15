#!/bin/bash

# Production Entrypoint Script for PDE-Fluid-Φ Neural Operator Framework
# Handles initialization, configuration, and command routing

set -euo pipefail

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Default values
ENVIRONMENT=${ENVIRONMENT:-production}
CONFIG_PATH=${CONFIG_PATH:-/app/configs/production_config.yaml}
LOG_LEVEL=${LOG_LEVEL:-INFO}
PYTHONPATH=${PYTHONPATH:-/app}

# Export environment variables
export ENVIRONMENT CONFIG_PATH LOG_LEVEL PYTHONPATH

# Set CUDA environment
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Function for logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ENTRYPOINT] $1" | tee -a /app/logs/entrypoint.log
}

log "Starting PDE-Fluid-Φ Neural Operator Framework"
log "Environment: $ENVIRONMENT"
log "Config Path: $CONFIG_PATH"
log "Log Level: $LOG_LEVEL"

# =============================================================================
# SYSTEM CHECKS
# =============================================================================

# Check if running as non-root user
if [ "$(id -u)" = "0" ]; then
    log "WARNING: Running as root user. Consider using non-root user for security."
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    log "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
        log "  GPU: $line"
    done
else
    log "No NVIDIA GPU detected or nvidia-smi not available"
fi

# Check Python environment
log "Python version: $(python --version)"
log "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
log "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    log "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"
fi

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    log "ERROR: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

log "Configuration file found: $CONFIG_PATH"

# Validate configuration
python -c "
import yaml
import sys
try:
    with open('$CONFIG_PATH', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration validation: PASSED')
except Exception as e:
    print(f'Configuration validation: FAILED - {e}')
    sys.exit(1)
" || exit 1

# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

log "Checking Python dependencies..."

# Check critical dependencies
python -c "
import sys
dependencies = [
    'torch', 'numpy', 'scipy', 'matplotlib', 'tqdm', 'pyyaml', 
    'h5py', 'netcdf4', 'einops', 'psutil'
]

missing = []
for dep in dependencies:
    try:
        __import__(dep)
    except ImportError:
        missing.append(dep)

if missing:
    print(f'Missing dependencies: {missing}')
    sys.exit(1)
else:
    print('All critical dependencies available')
"

# Check optional dependencies
python -c "
optional_deps = ['wandb', 'tensorboard']
available = []
for dep in optional_deps:
    try:
        __import__(dep)
        available.append(dep)
    except ImportError:
        pass

if available:
    print(f'Optional dependencies available: {available}')
else:
    print('No optional dependencies found')
" || true

# =============================================================================
# DATA DIRECTORY SETUP
# =============================================================================

# Create necessary directories
mkdir -p /app/data/train
mkdir -p /app/data/validation
mkdir -p /app/data/test
mkdir -p /app/checkpoints
mkdir -p /app/artifacts
mkdir -p /app/logs/training
mkdir -p /app/logs/monitoring

log "Data directories created"

# Check data availability
if [ -d "/app/data/train" ] && [ "$(ls -A /app/data/train 2>/dev/null)" ]; then
    log "Training data found in /app/data/train"
else
    log "WARNING: No training data found in /app/data/train"
fi

# =============================================================================
# DISTRIBUTED TRAINING SETUP
# =============================================================================

# Set up distributed training environment if multiple GPUs are available
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -gt 1 ]; then
        log "Multi-GPU setup detected: $GPU_COUNT GPUs"
        export WORLD_SIZE=${WORLD_SIZE:-$GPU_COUNT}
        export RANK=${RANK:-0}
        export LOCAL_RANK=${LOCAL_RANK:-0}
        export MASTER_ADDR=${MASTER_ADDR:-localhost}
        export MASTER_PORT=${MASTER_PORT:-12355}
        
        log "Distributed training environment:"
        log "  WORLD_SIZE: $WORLD_SIZE"
        log "  RANK: $RANK"
        log "  LOCAL_RANK: $LOCAL_RANK"
        log "  MASTER_ADDR: $MASTER_ADDR"
        log "  MASTER_PORT: $MASTER_PORT"
    else
        log "Single GPU setup detected"
    fi
fi

# =============================================================================
# MONITORING SETUP
# =============================================================================

# Start monitoring in background if requested
if [ "${START_MONITORING:-false}" = "true" ]; then
    log "Starting monitoring services..."
    python /app/scripts/monitor.py &
    MONITOR_PID=$!
    log "Monitoring started with PID: $MONITOR_PID"
fi

# =============================================================================
# SIGNAL HANDLERS
# =============================================================================

# Graceful shutdown handler
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill monitoring if running
    if [ -n "${MONITOR_PID:-}" ]; then
        log "Stopping monitoring (PID: $MONITOR_PID)"
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    # Additional cleanup
    log "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# =============================================================================
# COMMAND ROUTING
# =============================================================================

# Get command
COMMAND=${1:-train}

log "Executing command: $COMMAND"

case "$COMMAND" in
    "train")
        log "Starting training mode"
        exec python /app/scripts/train.py "${@:2}"
        ;;
    
    "infer"|"inference")
        log "Starting inference mode"
        exec python -m src.pde_fluid_phi.inference.inference_engine "${@:2}"
        ;;
    
    "api")
        log "Starting API server mode"
        exec python -m src.pde_fluid_phi.api.server "${@:2}"
        ;;
    
    "validate")
        log "Starting validation mode"
        exec python -m src.pde_fluid_phi.validation.validator "${@:2}"
        ;;
    
    "benchmark")
        log "Starting benchmark mode"
        exec python -m src.pde_fluid_phi.optimization.benchmark "${@:2}"
        ;;
    
    "test")
        log "Starting test mode"
        exec python -m pytest tests/ "${@:2}"
        ;;
    
    "shell"|"bash")
        log "Starting interactive shell"
        exec /bin/bash
        ;;
    
    "python")
        log "Starting Python interpreter"
        exec python "${@:2}"
        ;;
    
    "jupyter")
        log "Starting Jupyter server"
        exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root "${@:2}"
        ;;
    
    "tensorboard")
        log "Starting TensorBoard"
        exec tensorboard --logdir=/app/logs --host=0.0.0.0 --port=6006 "${@:2}"
        ;;
    
    "monitor")
        log "Starting monitoring dashboard"
        exec python /app/scripts/monitor.py "${@:2}"
        ;;
    
    "help"|"--help"|"-h")
        echo "PDE-Fluid-Φ Neural Operator Framework"
        echo "Available commands:"
        echo "  train      - Start training (default)"
        echo "  infer      - Run inference"
        echo "  api        - Start API server"
        echo "  validate   - Validate model and data"
        echo "  benchmark  - Run performance benchmarks"
        echo "  test       - Run test suite"
        echo "  shell      - Interactive shell"
        echo "  python     - Python interpreter"
        echo "  jupyter    - Jupyter Lab server"
        echo "  tensorboard- TensorBoard server"
        echo "  monitor    - Monitoring dashboard"
        echo "  help       - Show this help"
        exit 0
        ;;
    
    *)
        log "ERROR: Unknown command: $COMMAND"
        log "Use 'help' to see available commands"
        exit 1
        ;;
esac

# =============================================================================
# FALLBACK
# =============================================================================

# This should not be reached, but just in case
log "ERROR: Command execution failed"
exit 1