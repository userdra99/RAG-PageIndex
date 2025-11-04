#!/bin/bash
#
# Health Check Script for vLLM + Qwen3 + PageIndex Integration
# Part of Compatibility Test Strategy
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_info "✓ $1 is available"
        return 0
    else
        log_error "✗ $1 is not available"
        return 1
    fi
}

# ==========================================
# 1. SYSTEM PREREQUISITES
# ==========================================
log_info "============================================"
log_info "1. CHECKING SYSTEM PREREQUISITES"
log_info "============================================"

check_command docker || exit 1
check_command nvidia-smi || exit 1
check_command curl || exit 1

# ==========================================
# 2. GPU AVAILABILITY
# ==========================================
log_info ""
log_info "============================================"
log_info "2. CHECKING GPU AVAILABILITY"
log_info "============================================"

if nvidia-smi &> /dev/null; then
    log_info "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_error "✗ No NVIDIA GPU detected"
    exit 1
fi

# ==========================================
# 3. DOCKER GPU RUNTIME
# ==========================================
log_info ""
log_info "============================================"
log_info "3. CHECKING DOCKER GPU RUNTIME"
log_info "============================================"

if docker info | grep -q "nvidia"; then
    log_info "✓ Docker NVIDIA runtime detected"
else
    log_warn "⚠ Docker NVIDIA runtime not configured"
    log_info "Run: sudo nvidia-ctk runtime configure --runtime=docker"
fi

# ==========================================
# 4. GPU ACCESS FROM CONTAINER
# ==========================================
log_info ""
log_info "============================================"
log_info "4. TESTING GPU ACCESS FROM CONTAINER"
log_info "============================================"

if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    log_info "✓ Container can access GPU"
else
    log_error "✗ Container cannot access GPU"
    exit 1
fi

# ==========================================
# 5. PYTHON ENVIRONMENT
# ==========================================
log_info ""
log_info "============================================"
log_info "5. CHECKING PYTHON ENVIRONMENT"
log_info "============================================"

python3 --version
pip3 --version 2>/dev/null || log_warn "pip3 not available"

# ==========================================
# 6. NETWORK CONNECTIVITY
# ==========================================
log_info ""
log_info "============================================"
log_info "6. CHECKING NETWORK CONNECTIVITY"
log_info "============================================"

if curl -s -I https://huggingface.co | grep -q "200"; then
    log_info "✓ Can reach HuggingFace"
else
    log_warn "⚠ Cannot reach HuggingFace (may affect model downloads)"
fi

# ==========================================
# 7. DISK SPACE
# ==========================================
log_info ""
log_info "============================================"
log_info "7. CHECKING DISK SPACE"
log_info "============================================"

AVAILABLE=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
log_info "Available space: $AVAILABLE"
log_info "(Qwen3-7B requires ~15GB)"

# ==========================================
# 8. OPTIONAL: VLLM INSTALLATION CHECK
# ==========================================
log_info ""
log_info "============================================"
log_info "8. CHECKING VLLM INSTALLATION (if available)"
log_info "============================================"

# Check if virtual environment exists
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    if python -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)")
        log_info "✓ vLLM $VLLM_VERSION installed (in virtual environment)"
    else
        log_warn "⚠ vLLM not installed in virtual environment"
        log_info "  Run: ./scripts/setup_venv.sh"
    fi
    deactivate 2>/dev/null || true
else
    log_warn "⚠ Virtual environment not found"
    log_info "  Run: ./scripts/setup_venv.sh to create it"
fi

# ==========================================
# 9. OPTIONAL: TEST CONTAINER CHECK
# ==========================================
log_info ""
log_info "============================================"
log_info "9. CHECKING FOR TEST CONTAINERS"
log_info "============================================"

if docker ps -a --format '{{.Names}}' | grep -q "vllm"; then
    log_info "Existing vLLM containers:"
    docker ps -a --filter "name=vllm" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
else
    log_info "No vLLM containers found"
fi

# ==========================================
# SUMMARY
# ==========================================
log_info ""
log_info "============================================"
log_info "HEALTH CHECK COMPLETE"
log_info "============================================"
log_info "✓ System is ready for vLLM + Qwen3 testing"
log_info ""
log_info "Next steps:"
log_info "1. Run GPU detection tests: python3 tests/test_gpu_detection.py"
log_info "2. Build Docker image: docker build -t vllm-qwen3:test ."
log_info "3. Run integration tests: pytest tests/integration/"
log_info ""

exit 0
