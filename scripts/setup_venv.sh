#!/bin/bash
# Virtual Environment Setup Script for vLLM + Qwen3 Testing
# This script creates and configures the Python virtual environment

set -e

VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT"

echo "[INFO] ============================================"
echo "[INFO] vLLM Virtual Environment Setup"
echo "[INFO] ============================================"
echo ""

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "[INFO] ✓ Virtual environment already exists"
else
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "[INFO] ✓ Virtual environment created"
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "[INFO] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install vLLM if not already installed
if python -c "import vllm" 2>/dev/null; then
    echo "[INFO] ✓ vLLM already installed"
    python -c "import vllm; print(f'[INFO]   Version: {vllm.__version__}')"
else
    echo "[INFO] Installing vLLM (this may take several minutes)..."
    pip install vllm
    echo "[INFO] ✓ vLLM installed successfully"
fi

# Install additional testing dependencies
echo "[INFO] Installing testing dependencies..."
pip install pytest pytest-asyncio pytest-cov httpx 2>/dev/null || true

echo ""
echo "[INFO] ============================================"
echo "[INFO] Setup Complete!"
echo "[INFO] ============================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  source .venv/bin/activate"
echo "  python tests/test_gpu_detection.py"
echo ""
