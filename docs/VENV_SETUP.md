# Virtual Environment Setup Guide

## Overview

This project uses a Python virtual environment to isolate dependencies and avoid conflicts with system packages. This guide explains how to set up and use the virtual environment for vLLM + Qwen3 testing.

## Quick Start

### Automated Setup

Run the setup script to create and configure the virtual environment automatically:

```bash
./scripts/setup_venv.sh
```

This script will:
- Create a Python virtual environment in `.venv/`
- Upgrade pip, setuptools, and wheel
- Install vLLM and all dependencies
- Install testing dependencies (pytest, httpx, etc.)

### Manual Setup

If you prefer to set up manually:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install vLLM
pip install vllm

# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov httpx
```

## Using the Virtual Environment

### Activating the Environment

Before running any Python scripts or commands, activate the virtual environment:

```bash
source .venv/bin/activate
```

You'll see `(.venv)` prepended to your terminal prompt, indicating the environment is active.

### Running Tests

With the virtual environment activated:

```bash
# Run GPU detection tests
python tests/test_gpu_detection.py

# Run health check
./tests/health_check.sh

# Run pytest suite (when available)
pytest tests/
```

### Deactivating the Environment

When you're done working:

```bash
deactivate
```

## Verifying Installation

Check that vLLM is installed correctly:

```bash
source .venv/bin/activate
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

Expected output:
```
vLLM version: 0.11.0
```

## Installed Packages

The virtual environment includes:

### Core Dependencies
- **vLLM 0.11.0**: High-performance LLM inference engine
- **PyTorch 2.8.0**: Deep learning framework with CUDA 12.8 support
- **transformers**: HuggingFace model loading
- **xformers**: Memory-efficient attention mechanisms

### GPU Support
- **CUDA 12.8**: NVIDIA GPU acceleration
- **cuDNN 9.10**: Neural network primitives
- **NCCL**: Multi-GPU communication
- **Triton 3.4.0**: GPU programming language

### API & Serving
- **FastAPI**: REST API framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation

### Testing
- **pytest**: Testing framework
- **pytest-asyncio**: Async testing support
- **httpx**: Async HTTP client

## Troubleshooting

### Permission Errors

If you encounter permission errors, ensure you're not using `sudo` with the virtual environment:

```bash
# ✗ Wrong
sudo pip install vllm

# ✓ Correct
source .venv/bin/activate
pip install vllm
```

### Externally-Managed Environment Error

If you see this error when trying to install packages:

```
error: externally-managed-environment
```

**Solution**: Always use the virtual environment! Run:

```bash
source .venv/bin/activate
```

### Missing Dependencies

If tests fail due to missing packages:

```bash
source .venv/bin/activate
pip install pytest pytest-asyncio httpx
```

### GPU Not Detected in Virtual Environment

The virtual environment doesn't affect GPU detection - ensure you have:
- NVIDIA drivers installed (580.65.06 or newer)
- Docker with NVIDIA runtime configured
- `nvidia-smi` accessible from terminal

## Python Version Support

### Supported Versions
- Python 3.10 ✓
- Python 3.11 ✓
- Python 3.12 ✓ (current: 3.12.3)

### Note on Version Checks

The test script `test_gpu_detection.py` has been updated to accept Python 3.10+ (previously flagged 3.12 as incompatible, which was incorrect).

## Directory Structure

```
PageIndex-Home/
├── .venv/                 # Virtual environment (not in git)
│   ├── bin/              # Executables (python, pip, etc.)
│   ├── lib/              # Python packages
│   └── pyvenv.cfg        # Environment configuration
├── scripts/
│   └── setup_venv.sh     # Automated setup script
├── tests/
│   ├── test_gpu_detection.py
│   └── health_check.sh
└── docs/
    └── VENV_SETUP.md     # This file
```

## Best Practices

1. **Always activate before working**:
   ```bash
   source .venv/bin/activate
   ```

2. **Don't commit the virtual environment**:
   - `.venv/` is in `.gitignore`
   - Other developers should run `./scripts/setup_venv.sh`

3. **Keep dependencies updated**:
   ```bash
   source .venv/bin/activate
   pip install --upgrade vllm
   ```

4. **Use requirements.txt for reproducibility**:
   ```bash
   # Export current environment
   pip freeze > requirements.txt

   # Install from requirements
   pip install -r requirements.txt
   ```

## Integration with Docker

The virtual environment is for **local testing only**. For production deployment:

1. **Local development**: Use `.venv/` for testing scripts
2. **Docker deployment**: Use `Dockerfile` for containerized inference

The Docker container has its own isolated Python environment with vLLM pre-installed.

## Next Steps

After setting up the virtual environment:

1. **Run health checks**: `./tests/health_check.sh`
2. **Test GPU detection**: `python tests/test_gpu_detection.py`
3. **Build Docker image**: `docker build -t vllm-qwen3:test .`
4. **Run integration tests**: `pytest tests/integration/` (when available)

## Support

For issues with the virtual environment:
- Check this guide's troubleshooting section
- Review `scripts/setup_venv.sh` for setup details
- Ensure Python 3.10+ is installed on your system
