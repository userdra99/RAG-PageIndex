# PageIndex Quick Start Guide

## Overview

Simple Docker setup for PageIndex with vLLM (Qwen3 model) integration.

## Prerequisites

1. **Docker & Docker Compose**
2. **NVIDIA GPU with CUDA support**
3. **NVIDIA Container Toolkit**

## Quick Setup

### 1. Install NVIDIA Container Toolkit (One-time)

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 2. Configure Environment

```bash
# Copy environment template
cp config/.env.example .env

# No changes needed for defaults, or customize:
# - Model name
# - Port numbers
# - GPU settings
```

### 3. Launch Services

```bash
# Start both services (vLLM + PageIndex)
docker compose -f config/docker-compose.yml up -d

# First run takes ~5-10 minutes (downloading Qwen3 model ~7GB)
# Watch progress
docker compose -f config/docker-compose.yml logs -f vllm
```

### 4. Verify Setup

```bash
# Check service status
docker compose -f config/docker-compose.yml ps

# Test vLLM API
curl http://localhost:8000/v1/models

# Test PageIndex
curl http://localhost:3000/health
```

## Architecture

```
Host Machine (GPU)
├── vLLM Service (Port 8000)
│   ├── Qwen/Qwen2.5-7B-Instruct model
│   ├── OpenAI-compatible API
│   └── GPU acceleration
│
└── PageIndex App (Port 3000)
    ├── Node.js application
    ├── Connects to vLLM internally
    └── REST API endpoints
```

## API Usage

### Chat Completion Example

```javascript
const axios = require('axios');

const response = await axios.post('http://localhost:8000/v1/chat/completions', {
  model: 'Qwen/Qwen2.5-7B-Instruct',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Hello!' }
  ],
  max_tokens: 512,
  temperature: 0.7
});

console.log(response.data.choices[0].message.content);
```

### Text Completion Example

```javascript
const response = await axios.post('http://localhost:8000/v1/completions', {
  model: 'Qwen/Qwen2.5-7B-Instruct',
  prompt: 'Explain Docker in simple terms:',
  max_tokens: 256
});

console.log(response.data.choices[0].text);
```

## Management Commands

```bash
# View logs
docker compose -f config/docker-compose.yml logs -f

# Restart services
docker compose -f config/docker-compose.yml restart

# Stop services
docker compose -f config/docker-compose.yml down

# Stop and remove volumes (fresh start)
docker compose -f config/docker-compose.yml down -v

# Update services
docker compose -f config/docker-compose.yml pull
docker compose -f config/docker-compose.yml up -d
```

## Troubleshooting

### vLLM not starting
```bash
# Check GPU availability
nvidia-smi

# View detailed logs
docker compose -f config/docker-compose.yml logs vllm

# Common issues:
# - Out of GPU memory: Reduce GPU_MEMORY_UTILIZATION in .env
# - CUDA version mismatch: Update NVIDIA drivers
# - Model download failed: Check internet connection
```

### PageIndex can't connect to vLLM
```bash
# Test internal connectivity
docker compose -f config/docker-compose.yml exec pageindex curl http://vllm:8000/health

# Check network
docker network inspect pageindex-network

# Restart both services
docker compose -f config/docker-compose.yml restart
```

### Slow performance
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Check memory usage
docker stats

# Optimize:
# - Increase GPU_MEMORY_UTILIZATION (up to 0.95)
# - Reduce MAX_MODEL_LEN if not needed
# - Use batching for multiple requests
```

## Files Created

- `/config/docker-compose.yml` - Main orchestration
- `/config/.env.example` - Configuration template
- `/Dockerfile` - PageIndex container build
- `/docs/ARCHITECTURE.md` - Detailed architecture
- `/docs/INTEGRATION_GUIDE.md` - Code integration examples
- `/docs/QUICK_START.md` - This file

## Next Steps

1. **Integration**: See `/docs/INTEGRATION_GUIDE.md` for code examples
2. **Customization**: Edit `.env` for your requirements
3. **Scaling**: See architecture docs for multi-GPU setup
4. **Production**: Add monitoring, backups, SSL/TLS

## Key Features

- Simple 2-service architecture
- OpenAI-compatible API
- GPU-accelerated inference
- Automatic model caching
- Health checks & auto-restart
- Persistent data storage
- Easy maintenance

## Support

- Full architecture: `/docs/ARCHITECTURE.md`
- Integration guide: `/docs/INTEGRATION_GUIDE.md`
- vLLM docs: https://docs.vllm.ai
- Qwen3 docs: https://huggingface.co/Qwen
