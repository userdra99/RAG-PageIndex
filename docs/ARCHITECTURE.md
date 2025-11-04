# PageIndex Docker Architecture

## Overview

Simple, maintainable Docker architecture for PageIndex with vLLM integration.

## Architecture Design

```
┌─────────────────────────────────────────────┐
│            Host System (GPU)                │
│                                             │
│  ┌────────────────────────────────────┐   │
│  │  PageIndex Network (Bridge)        │   │
│  │                                     │   │
│  │  ┌─────────────┐  ┌──────────────┐ │   │
│  │  │   vLLM      │  │  PageIndex   │ │   │
│  │  │   Service   │  │  App         │ │   │
│  │  │             │  │              │ │   │
│  │  │ Port: 8000  │◄─┤ Port: 3000   │ │   │
│  │  │ GPU Access  │  │              │ │   │
│  │  └─────────────┘  └──────────────┘ │   │
│  └────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Services

### 1. vLLM Service (`vllm`)

**Purpose:** Serves Qwen3 model via OpenAI-compatible API

**Configuration:**
- Base Image: `vllm/vllm-openai:latest`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- API Port: 8000
- GPU: NVIDIA GPU with CUDA support
- Memory: 90% GPU memory utilization

**Key Features:**
- OpenAI-compatible API endpoints
- Automatic model caching
- Health checks for reliability
- Optimized for single GPU

**Environment Variables:**
- `MODEL_NAME`: Specifies which model to load
- `TENSOR_PARALLEL_SIZE`: Number of GPUs (1 for single GPU)
- `GPU_MEMORY_UTILIZATION`: GPU memory allocation (0.9 = 90%)
- `MAX_MODEL_LEN`: Maximum sequence length
- `TRUST_REMOTE_CODE`: Required for Qwen models

### 2. PageIndex Application (`pageindex`)

**Purpose:** Main application service that uses vLLM for AI features

**Configuration:**
- Base Image: Node.js 20 Alpine
- Application Port: 3000
- Depends on: vLLM service health

**Key Features:**
- Multi-stage build for smaller image size
- Non-root user for security
- Health checks
- Persistent data storage
- Log management

**Environment Variables:**
- `VLLM_API_URL`: Internal URL to vLLM service
- `VLLM_MODEL_NAME`: Model identifier
- `NODE_ENV`: Environment (production/development)
- `PORT`: Application port

## Networking

**Network Type:** Bridge network (`pageindex-network`)

**Service Communication:**
- Internal: Services communicate via service names
- External: Ports exposed to host (3000, 8000)

**DNS Resolution:**
- vLLM accessible at: `http://vllm:8000`
- PageIndex accessible at: `http://pageindex:3000`

## Data Persistence

### Volumes

1. **vllm-cache**: Model cache (~7-15GB)
   - Prevents re-downloading models
   - Mounted at `/root/.cache/huggingface`

2. **pageindex-data**: Application data
   - User data, configurations
   - Mounted at `/app/data`

3. **pageindex-logs**: Application logs
   - Log files for debugging
   - Mounted at `/app/logs`

## GPU Configuration

**Requirements:**
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker Compose v1.28+ (for GPU support)

**Configuration:**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Health Checks

### vLLM Service
- Endpoint: `http://localhost:8000/health`
- Interval: 30s
- Start period: 60s (model loading time)

### PageIndex Service
- Endpoint: `http://localhost:3000/health`
- Interval: 30s
- Start period: 30s

## Security Considerations

1. **Non-root User**: Application runs as `nodejs` user
2. **Environment Variables**: Sensitive data in `.env` file
3. **Network Isolation**: Services on private bridge network
4. **Health Checks**: Automatic restart on failures

## Deployment Steps

1. **Prerequisites:**
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Configuration:**
   ```bash
   # Copy environment template
   cp config/.env.example .env

   # Edit configuration (optional)
   nano .env
   ```

3. **Launch:**
   ```bash
   # Start services
   docker compose -f config/docker-compose.yml up -d

   # View logs
   docker compose -f config/docker-compose.yml logs -f

   # Check status
   docker compose -f config/docker-compose.yml ps
   ```

4. **Verification:**
   ```bash
   # Test vLLM API
   curl http://localhost:8000/v1/models

   # Test PageIndex
   curl http://localhost:3000/health
   ```

## Integration Points

### vLLM API Usage

The PageIndex application connects to vLLM using OpenAI-compatible endpoints:

```javascript
// Example integration code
const axios = require('axios');

const vllmClient = axios.create({
  baseURL: process.env.VLLM_API_URL || 'http://vllm:8000',
  headers: {
    'Content-Type': 'application/json'
  }
});

// List available models
async function listModels() {
  const response = await vllmClient.get('/v1/models');
  return response.data;
}

// Generate completion
async function complete(prompt, options = {}) {
  const response = await vllmClient.post('/v1/completions', {
    model: process.env.VLLM_MODEL_NAME,
    prompt: prompt,
    max_tokens: options.maxTokens || 512,
    temperature: options.temperature || 0.7
  });
  return response.data;
}

// Chat completion
async function chat(messages, options = {}) {
  const response = await vllmClient.post('/v1/chat/completions', {
    model: process.env.VLLM_MODEL_NAME,
    messages: messages,
    max_tokens: options.maxTokens || 512,
    temperature: options.temperature || 0.7
  });
  return response.data;
}
```

## Scaling Considerations

### Single GPU Setup (Current)
- TENSOR_PARALLEL_SIZE=1
- Suitable for development and small-scale production

### Multi-GPU Setup (Future)
```yaml
environment:
  - TENSOR_PARALLEL_SIZE=2  # Number of GPUs
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Match TENSOR_PARALLEL_SIZE
          capabilities: [gpu]
```

## Maintenance

### Update Model
```bash
# Pull new model version
docker compose exec vllm huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Restart service
docker compose restart vllm
```

### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f vllm
docker compose logs -f pageindex
```

### Backup Data
```bash
# Backup volumes
docker run --rm -v pageindex-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/pageindex-data.tar.gz -C /data .
```

## Troubleshooting

### vLLM Issues
- Check GPU availability: `nvidia-smi`
- Verify CUDA compatibility
- Review memory usage: `docker stats`
- Check model download: `docker compose logs vllm`

### PageIndex Issues
- Verify vLLM connectivity: `docker compose exec pageindex curl http://vllm:8000/health`
- Check environment variables: `docker compose config`
- Review application logs: `docker compose logs pageindex`

## Design Principles

1. **Simplicity**: Minimal services, clear responsibilities
2. **Maintainability**: Standard images, clear configuration
3. **Reliability**: Health checks, automatic restarts
4. **Scalability**: Easy to add replicas or GPUs
5. **Security**: Non-root users, isolated networks
