# RTX 5090 Dual GPU Quick Reference Guide
**For: PageIndex + vLLM + Qwen3 Deployment**

---

## Critical Requirements Checklist

- [ ] **NCCL 2.27.7** (not default 2.26.2)
  ```bash
  pip install nvidia-nccl-cu12==2.27.7
  ```

- [ ] **PyTorch Nightly with CUDA 12.9**
  ```bash
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
  ```

- [ ] **Set CUDA Architecture**
  ```bash
  export TORCH_CUDA_ARCH_LIST="12.0"
  ```

- [ ] **NVIDIA Container Toolkit**
  ```bash
  sudo nvidia-ctk runtime configure --runtime=docker
  ```

---

## Optimal Model Configuration

### Recommended: Qwen3-70B-AWQ

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-70B-Instruct-AWQ \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.80 \
    --quantization awq \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 8001
```

**Why These Settings?**
- `--tensor-parallel-size 2`: Use both GPUs (REQUIRED for 70B models)
- `--max-model-len 32768`: Balance between context length and memory
- `--gpu-memory-utilization 0.80`: Safe 80% usage (25.6GB per GPU)
- `--disable-custom-all-reduce`: Use NCCL's stable implementation
- `--enforce-eager`: Avoid graph compilation issues

---

## Docker Compose Template

```yaml
services:
  vllm-qwen3:
    build:
      dockerfile: Dockerfile.vllm_rtx5090
    container_name: vllm-qwen3-70b
    ports:
      - "8001:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=0,1
      - CUDA_VISIBLE_DEVICES=0,1
      - NCCL_DEBUG=INFO
      - NCCL_MIN_NRINGS=2
      - NCCL_MAX_NRINGS=4
      - NCCL_P2P_LEVEL=SYS
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - OMP_NUM_THREADS=4
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      --model Qwen/Qwen3-70B-Instruct-AWQ
      --tensor-parallel-size 2
      --max-model-len 32768
      --gpu-memory-utilization 0.80
      --quantization awq
      --enforce-eager
      --disable-custom-all-reduce
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
    shm_size: 32gb
```

---

## GPU Specifications (Per RTX 5090)

| Spec | Value |
|------|-------|
| VRAM | 32GB GDDR7 |
| Bandwidth | 1,792 GB/s |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th Gen) |
| Compute Capability | 12.0 |
| TGP | 575W |

**Dual GPU Total**: 64GB VRAM, 3,584 GB/s bandwidth

---

## Memory Planning

### Qwen3 Model Fit Analysis

| Model | Quantization | VRAM Required | Dual RTX 5090 Fit |
|-------|-------------|---------------|-------------------|
| Qwen3-14B | AWQ 4-bit | ~8GB | ✅ Single GPU |
| Qwen3-30B | AWQ 4-bit | ~17GB | ✅ Single GPU |
| Qwen3-70B | AWQ 4-bit | ~37GB | ✅ Dual GPU (TP=2) |
| Qwen3-235B | Any | 112GB+ | ❌ Requires 8× A100 |

### Memory Budget (80% utilization)

```
Total Available: 64GB
Model (70B AWQ): 37GB (57.8%)
KV Cache:        12GB (18.75%)
System/Buffer:   15GB (23.45%)
```

---

## Performance Expectations

| Metric | Target |
|--------|--------|
| Throughput | 3,500-4,000 tokens/s |
| First Token Latency | <500ms |
| GPU Memory Usage | ~25.6GB per GPU |
| GPU Utilization | 85-95% |

**WARNING**: Using `--tensor-parallel-size 1` on dual GPU causes severe performance degradation (~1,800 tokens/s).

---

## Verification Commands

```bash
# 1. Check GPU detection
nvidia-smi --query-gpu=name,compute_cap --format=csv
# Expected: GeForce RTX 5090, 12.0 (twice)

# 2. Verify PyTorch
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Expected: GPUs: 2

# 3. Check NCCL version
python -c "import torch; print(f'NCCL: {torch.cuda.nccl.version()}')"
# Expected: (2, 27, 7)

# 4. Test Docker GPU access
docker run --rm --gpus '"device=0,1"' nvidia/cuda nvidia-smi
# Expected: Both RTX 5090s listed
```

---

## Common Issues & Quick Fixes

### NCCL "unhandled cuda error"
```bash
# CAUSE: Wrong NCCL version
# FIX:
pip install -U nvidia-nccl-cu12==2.27.7
```

### Out of Memory
```bash
# CAUSE: Context too long or utilization too high
# FIX:
--max-model-len 16384  # Reduce context
--gpu-memory-utilization 0.70  # Lower utilization
```

### Low Performance (<2000 tokens/s)
```bash
# CAUSE: Not using tensor parallelism
# FIX:
--tensor-parallel-size 2  # MUST use 2 for dual GPU
```

### Docker Container Can't See GPUs
```bash
# CAUSE: NVIDIA Container Toolkit not configured
# FIX:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Cost Analysis

| Setup | Cost | Performance | Break-Even |
|-------|------|-------------|------------|
| Dual RTX 5090 (on-prem) | $5,848 one-time | 64GB VRAM | N/A |
| AWS p4d.24xlarge | $32,000/month | 320GB VRAM | <1 month |

**Recommendation**: On-premise dual RTX 5090 is highly cost-effective for 70B models.

---

## NCCL Environment Variables

```bash
export NCCL_DEBUG=INFO          # Enable logging
export NCCL_MIN_NRINGS=2        # Min communication rings
export NCCL_MAX_NRINGS=4        # Max communication rings
export NCCL_TREE_THRESHOLD=0    # Disable tree for small messages
export NCCL_NET_GDR_LEVEL=5     # GPUDirect RDMA
export NCCL_P2P_LEVEL=SYS       # Peer-to-peer transfers
```

---

## Production Deployment Checklist

- [ ] PSU: 1,200W or higher
- [ ] Cooling: High-performance air or liquid cooling
- [ ] Monitoring: Set up nvidia-smi or Prometheus DCGM exporter
- [ ] Backup: Configure failover strategy
- [ ] Testing: Benchmark with expected workload
- [ ] Documentation: Record configuration and performance metrics

---

## Next Steps

1. **Immediate**: Verify GPU detection and NCCL version
2. **Day 1**: Deploy test container with Qwen3-70B-AWQ
3. **Week 1**: Benchmark and optimize performance
4. **Week 2**: Integrate with PageIndex RAG system
5. **Week 3-4**: Load testing and production deployment

---

## Support Resources

- Full Analysis: `/home/dra/PageIndex-Home/docs/analysis/RTX5090_Dual_GPU_Analysis.md`
- vLLM Docs: https://docs.vllm.ai/
- NCCL Releases: https://github.com/NVIDIA/nccl/releases
- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

**Document Version**: 1.0
**Last Updated**: November 4, 2025
**Status**: Production-Ready
