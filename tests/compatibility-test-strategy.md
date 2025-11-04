# Compatibility Test Strategy: vLLM + Qwen3 + PageIndex

**Test Session ID**: swarm-1762209620591-1tvm00j4g
**Agent**: Tester
**Date**: 2025-11-04
**Environment**: Docker 28.4.0, NVIDIA 580.65.06, CUDA 13.0

---

## 1. COMPATIBILITY TEST STRATEGY

### 1.1 System Requirements Validation

**GPU Infrastructure Tests**:
- ✓ Docker Runtime: NVIDIA runtime detected and set as default
- ✓ NVIDIA Driver: 580.65.06 (CUDA 13.0 compatible)
- ⚠️ Docker GPU Runtime: `nvidia` available (needs functional test)

**Software Stack Compatibility Matrix**:
```
Component         | Version Required | Test Priority | Risk Level
------------------|------------------|---------------|------------
vLLM              | 0.6.0+          | CRITICAL      | MEDIUM
Qwen3             | Latest          | CRITICAL      | HIGH
PyTorch           | 2.1.0+          | HIGH          | LOW
CUDA              | 12.1+ (13.0✓)   | CRITICAL      | LOW
Docker            | 24.0+  (28.4✓)  | HIGH          | LOW
Python            | 3.10-3.11       | MEDIUM        | LOW
```

### 1.2 Three-Phase Testing Approach

**Phase 1: Isolated Component Testing** (Days 1-2)
- Test vLLM installation independently
- Test Qwen3 model download and loading
- Test Docker GPU passthrough
- Baseline performance benchmarks

**Phase 2: Integration Testing** (Days 3-4)
- vLLM + Qwen3 in Docker container
- PageIndex API integration
- GPU memory management under load
- Multi-request concurrency

**Phase 3: System Validation** (Days 5-6)
- End-to-end workflow testing
- Stress testing and fault injection
- Performance benchmarking
- Documentation validation

---

## 2. VALIDATION CHECKLIST

### 2.1 GPU Detection & Allocation

```bash
# Test 1: GPU Visibility in Container
□ docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
  Expected: GPU list with utilization metrics
  Risk: Container cannot access GPU (HIGH)

# Test 2: Multi-GPU Detection
□ docker run --gpus all ubuntu:22.04 nvidia-smi --list-gpus
  Expected: All available GPUs listed
  Risk: Partial GPU allocation (MEDIUM)

# Test 3: GPU Memory Allocation
□ docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 ubuntu:22.04 nvidia-smi
  Expected: Single GPU allocated, memory available
  Risk: Memory allocation failure (HIGH)

# Test 4: CUDA Runtime Version
□ docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvcc --version
  Expected: CUDA 12.1+ or 13.0
  Risk: Version mismatch (MEDIUM)
```

### 2.2 Model Loading & Inference

```python
# Test 5: vLLM Installation Verification
□ pip install vllm && python -c "import vllm; print(vllm.__version__)"
  Expected: vLLM version >= 0.6.0
  Risk: Import errors, dependency conflicts (HIGH)

# Test 6: Qwen3 Model Download
□ from huggingface_hub import snapshot_download
  snapshot_download("Qwen/Qwen3-7B")
  Expected: Model files downloaded (7-14GB)
  Risk: Network timeout, disk space (MEDIUM)

# Test 7: Model Loading in vLLM
□ from vllm import LLM
  llm = LLM(model="Qwen/Qwen3-7B", gpu_memory_utilization=0.9)
  Expected: Model loads, GPU memory allocated
  Risk: OOM error, slow loading (HIGH)

# Test 8: Basic Inference
□ outputs = llm.generate(["Hello, how are you?"])
  Expected: Generated text response in <2 seconds
  Risk: Inference failure, timeout (CRITICAL)

# Test 9: Batch Inference
□ prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
  outputs = llm.generate(prompts)
  Expected: All responses generated
  Risk: Batch processing errors (MEDIUM)

# Test 10: Streaming Output
□ for output in llm.stream("Tell me a story"):
      print(output)
  Expected: Token-by-token streaming
  Risk: Stream interruption (LOW)
```

### 2.3 API Endpoint Functionality

```bash
# Test 11: Health Check Endpoint
□ curl http://localhost:8000/health
  Expected: {"status": "ok", "model": "Qwen3-7B"}
  Risk: Service not responding (CRITICAL)

# Test 12: Generation Endpoint
□ curl -X POST http://localhost:8000/v1/completions \
    -d '{"prompt": "Test", "max_tokens": 50}'
  Expected: JSON response with generated text
  Risk: API errors, timeout (HIGH)

# Test 13: PageIndex Integration
□ Test PageIndex search → LLM summarization workflow
  Expected: Search results processed by Qwen3
  Risk: Integration failures (HIGH)

# Test 14: Concurrent Requests
□ ab -n 100 -c 10 http://localhost:8000/v1/completions
  Expected: All requests succeed, avg latency <5s
  Risk: Connection timeouts, rate limiting (MEDIUM)

# Test 15: Large Input Handling
□ Send 4096-token prompt to API
  Expected: Successful processing or graceful error
  Risk: Memory overflow, truncation errors (HIGH)
```

### 2.4 Performance Benchmarks

```python
# Test 16: Token Throughput
□ Measure tokens/second for various batch sizes
  Target: >50 tokens/sec for single request
  Risk: Suboptimal performance (MEDIUM)

# Test 17: GPU Memory Utilization
□ Monitor GPU memory during peak load
  Target: 80-90% utilization without OOM
  Risk: Memory leaks, inefficient allocation (HIGH)

# Test 18: Latency Percentiles
□ Measure p50, p95, p99 latency for 1000 requests
  Target: p95 < 5s, p99 < 10s
  Risk: High tail latency (MEDIUM)

# Test 19: Cold Start Time
□ Measure time from container start to first inference
  Target: <60 seconds
  Risk: Slow initialization (LOW)

# Test 20: Multi-User Load
□ Simulate 50 concurrent users for 10 minutes
  Target: No errors, stable throughput
  Risk: Resource exhaustion (HIGH)
```

---

## 3. POTENTIAL FAILURE POINTS & RISK ASSESSMENT

### 3.1 GPU Memory Overflow Scenarios

**Risk Level: HIGH**

| Scenario | Trigger | Impact | Mitigation |
|----------|---------|--------|------------|
| **OOM on Model Loading** | Qwen3-7B requires 14GB VRAM | Container crash | Use `--gpu-memory-utilization=0.85` |
| **OOM During Batch Inference** | Large batch size + long sequences | Request failures | Implement batch size limits (4-8) |
| **Memory Fragmentation** | Long-running container | Gradual degradation | Periodic container restart |
| **Multi-Container Conflict** | Multiple vLLM instances | GPU memory contention | Use CUDA_VISIBLE_DEVICES |

**Test Cases**:
```bash
# TC-GPU-001: Force OOM Condition
docker run --gpus all --memory=8g vllm-qwen3 \
  python -c "from vllm import LLM; LLM('Qwen/Qwen3-70B')"
Expected: Graceful OOM error, not container crash

# TC-GPU-002: Memory Leak Detection
while true; do
  docker exec vllm-container nvidia-smi --query-gpu=memory.used \
    --format=csv,noheader >> memory.log
  sleep 5
done
Expected: Stable memory usage over 1 hour

# TC-GPU-003: Concurrent Model Loading
docker compose up -d vllm-qwen3-1 vllm-qwen3-2
Expected: Both containers load successfully with memory partitioning
```

### 3.2 Docker Networking Issues

**Risk Level: MEDIUM**

| Issue | Symptom | Detection | Resolution |
|-------|---------|-----------|------------|
| **Port Conflict** | "Address already in use" | Container fails to start | Use dynamic port mapping |
| **Network Isolation** | PageIndex cannot reach vLLM | Timeout errors | Check docker network config |
| **DNS Resolution** | Container name not resolving | Connection refused | Use explicit service names |
| **Firewall Blocking** | External access fails | Curl timeout | Configure iptables rules |

**Test Cases**:
```bash
# TC-NET-001: Port Availability
docker run -p 8000:8000 vllm-qwen3
docker run -p 8000:8000 vllm-qwen3  # Should fail
Expected: Second container fails with port conflict error

# TC-NET-002: Inter-Container Communication
docker network create pageindex-net
docker run --network pageindex-net --name vllm vllm-qwen3
docker run --network pageindex-net curlimages/curl curl http://vllm:8000/health
Expected: Successful HTTP response

# TC-NET-003: External Access
docker run -p 8000:8000 vllm-qwen3
curl http://localhost:8000/health
curl http://$(hostname -I | awk '{print $1}'):8000/health
Expected: Both local and network access succeed
```

### 3.3 Model Compatibility Problems

**Risk Level: HIGH**

| Problem | Cause | Symptoms | Fix |
|---------|-------|----------|-----|
| **Tokenizer Mismatch** | vLLM version incompatible | Encoding errors | Pin vLLM version |
| **Missing Model Files** | Incomplete download | FileNotFoundError | Verify checksums |
| **Config Incompatibility** | Qwen3 config not supported | ValueError on load | Update vLLM |
| **Flash Attention Error** | CUDA compute capability | ImportError | Fallback to standard attention |

**Test Cases**:
```python
# TC-MODEL-001: Tokenizer Validation
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-7B")
test_text = "Hello, 世界! 123 @#$%"
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)
assert decoded == test_text, "Tokenizer round-trip failed"

# TC-MODEL-002: Model File Integrity
import hashlib
model_files = ["model.safetensors", "config.json", "tokenizer.json"]
for file in model_files:
    with open(f"/models/Qwen3-7B/{file}", "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()
        # Verify against known good checksums

# TC-MODEL-003: Config Loading
from transformers import AutoConfig
config = AutoConfig.from_pretrained("Qwen/Qwen3-7B")
assert config.architectures[0] == "Qwen3ForCausalLM"
assert config.vocab_size > 0
```

### 3.4 PageIndex Integration Risks

**Risk Level: MEDIUM**

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|---------------------|
| **API Contract Mismatch** | Breaking changes | 40% | Version pinning, integration tests |
| **Timeout on Large Docs** | User experience degradation | 60% | Implement streaming, chunking |
| **Encoding Issues** | Garbled text | 30% | UTF-8 validation, error handling |
| **Rate Limiting** | Service degradation | 50% | Implement request queue |

---

## 4. TESTING RECOMMENDATIONS

### 4.1 Immediate Actions (Priority: CRITICAL)

1. **Create GPU Test Container**
   ```dockerfile
   FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
   RUN apt-get update && apt-get install -y python3.11 python3-pip
   RUN pip install vllm torch torchvision
   COPY test_gpu.py /app/
   CMD ["python3", "/app/test_gpu.py"]
   ```

2. **Implement Health Check Script**
   ```bash
   #!/bin/bash
   # tests/health_check.sh

   echo "1. Checking Docker GPU runtime..."
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi || exit 1

   echo "2. Checking vLLM installation..."
   docker run --rm vllm-qwen3:test python -c "import vllm; print(vllm.__version__)" || exit 1

   echo "3. Checking model availability..."
   docker run --rm vllm-qwen3:test ls -lh /models/Qwen3-7B/ || exit 1

   echo "4. Starting vLLM service..."
   docker run -d --name vllm-test --gpus all -p 8000:8000 vllm-qwen3:test
   sleep 30

   echo "5. Testing API endpoint..."
   curl -f http://localhost:8000/health || exit 1

   echo "6. Testing inference..."
   curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 10}' || exit 1

   docker stop vllm-test && docker rm vllm-test
   echo "✓ All health checks passed"
   ```

3. **Set Up Monitoring**
   ```yaml
   # config/prometheus.yml
   scrape_configs:
     - job_name: 'vllm'
       scrape_interval: 5s
       static_configs:
         - targets: ['vllm:8000']

     - job_name: 'gpu'
       scrape_interval: 10s
       static_configs:
         - targets: ['dcgm-exporter:9400']
   ```

### 4.2 Short-Term (Week 1)

4. **Automated Test Suite**
   - Create pytest framework with 50+ test cases
   - Implement CI/CD pipeline with GitHub Actions
   - Set up test coverage reporting (target: 80%+)

5. **Load Testing Infrastructure**
   - Deploy Locust for load testing
   - Configure realistic user scenarios
   - Establish performance baselines

6. **Failure Injection Testing**
   - Chaos engineering with Toxiproxy
   - Network latency simulation
   - GPU failure scenarios

### 4.3 Long-Term (Month 1)

7. **Production Readiness**
   - Security audit (OWASP top 10)
   - Disaster recovery testing
   - Scalability testing (100+ concurrent users)

8. **Documentation & Training**
   - Create runbook for common issues
   - Document troubleshooting procedures
   - Train team on testing protocols

---

## 5. TESTING TOOLS & FRAMEWORKS

### 5.1 Required Tools

```bash
# Install testing dependencies
pip install pytest pytest-asyncio pytest-cov locust
pip install docker-py requests prometheus-client
pip install nvidia-ml-py3  # For GPU monitoring
```

### 5.2 Test Execution Command Reference

```bash
# Unit tests
pytest tests/unit/ -v --cov=src --cov-report=html

# Integration tests
pytest tests/integration/ -v -s --log-cli-level=INFO

# Performance tests
locust -f tests/load/locustfile.py --headless -u 50 -r 10 -t 5m

# GPU monitoring during tests
nvidia-smi dmon -s pucvmet -d 5 > gpu_metrics.log &
pytest tests/ && kill %1

# Docker-compose test environment
docker compose -f tests/docker compose.test.yml up --abort-on-container-exit
```

---

## 6. SUCCESS CRITERIA

### 6.1 Minimum Viable Tests (MVP)

- [ ] All 20 checklist items pass
- [ ] Zero CRITICAL failures
- [ ] <2 HIGH-risk failures (with workarounds)
- [ ] Documentation complete
- [ ] Team sign-off

### 6.2 Production Ready Tests

- [ ] 90%+ test coverage
- [ ] All load tests pass at 2x expected capacity
- [ ] Mean Time To Recovery (MTTR) < 15 minutes
- [ ] Zero data loss scenarios
- [ ] Security scan passed

---

## 7. NEXT STEPS

**Immediate (Today)**:
1. Wait for researcher agent's technical findings
2. Wait for coder agent's Dockerfile/config specs
3. Integrate findings into executable test plan

**Tomorrow**:
1. Create test Docker environment
2. Execute Phase 1 tests (GPU detection)
3. Report findings to hive coordination

**This Week**:
1. Complete all 20 validation tests
2. Develop automated test suite
3. Present comprehensive test report

---

## 8. COORDINATION NOTES

**Memory Keys for Hive Coordination**:
- `hive/testing/compatibility` - This test strategy
- `hive/testing/validation` - Test execution results
- `hive/testing/failures` - Identified issues
- `hive/testing/recommendations` - Action items

**Dependencies**:
- Waiting for: Researcher's vLLM compatibility report
- Waiting for: Coder's Docker configuration
- Blocking: Integration test execution

**Contact**:
- Agent: Tester
- Session: swarm-1762209620591-1tvm00j4g
- Status: Strategy complete, awaiting inputs

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04 06:41 UTC
**Approval Status**: Pending review by coordinator
