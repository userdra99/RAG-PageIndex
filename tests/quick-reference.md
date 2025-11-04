# Quick Reference - Testing Commands

## Immediate Testing Commands

### Run System Health Check
```bash
/home/dra/PageIndex-Home/tests/health_check.sh
```

### Run GPU Detection Tests
```bash
python3 /home/dra/PageIndex-Home/tests/test_gpu_detection.py
```

### View Test Results
```bash
cat /home/dra/PageIndex-Home/tests/gpu_test_results.json
```

## System Information (Validated)

**GPUs**: 2x NVIDIA GeForce RTX 5090 (32GB VRAM each)
**Docker**: 28.4.0 with NVIDIA runtime (default)
**CUDA**: 13.0
**Driver**: 580.65.06
**Python**: 3.12.3
**Disk Space**: 207GB available

## Critical Next Steps

1. **Install vLLM** (after researcher specifies version):
   ```bash
   pip install vllm==<version>
   ```

2. **Download Qwen3 Model**:
   ```bash
   huggingface-cli download Qwen/Qwen3-7B
   ```

3. **Build Docker Image** (after coder provides Dockerfile):
   ```bash
   docker build -t vllm-qwen3:test .
   ```

4. **Run Container**:
   ```bash
   docker run --gpus all -p 8000:8000 vllm-qwen3:test
   ```

5. **Test API**:
   ```bash
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 50}'
   ```

## Test Documentation

- **Full Strategy**: `/home/dra/PageIndex-Home/tests/compatibility-test-strategy.md`
- **Summary**: `/home/dra/PageIndex-Home/tests/TESTING_SUMMARY.md`
- **This Reference**: `/home/dra/PageIndex-Home/tests/quick-reference.md`

## Coordination

**Session**: swarm-1762209620591-1tvm00j4g
**Memory Keys**: `hive/testing/compatibility`, `hive/testing/validation`
**Status**: ✅ Ready for integration after researcher/coder deliver specs
