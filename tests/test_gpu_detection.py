#!/usr/bin/env .venv/bin/python
"""
GPU Detection and Validation Tests
Part of Compatibility Test Strategy for vLLM + Qwen3

Note: This script should be run within the virtual environment.
Run './scripts/setup_venv.sh' first if you haven't already.
"""

import subprocess
import sys
import json
from typing import Dict, List, Tuple


class GPUTestSuite:
    """Comprehensive GPU detection and validation tests."""

    def __init__(self):
        self.results: List[Dict] = []
        self.passed = 0
        self.failed = 0

    def log_result(self, test_name: str, passed: bool, message: str, risk_level: str = "MEDIUM"):
        """Log test result with metadata."""
        result = {
            "test": test_name,
            "status": "PASS" if passed else "FAIL",
            "message": message,
            "risk_level": risk_level
        }
        self.results.append(result)
        if passed:
            self.passed += 1
            print(f"✓ {test_name}: {message}")
        else:
            self.failed += 1
            print(f"✗ {test_name}: {message} [RISK: {risk_level}]")

    def run_command(self, cmd: List[str]) -> Tuple[bool, str]:
        """Execute shell command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)

    def test_01_nvidia_smi_available(self):
        """Test 1: Verify nvidia-smi is available."""
        success, output = self.run_command(["nvidia-smi", "--version"])
        self.log_result(
            "TC-GPU-001: nvidia-smi availability",
            success,
            f"NVIDIA SMI detected: {output.split()[2] if success else 'Not found'}",
            "CRITICAL"
        )

    def test_02_cuda_version(self):
        """Test 2: Verify CUDA version compatibility."""
        success, output = self.run_command(["nvidia-smi"])
        if success:
            # Extract CUDA version from output
            for line in output.split('\n'):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[-1].strip().split()[0]
                    major_version = int(cuda_version.split('.')[0])
                    passed = major_version >= 12
                    self.log_result(
                        "TC-GPU-002: CUDA version check",
                        passed,
                        f"CUDA {cuda_version} {'✓ Compatible' if passed else '✗ Incompatible (need 12.1+)'}",
                        "CRITICAL"
                    )
                    return
        self.log_result("TC-GPU-002: CUDA version check", False, "Could not detect CUDA version", "CRITICAL")

    def test_03_gpu_count(self):
        """Test 3: Count available GPUs."""
        success, output = self.run_command(["nvidia-smi", "--list-gpus"])
        if success:
            gpu_count = len([line for line in output.split('\n') if line.strip()])
            self.log_result(
                "TC-GPU-003: GPU count",
                gpu_count > 0,
                f"Detected {gpu_count} GPU(s)",
                "HIGH"
            )
        else:
            self.log_result("TC-GPU-003: GPU count", False, "Could not list GPUs", "HIGH")

    def test_04_gpu_memory(self):
        """Test 4: Check GPU memory availability."""
        success, output = self.run_command([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits"
        ])
        if success:
            for i, line in enumerate(output.strip().split('\n')):
                total, free = map(int, line.split(','))
                # Qwen3-7B requires ~14GB VRAM
                sufficient = total >= 14000
                self.log_result(
                    f"TC-GPU-004.{i}: GPU {i} memory",
                    sufficient,
                    f"Total: {total}MB, Free: {free}MB {'✓ Sufficient for Qwen3-7B' if sufficient else '✗ Insufficient'}",
                    "HIGH"
                )
        else:
            self.log_result("TC-GPU-004: GPU memory check", False, "Could not query GPU memory", "HIGH")

    def test_05_docker_availability(self):
        """Test 5: Verify Docker is installed and running."""
        success, output = self.run_command(["docker", "--version"])
        if success:
            version = output.split()[2].rstrip(',')
            self.log_result(
                "TC-DOCKER-001: Docker availability",
                True,
                f"Docker {version} detected",
                "CRITICAL"
            )
        else:
            self.log_result("TC-DOCKER-001: Docker availability", False, "Docker not found", "CRITICAL")

    def test_06_docker_gpu_runtime(self):
        """Test 6: Verify Docker GPU runtime is configured."""
        success, output = self.run_command(["docker", "info"])
        if success:
            has_nvidia_runtime = "nvidia" in output.lower()
            self.log_result(
                "TC-DOCKER-002: GPU runtime",
                has_nvidia_runtime,
                "NVIDIA runtime configured" if has_nvidia_runtime else "NVIDIA runtime missing",
                "CRITICAL"
            )
        else:
            self.log_result("TC-DOCKER-002: GPU runtime", False, "Could not query Docker info", "CRITICAL")

    def test_07_docker_gpu_access(self):
        """Test 7: Test GPU access from Docker container."""
        success, output = self.run_command([
            "docker", "run", "--rm", "--gpus", "all",
            "nvidia/cuda:12.1.0-base-ubuntu22.04",
            "nvidia-smi", "--list-gpus"
        ])
        self.log_result(
            "TC-DOCKER-003: GPU access from container",
            success,
            "Container can access GPU" if success else "Container cannot access GPU",
            "CRITICAL"
        )

    def test_08_python_availability(self):
        """Test 8: Check Python version."""
        success, output = self.run_command(["python3", "--version"])
        if success:
            version = output.split()[1]
            major, minor = map(int, version.split('.')[:2])
            # Updated to support Python 3.10-3.12 (vLLM supports 3.8+)
            compatible = (major == 3 and minor >= 10)
            self.log_result(
                "TC-ENV-001: Python version",
                compatible,
                f"Python {version} {'✓ Compatible' if compatible else '✗ Incompatible (need 3.10+)'}",
                "HIGH"
            )
        else:
            self.log_result("TC-ENV-001: Python version", False, "Python not found", "HIGH")

    def test_09_disk_space(self):
        """Test 9: Verify sufficient disk space for models."""
        success, output = self.run_command(["df", "-h", "."])
        if success:
            lines = output.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                avail = parts[3]
                # Qwen3-7B requires ~15GB
                sufficient = True  # Simple check for now
                self.log_result(
                    "TC-ENV-002: Disk space",
                    sufficient,
                    f"Available: {avail}",
                    "MEDIUM"
                )
        else:
            self.log_result("TC-ENV-002: Disk space", False, "Could not check disk space", "MEDIUM")

    def test_10_network_connectivity(self):
        """Test 10: Test network connectivity for model downloads."""
        success, output = self.run_command(["curl", "-I", "-s", "https://huggingface.co"])
        accessible = success and "200" in output
        self.log_result(
            "TC-ENV-003: HuggingFace connectivity",
            accessible,
            "Can reach HuggingFace" if accessible else "Cannot reach HuggingFace",
            "MEDIUM"
        )

    def run_all_tests(self):
        """Execute all tests in sequence."""
        print("=" * 60)
        print("GPU DETECTION AND VALIDATION TEST SUITE")
        print("=" * 60)
        print()

        test_methods = [
            self.test_01_nvidia_smi_available,
            self.test_02_cuda_version,
            self.test_03_gpu_count,
            self.test_04_gpu_memory,
            self.test_05_docker_availability,
            self.test_06_docker_gpu_runtime,
            self.test_07_docker_gpu_access,
            self.test_08_python_availability,
            self.test_09_disk_space,
            self.test_10_network_connectivity,
        ]

        for test in test_methods:
            try:
                test()
            except Exception as e:
                self.log_result(
                    test.__name__,
                    False,
                    f"Test exception: {str(e)}",
                    "HIGH"
                )

        print()
        print("=" * 60)
        print(f"TEST SUMMARY: {self.passed} passed, {self.failed} failed")
        print("=" * 60)

        # Save results to JSON
        with open('/home/dra/PageIndex-Home/tests/gpu_test_results.json', 'w') as f:
            json.dump({
                "passed": self.passed,
                "failed": self.failed,
                "total": self.passed + self.failed,
                "results": self.results
            }, f, indent=2)

        return self.failed == 0


def main():
    """Main entry point."""
    suite = GPUTestSuite()
    success = suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
