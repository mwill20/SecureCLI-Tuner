# Deployment Guide: SecureCLI-Tuner V2

This guide details the operationalization of SecureCLI-Tuner V2 for local inference and cloud-scale deployment.

## 1. Local Quickstart (5-Minute Setup)

### Prerequisites

- Python 3.10+
- NVIDIA GPU (vRAM ≥ 12GB for 4-bit inference)
- CUDA 12.x installed

### Installation

```bash
git clone https://github.com/mwill-itmission/SecureCLI-Tuner
cd SecureCLI-Tuner
pip install -r requirements.txt
```

### Inference Snippet

```python
from cli_tuner.generator import CLIGenerator

# Loads the LoRA adapters + Base Model automatically
generator = CLIGenerator(checkpoint="model/checkpoints/checkpoint-500")

response = generator.generate("List all docker containers running on port 80")
print(f"Generated Command: {response.command}")
print(f"Security Status: {response.security_check}")
```

## 2. Cloud Deployment (RunPod / AWS)

SecureCLI-Tuner V2 is optimized for **RunPod** environments.

### Deployment via Docker

A pre-built environment snapshot is provided in `docs/training/runs/v2_verified_A100/run_requirements.txt`.

1. **Start Pod:** Choose an NVIDIA A100 or 4090.
2. **Initialize:**

   ```bash
   bash scripts/setup_runpod.sh
   ```

3. **Serve API:**

   ```bash
   uvicorn deployment.api.main:app --host 0.0.0.0 --port 8000
   ```

## 3. CommandRisk Guardrail Configuration

You can adjust the strictness of the security layers in `configs/security_policy.yaml`:

| Level | Strategy | Recommended For |
| :------- | :------- | :------- |
| **Strict** | Block on any L1 match OR L3 < 0.9 | Critical Production |
| **Balanced** | Block on L1; Warn on L3 < 0.7 | Development |
| **LogOnly** | Pass all; Log anomalies to WandB | Research / Auditing |

## 4. Performance Benchmarks

| Metric | Result (A100) | Result (RTX 4090) |
| :------- | :------- | :------- |
| **TTFT (Time to First Token)** | ~45ms | ~60ms |
| **Throughput** | 120 tokens/sec | 85 tokens/sec |
| **vRAM Footprint (4-bit)** | 6.8 GB | 7.2 GB |

---
*For troubleshooting, refer to `docs/training/runs/v2_verified_A100/training_output.log` for initialization traces.*
