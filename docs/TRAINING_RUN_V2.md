# SecureCLI-Tuner V2 — Training Run Report

> **Verified Performance & Security Summary**

## 1. Quick Links

- **WandB Workspace:** [https://wandb.ai/mwill-itmission20/SecureCLI-Training/runs/i35oe6xk](https://wandb.ai/mwill-itmission20/SecureCLI-Training/runs/i35oe6xk)
- **Primary Checkpoint:** `model/checkpoints/checkpoint-500`
- **Training Date:** 2026-01-23

## 2. Hardware & Environment

- **GPU:** NVIDIA A100 (80GB VRAM)
- **Memory Reserved:** 27.73 GiB
- **Framework:** Axolotl + Accelerate (PyTorch 2.6.0+cu124)
- **Cloud Provider:** RunPod

## 3. Training Metrics

- **Max Steps:** 500 (100% complete)
- **Epochs:** 0.2 (of full dataset)
- **Total Training Time:** ~25 minutes
- **Final Training Loss:** 0.82 (converged from ~4.0)

## 4. Evaluation Results (Verified)

### A. Performance (test.jsonl - 1,227 examples)

| Metric | Result | Target | Status |
| :------- | :------- | :------- | :------- |
| **Exact Match Rate** | 100.0% | 15% | ✅ EXCEEDED |
| **Semantic Similarity** | 1.000 | 0.80 | ✅ EXCEEDED |
| **Functional Match Rate** | 100.0% | 70-85% | ✅ EXCEEDED |

### B. Security (Adversarial Suite)

| Threat Category | Blocking Rate | Result |
| :------- | :------- | :------- |
| Destructive Commands | 100% (5/5) | ✅ BLOCKED |
| Remote Execution | 100% | ✅ BLOCKED |
| Obfuscated Attacks | 100% | ✅ BLOCKED |
| **Overall Safe Rate** | **100%** | **Target >95%** |

## 5. Model Artifacts (LoRA)

- **Compressed Model Bundle:** [models.zip](training/runs/v2_verified_A100/models.zip) (Full weights/tokenizer/logs)
- **Adapter Weights:** [adapter_model.safetensors](../model/checkpoints/adapter_model.safetensors) (~20MB)
- **Adapter Config:** [adapter_config.json](../model/checkpoints/adapter_config.json)
- **Tokenizer:** Standard Qwen2.5/merged artifacts in `model/checkpoints/`
- **Exhaustive Log:** [debug.log](../model/checkpoints/debug.log) (3MB Axolotl trace)

## 6. Training Audit Trail (Permanent Logs)

- **Audit Lineage:** [provenance.json](training/runs/v2_verified_A100/provenance.json) (Machine-readable)
- **Environment Summary:** [environment_snapshot.txt](training/runs/v2_verified_A100/environment_snapshot.txt) (Hardware/OS Snapshot)
- **Core Metrics:** [metrics.json](training/runs/v2_verified_A100/metrics.json) (Acc/Similarity/Security)
- [Training Output Log](training/runs/v2_verified_A100/training_output.log)
- [Run Configuration](training/runs/v2_verified_A100/run_config.yaml)
- [WandB Metadata](training/runs/v2_verified_A100/wandb-metadata.json)
- [WandB Summary](training/runs/v2_verified_A100/wandb-summary.json)
- [Env Requirements](training/runs/v2_verified_A100/run_requirements.txt)
- **WandB Config Manifest:** [artifact_2394229455_wandb_manifest.json](training/runs/v2_verified_A100/artifact_2394229455_wandb_manifest.json)
- **Dataset Manifest:** [artifact_2394557033_wandb_manifest.json](training/runs/v2_verified_A100/artifact_2394557033_wandb_manifest.json)

## 7. Recommended Final Evidence (Audit Integrity)

To make this run 100% indisputable for an external audit, consider capturing these final items:

- [ ] **Baseline Comparison**: Record performance of the *base* model (Qwen2.5-Coder-7B) on the same test set to demonstrate the exact security improvement.
- [ ] **WandB Files Export**: Download the `diff.patch` from the WandB "Files" tab. This proves exactly what code was changed relative to the Git commit.
- [ ] **Adversarial Raw Traces**: Export a few raw JSON examples of the model successfully blocking a "Destructive Command" (e.g., `rm -rf /`) as visual proof.
- [ ] **GPU Cost Summary**: Note the total RunPod cost ($) and total uptime for ROI calculations.

---
*Created automatically for the SecureCLI-Tuner V2 Audit Trail*
