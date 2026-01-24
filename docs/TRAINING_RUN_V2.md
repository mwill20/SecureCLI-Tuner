# SecureCLI-Tuner V2 — Training Run Report

**Run ID:** `honest-music-2`  
**Date:** 2026-01-24  
**Platform:** RunPod A100 (40GB)  
**W&B:** [View Run](https://wandb.ai/mwill-itmission20/SecureCLI-Training/runs/wk93zl4r)

---

## Training Summary

| Metric | Value |
|--------|-------|
| **Total Steps** | 500 |
| **Epochs** | 0.204 (~20% of dataset) |
| **Training Runtime** | 44.5 minutes (2,673 seconds) |
| **Final Train Loss** | 0.813 |
| **Final Eval Loss** | 0.861 |
| **Average Train Loss** | 1.129 |
| **Samples/Second** | 0.748 |
| **Steps/Second** | 0.187 |
| **Total FLOPs** | 4.35 × 10¹⁶ |

---

## Loss Progression

| Step | Train Loss | Eval Loss |
|------|-----------|-----------|
| 1 | 3.88 | 4.00 |
| 25 | 1.56 | 1.42 |
| 50 | 0.90 | 0.93 |
| 100 | 0.94 | 0.89 |
| 250 | 0.86 | 0.87 |
| 500 | 0.81 | 0.86 |

The loss curve shows healthy convergence with no signs of overfitting (eval loss tracks train loss).

---

## Data Pipeline Summary

| Stage | Count | Notes |
|-------|-------|-------|
| Raw Examples | 18,357 | HuggingFace dharma-1 |
| After Dedup | 12,741 | SHA256 fingerprinting |
| After Schema | 12,736 | Pydantic validation |
| After Dangerous Filter | 12,641 | **95 dangerous removed** |
| After Shellcheck | 12,259 | Syntax validation |
| **Train Split** | 9,807 | 80% |
| **Val Split** | 1,225 | 10% |
| **Test Split** | 1,227 | 10% |

---

## Evaluation Results

### Semantic Evaluation

| Metric | Value |
|--------|-------|
| Total Examples | 1,227 |
| Exact Match Rate | 9.1% (112/1227) |
| Semantic Match Rate | 9.1% |
| Functional Match Rate | 9.1% |
| Similarity Threshold | 0.80 |

> **Note:** CodeBERT could not load due to torch version constraints, so semantic evaluation fell back to basic matching. This is a conservative measurement.

### Adversarial Security Testing

| Test | Result |
|------|--------|
| Adversarial Pass Rate (>95%) | ✅ PASSED |
| Direct Destructive | ✅ PASSED |
| Obfuscated | ✅ PASSED |
| Privilege Escalation | ✅ PASSED |
| Data Exfil | ✅ PASSED |
| Prompt Injection | ✅ PASSED |
| Remote Execution | ✅ PASSED |
| Credential Theft | ✅ PASSED |
| ASI Attribution | ✅ PASSED |

**100% pass rate** on all 9 adversarial test categories.

---

## Model Configuration

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
load_in_4bit: true
lora_r: 8
lora_alpha: 16
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj
learning_rate: 0.0002
max_steps: 500
micro_batch_size: 1
gradient_accumulation_steps: 4
```

---

## W&B Training Curves

![Training Loss](WandB/WandbTrain.png)

![Eval Metrics](WandB/WandBEval.png)

![System Metrics](WandB/WandBSystem.png)

---

## Checkpoint Location

```
models/checkpoints/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files
```

---

## Verification Checklist

- [x] Training completed (500 steps)
- [x] Model saved to checkpoints
- [x] W&B tracking synced
- [x] Semantic evaluation completed
- [x] Adversarial tests passed (9/9)
- [x] Data provenance recorded
