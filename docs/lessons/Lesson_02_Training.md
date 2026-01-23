# Lesson 2: QLoRA Fine-Tuning

## 1. Introduction

### Learning Objectives

- Configure Axolotl for QLoRA fine-tuning
- Understand 4-bit NF4 quantization for memory efficiency
- Track experiments with Weights & Biases
- Validate checkpoints before deployment

### Why QLoRA?

Full fine-tuning of Qwen2.5-Coder-7B requires ~60GB VRAM. QLoRA reduces this to ~24GB by:

1. **4-bit quantization** — Model weights stored in NF4 format
2. **Low-rank adapters** — Only train small adapter matrices
3. **Gradient checkpointing** — Trade compute for memory

---

## 2. Configuration

### Key Settings (`training/axolotl_config.yaml`)

```yaml
# Base model
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
load_in_4bit: true

# LoRA configuration
lora_r: 8              # Rank (lower = smaller adapter)
lora_alpha: 16         # Scaling factor
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

# Training
micro_batch_size: 1
gradient_accumulation_steps: 4  # Effective batch = 4
max_steps: 500
learning_rate: 0.0002
```

---

## 3. Running Training

```bash
# On RunPod A100 (Linux)
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

### Expected Output

```
Epoch 1/1: 100%|████████████████| 500/500 [45:23<00:00]
Training Loss: 1.0949
Eval Loss: 0.8840
Checkpoint saved: models/checkpoints/phase2-final/
```

---

## 4. W&B Tracking

Key metrics to monitor:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss curve |
| `eval/loss` | Validation loss |
| `gpu_memory` | VRAM usage |

---

## 5. Interview Preparation

### Q: Why use QLoRA instead of full fine-tuning?

**Model Answer:** "QLoRA combines 4-bit quantization with low-rank adapters. The base model stays frozen in 4-bit precision while we train small adapter matrices. This reduces memory from ~60GB to ~24GB, making it possible to fine-tune 7B models on consumer GPUs. The quality loss is minimal—typically within 1% of full fine-tuning."

### Q: How do you choose the LoRA rank?

**Model Answer:** "Rank controls adapter capacity. Higher rank = more parameters = better expressivity but more memory. For domain adaptation like NL-to-Bash, rank 8 is usually sufficient. I'd experiment with rank 16 if I saw underfitting, but for our 1,388 training examples, rank 8 works well."

---

## 6. Key Takeaways

- ✅ QLoRA enables 7B model fine-tuning on 24GB GPUs
- ✅ LoRA rank 8, alpha 16 for domain adaptation
- ✅ Track experiments with W&B SecureCLI-Training project
- ✅ Validate checkpoints before deployment

---

## 7. Next Steps

- Proceed to [Lesson 3: Safety Evaluation](Lesson_03_Evaluation.md)
