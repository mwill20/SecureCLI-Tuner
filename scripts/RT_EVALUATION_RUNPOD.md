# RT Certification Evaluation - RunPod Instructions

## Quick Start Commands

Run these commands on RunPod to complete the RT certification requirements.

### 1. Setup (if not already done)

```bash
cd /workspace
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

pip install transformers peft torch accelerate bitsandbytes datasets tqdm
```

### 2. Baseline Evaluation (~30 min)

Evaluate the base Qwen model BEFORE fine-tuning on your test set:

```bash
# Full test set (1,227 examples) - ~30 min
python scripts/evaluate_baseline.py --test-file data/processed/test.jsonl

# Or quick test with 50 examples (~5 min)
python scripts/evaluate_baseline.py --test-file data/processed/test.jsonl --max-examples 50
```

**Output:** `evaluation/results/baseline_metrics.json`

---

### 3. MMLU Benchmark (~45 min for both models)

Check for catastrophic forgetting by comparing base vs fine-tuned on MMLU:

```bash
# Full MMLU subset (100 questions, both models) - ~45 min
python scripts/evaluate_mmlu.py --both --num-questions 100

# Quick test (20 questions) - ~10 min
python scripts/evaluate_mmlu.py --both --num-questions 20
```

**Output:** `evaluation/results/mmlu_results.json`

---

### 4. Copy Results to Local Machine

After running, copy the results:

```bash
# View results
cat evaluation/results/baseline_metrics.json
cat evaluation/results/mmlu_results.json
```

Then copy the JSON files to your local `evaluation/results/` directory.

---

## Expected Results

### Baseline Evaluation

| Model | Exact Match | Command-Only |
|-------|-------------|--------------|
| Base Qwen-7B (pre-finetune) | ~5-15% | ~60-80% |
| SecureCLI-Tuner V2 (post) | 9.1% | 99.0% |

### MMLU Benchmark

| Model | MMLU Accuracy | Status |
|-------|---------------|--------|
| Base Qwen-7B | ~60-70% | Reference |
| SecureCLI-Tuner V2 | ~55-70% | Should be within 5% of base |

If fine-tuned accuracy is within 5% of base, no catastrophic forgetting occurred.

---

## Time Estimates

| Task | Quick Test | Full Run |
|------|------------|----------|
| Baseline Evaluation | ~5 min | ~30 min |
| MMLU Both Models | ~10 min | ~45 min |
| **Total** | **~15 min** | **~75 min** |
