# RunPod Rigor Recovery Guide

This guide ensures a clean-room execution of the SecureCLI-Tuner V2 pipeline to replace previous simulated metrics with real, measured data.

## 1. Environment Setup

Connect to your RunPod instance (A100 recommended) and ensure `shellcheck` is installed:

```bash
apt-get update && apt-get install -y shellcheck
pip install -r requirements.txt
```

## 2. Scientific Reset

Before re-running, clear any old checkpoints or processed data to avoid cross-contamination:

```bash
rm -rf model/checkpoints/*
rm -rf data/processed/*
```

## 3. Execution Path

Follow these steps in order. Each command is critical for data provenance.

### Step A: Data Preparation (Real Test Set)

This will download the full `dharma-1` dataset, run `shellcheck` validation, filter 17+ dangerous patterns, and generate a reproducible 80/10/10 split.

```bash
python data_pipeline/preprocess_data.py
```

*Note: Verify that `data/processed/test.jsonl` exists and has ~125-150 real examples.*

### Step B: The Training Run

Launch the Axolotl QLoRA training. 500 steps is sufficient for convergence.

```bash
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

### Step D: Security Evaluation (Final Boss)

Validate the model against the 183-case adversarial suite to ensure Layer 1-3 coverage.

```bash
python -m pytest tests/eval/test_adversarial.py -v
```

*Note: For RT Certification, the pass rate must exceed 95%.*

## 4. Verification Check

After the evaluation completes:

1. **Check W&B**: Confirm `exact_match_rate` is populated (likely 15-25%).
2. **Check Logs**: Ensure `evaluation/semantic/results.jsonl` contains 100+ entries with non-empty `generated` fields.
3. **Adversarial Pass**: Confirm `test_adversarial.py` shows >174/183 cases blocked.

---
**Next Step:** Once these results are in, paste the output metrics here and I will update the final RT Certification documents for you! 🏁🚀
