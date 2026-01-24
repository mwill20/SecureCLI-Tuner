# Data Provenance

## Overview

This document provides a complete audit trail for the SecureCLI-Tuner V2 training data, from source to final splits.

---

## Source Dataset

| Property | Value |
|----------|-------|
| **Dataset ID** | prabhanshubhowal/natural_language_to_linux |
| **Source URL** | <https://huggingface.co/datasets/prabhanshubhowal/natural_language_to_linux> |
| **Collection Date** | 2026-01-24T06:10:32+00:00 |
| **Base Model** | Qwen/Qwen2.5-Coder-7B-Instruct |

---

## Pipeline Statistics

| Stage | Count | Removed | Reason |
|-------|-------|---------|--------|
| Raw Download | 18,357 | — | — |
| After Dedup | 12,741 | 5,616 | SHA256 duplicate removal |
| After Schema | 12,736 | 5 | Pydantic validation failure |
| After Dangerous | 12,641 | **95** | Zero-tolerance pattern match |
| After Shellcheck | 12,259 | 382 | Syntax validation failure |

---

## Final Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 9,807 | 80% |
| Validation | 1,225 | 10% |
| Test | 1,227 | 10% |
| **Total** | 12,259 | 100% |

**Split Seed:** 42 (deterministic reproducibility)

---

## Dangerous Commands Removed

**Total Removed:** 95 commands

These commands matched one or more of the 17 zero-tolerance patterns:

- `rm -rf /` (root deletion)
- Fork bombs
- Disk wipes (`dd if=/dev/zero`)
- Permission bombs (`chmod 777 /`)
- Remote execution (`curl | bash`)
- And 12 additional patterns

All removed commands are logged in `data/logs/removed_dangerous.jsonl` for audit purposes.

---

## Shellcheck Validation

| Metric | Value |
|--------|-------|
| Commands Validated | 12,641 |
| Passed | 12,259 |
| Failed | 382 |
| Pass Rate | 97.0% |

Failed commands had Bash syntax errors and were excluded from training.

---

## Reproducibility

To reproduce this exact dataset:

```bash
# Clone repository
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

# Run data pipeline
python scripts/download_datasets.py
python scripts/prepare_training_data.py
```

**Expected Output:**

- `data/processed/train.jsonl` (9,807 examples)
- `data/processed/val.jsonl` (1,225 examples)
- `data/processed/test.jsonl` (1,227 examples)
- `data/processed/provenance.json` (this audit trail)

---

## Verification Checklist

- [x] Source dataset downloaded from HuggingFace
- [x] Deduplication applied (SHA256)
- [x] Schema validation passed
- [x] Dangerous commands filtered (95 removed)
- [x] Shellcheck validation passed (97%)
- [x] Splits created with seed 42
- [x] Provenance recorded
