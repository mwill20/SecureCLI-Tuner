# Lesson 1: Security-First Data Pipelines

## 1. Introduction

### Learning Objectives

By the end of this lesson, you will be able to:

- Explain how training data is normalized and filtered for security
- Run the preprocessing pipeline and interpret progress output
- Validate Bash syntax with Shellcheck and understand failure handling
- Apply Qwen chat templates and verify role tokens
- Mask user tokens so the model learns only assistant responses
- Enforce zero-tolerance dangerous command filtering
- Read provenance logs to verify data quality and traceability

### Plain-English Explanation

This phase turns raw NL-to-Bash pairs into **safe, validated training data**. Think of it like airport security for commands: every record is screened before it's allowed into training.

### Why This Matters

Without this step, the model could learn to generate catastrophic commands like `rm -rf /` from benign prompts. Phase 1 blocks those risks **before training starts**.

---

## 2. Key Concepts

### Domain Terminology

| Term | Definition |
|------|------------|
| **Chat Template** | Structured prompt format used by Qwen2.5-Coder |
| **Zero-Tolerance Patterns** | Regex rules that block catastrophic commands |
| **Shellcheck Validation** | Syntax validation for Bash commands |
| **Provenance Tracking** | Audit trail for what was filtered and why |

### Design Decisions

1. **Dangerous pattern filtering happens before training**, not just at inference
2. **Shellcheck catches syntax errors; safety checks catch malicious intent** — both are required
3. **Every filtered record is logged** for auditability and repeatability
4. **Sampling is random with a fixed seed** so test runs are reproducible

---

## 3. Data Flow

```
HuggingFace Dataset (18,357 examples)
  ↓
[1] Load + Field Mapping (nl_command → instruction, bash_code → output)
  ↓ [Random Sample: 1,835 examples, seed=42]
[2] Shellcheck Syntax Validation (1,793 passed, 42 failed)
  ↓
[3] Dangerous Pattern Filtering (0 dangerous commands found)
  ↓
[4] Qwen Chat Template (<|im_start|>system/user/assistant<|im_end|>)
  ↓
[5] Tokenize + Assistant-Only Masking (user tokens = -100)
  ↓
[6] Deduplicate (58 removed) + Split (80/10/10, seed=42)
  ↓
[7] Save Outputs + Provenance
  ↓
data/processed/
├── train.jsonl (1,388 examples)
├── val.jsonl (173 examples)
├── test.jsonl (174 examples)
└── provenance.json (full audit trail)
```

---

## 4. Zero-Tolerance Patterns

These 17 patterns are **blocked unconditionally**:

```python
ZERO_TOLERANCE_PATTERNS = [
    r"rm\s+-rf\s+/",                    # Root deletion
    r":\(\)\{\s*:\|:\s*&\s*\};:",       # Fork bomb
    r"dd\s+if=/dev/zero\s+of=/dev/sd",  # Disk wipe
    r"chmod\s+-R\s+777\s+/",            # Permission bomb
    r"curl\s+.*\|\s*bash",              # Remote execution
    r"wget\s+.*\|\s*sh",                # Remote execution
    r"mkfs\.",                          # Filesystem format
    # ... 10 more patterns
]
```

---

## 5. Hands-On Exercises

### Exercise 1: Run the Pipeline

```powershell
cd C:\Projects\SecureCLI-Tuner
python data_pipeline/preprocess_data.py
```

**Expected output:**

```
Step 1/7: Loading dataset...
  Sampling 1835 examples from 18357 total (seed=42)
Step 2/7: Running shellcheck...
  Shellcheck complete: 1793/1835 passed (97%)
Step 3/7: Filtering dangerous commands...
Step 4/7: Applying chat template...
Step 5/7: Tokenizing with assistant-only masking...
Step 6/7: Splitting train/val/test...
  Removed 58 duplicate examples
Step 7/7: Saving outputs...
Pipeline complete. 1388 train, 173 val, 174 test.
```

### Exercise 2: Inspect Provenance

```powershell
Get-Content data/processed/provenance.json
```

### Exercise 3: Verify No Dangerous Commands

```powershell
Get-Content data/logs/removed_dangerous.jsonl
# Should be empty (0 dangerous commands in clean dataset)
```

---

## 6. Interview Preparation

### Q: How do you ensure training data doesn't contain dangerous commands?

**Model Answer:** "I use a zero-tolerance filtering step before training. The 17 patterns in `guardrails/patterns.py` cover catastrophic actions like root deletion, disk wipes, and remote execution. Every command is checked with `is_dangerous_command()` and matches are logged and removed. This means the model **never sees dangerous commands during training**, which is safer than only filtering at inference time."

### Q: Why validate with Shellcheck if you're already filtering dangerous patterns?

**Model Answer:** "They solve different problems. Dangerous patterns catch harmful intent. Shellcheck catches invalid Bash syntax. A command might be safe but syntactically broken—I don't want the model to learn invalid syntax. Both checks are required."

### Q: How do you handle the tradeoff between data quality and dataset size?

**Model Answer:** "I track statistics in provenance. For the 10% sample, 42 commands failed Shellcheck and 58 duplicates were removed. The final 1,735 examples are clean. If a future dataset lost too many examples, the logs show exactly why—but safety filtering is non-negotiable."

---

## 7. Key Takeaways

- ✅ Zero-tolerance patterns block catastrophic commands before training
- ✅ Shellcheck validates syntax correctness (safety ≠ correctness)
- ✅ Chat templates preserve the base model's instruction format
- ✅ Masking ensures the model learns to generate commands, not prompts
- ✅ Provenance provides a full audit trail

---

## 8. Next Steps

- Review `data/logs/` to understand filtered records
- Proceed to [Lesson 2: QLoRA Fine-Tuning](Lesson_02_Training.md)
