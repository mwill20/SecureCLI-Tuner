# Lesson 5: RunPod Setup & Training Data Preparation

## 1. Introduction

### Learning Objectives

By the end of this lesson, you will be able to:

- Explain the end-to-end data preparation pipeline
- Run the one-command setup on RunPod
- Understand how datasets are downloaded, merged, and filtered
- Describe the deduplication and dangerous command filtering process
- Walk through the provenance tracking system
- Answer interview questions about ML data engineering

### Plain-English Explanation

This lesson covers the automation that transforms raw HuggingFace datasets into clean, safe, training-ready data. One script does everything: downloads data, filters dangerous commands, validates syntax, applies chat templates, and splits into train/val/test.

### Why This Matters

Data preparation is 80% of ML engineering. Automating it ensures:

- **Reproducibility**: Same script, same output every time
- **Safety**: Dangerous commands never reach training
- **Auditability**: Full provenance tracking for compliance

---

## 2. Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    setup_runpod.sh                              │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: apt-get install shellcheck                             │
│  Step 2: pip install -r requirements.txt                        │
│  Step 3: python scripts/download_datasets.py                    │
│  Step 4: python scripts/download_semantic_model.py              │
│  Step 5: python scripts/prepare_training_data.py                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Output: data/processed/                      │
├─────────────────────────────────────────────────────────────────┤
│  train.jsonl    (~14K examples)                                 │
│  val.jsonl      (~1.7K examples)                                │
│  test.jsonl     (~1.7K examples)                                │
│  provenance.json (audit trail)                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
HuggingFace Datasets
       ↓
[1] Download (prabhanshubhowal/natural_language_to_linux + TLDR)
       ↓
[2] Normalize fields (nl_command → instruction, bash_code → output)
       ↓
[3] Deduplicate (SHA256 fingerprinting)
       ↓
[4] Schema validation (Pydantic)
       ↓
[5] Filter dangerous commands (CommandRisk deterministic guardrail)
       ↓
[6] Shellcheck syntax validation
       ↓
[7] Apply Qwen chat template
       ↓
[8] Split 80/10/10 (train/val/test)
       ↓
[9] Save + Provenance
```

---

## 3. Key Scripts

### `scripts/setup_runpod.sh`

The master orchestrator. Runs everything in sequence.

```bash
#!/bin/bash
set -e  # Exit on any error

# Install system dependencies
apt-get update -qq
apt-get install -y -qq shellcheck

# Install Python dependencies
pip install -q -r requirements.txt
pip install -q axolotl accelerate bitsandbytes wandb

# Download and prepare data
python scripts/download_datasets.py
python scripts/download_semantic_model.py
python scripts/prepare_training_data.py
```

**Key design choice**: `set -e` ensures the script stops immediately if any step fails. No silent errors.

---

### `scripts/download_datasets.py`

Downloads datasets from HuggingFace and normalizes to our schema.

**Key functions:**

```python
def download_bash_dataset():
    """Downloads prabhanshubhowal/natural_language_to_linux (~18K examples)"""
    dataset = load_dataset("prabhanshubhowal/natural_language_to_linux")
    # Normalize: nl_command → instruction, bash_code → output
    
def download_tldr_dataset():
    """Downloads cheshire-cat-ai/tldr-pages, filters for Linux/common"""
    # Only keeps platform == "common" or "linux"
```

**Why normalize?** Different datasets use different field names. Normalizing to `instruction`/`output` ensures our pipeline works with any source.

---

### `scripts/prepare_training_data.py`

The core data engineering logic. 7 steps:

**Step 1: Load raw datasets**

```python
def load_raw_datasets() -> List[Dict]:
    """Loads all JSONL files from data/raw/"""
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        # Merge all sources
```

**Step 2: Deduplicate**

```python
def deduplicate(examples: List[Dict]) -> List[Dict]:
    """Remove duplicates using SHA256 fingerprints"""
    def compute_fingerprint(example):
        payload = json.dumps({
            "instruction": example["instruction"],
            "output": example["output"],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
```

**Why fingerprint?** Exact string matching can miss near-duplicates. SHA256 is fast and deterministic.

**Step 3: Schema validation**

```python
def validate_schema(examples: List[Dict]) -> List[Dict]:
    """Validate against Pydantic BashCommandExample"""
    validated = BashCommandExample(
        instruction=example["instruction"],
        output=example["output"]
    )
```

**Step 4: Filter dangerous commands**

```python
def filter_dangerous(examples: List[Dict]) -> List[Dict]:
    """Use CommandRisk deterministic guardrail"""
    guardrail = DeterministicGuardrail()
    
    if guardrail.is_dangerous(example["output"]):
        # Log to removed_dangerous.jsonl
        # Include ASI/MITRE attribution
```

**Why at training time?** The model should never see dangerous examples. Training-time filtering is the first line of defense.

**Step 5: Shellcheck validation**

```python
def validate_shellcheck(examples: List[Dict]) -> List[Dict]:
    """Validate Bash syntax"""
    result = subprocess.run(
        ["shellcheck", "--shell=bash", "--severity=error", "-"],
        input=example["output"].encode()
    )
```

**Step 6: Apply chat template**

```python
def apply_chat_template(examples: List[Dict]) -> List[Dict]:
    """Apply Qwen2.5 chat format"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages)
```

**Step 7: Split and save**

```python
def split_dataset(examples: List[Dict]) -> Dict:
    """80/10/10 split with fixed seed for reproducibility"""
    random.seed(42)
    random.shuffle(examples)
    # Split into train/val/test
```

---

### `scripts/download_semantic_model.py`

Downloads CodeBERT for the semantic guardrail (Layer 3).

```python
def download_codebert():
    model_name = "mrm8488/codebert-base-finetuned-detect-insecure-code"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Save locally for offline use
    tokenizer.save_pretrained("models/semantic/codebert-insecure-code")
    model.save_pretrained("models/semantic/codebert-insecure-code")
```

**Why download?** RunPod pods can lose network access. Saving locally ensures the model is always available.

---

## 4. How to Run

### On RunPod

```bash
# 1. Clone the repo
cd /workspace
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

# 2. Run one-command setup
bash scripts/setup_runpod.sh

# 3. Start training
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

### Expected Output

```
============================================================
SecureCLI-Tuner RunPod Setup
============================================================

Step 1/5: Installing system dependencies...
Step 2/5: Installing Python dependencies...
Step 3/5: Downloading datasets...
  Downloaded: 18357 examples
Step 4/5: Downloading CodeBERT semantic model...
  Model saved to: models/semantic/codebert-insecure-code
Step 5/5: Preparing training data...
  Removed 1247 duplicates
  Schema valid: 17110
  Dangerous removed: 0
  Shellcheck passed: 16823
  Split: 13458 train, 1682 val, 1683 test

✓ Setup complete!
```

---

## 5. Provenance Tracking

Every run creates `data/processed/provenance.json`:

```json
{
  "created_at": "2026-01-23T13:05:22+00:00",
  "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "total_raw": 18357,
  "after_dedup": 17110,
  "after_schema": 17110,
  "after_dangerous": 17110,
  "after_shellcheck": 16823,
  "train_size": 13458,
  "val_size": 1682,
  "test_size": 1683,
  "split_seed": 42
}
```

**Why provenance?** Reproducibility. If results change, you can trace back to exactly what data was used.

---

## 6. Interview Preparation

### Q: Walk me through your data preparation pipeline

**Model Answer:** "I run a single bash script that orchestrates five Python scripts. First, it downloads datasets from HuggingFace—about 18K Bash examples. Then it normalizes field names, deduplicates using SHA256 fingerprints, validates against a Pydantic schema, filters dangerous commands using our deterministic guardrail, validates Bash syntax with Shellcheck, applies the Qwen chat template, and splits 80/10/10. Everything is logged for auditability, and provenance is recorded with timestamps and counts at each stage."

### Q: Why do you filter dangerous commands at training time instead of just inference time?

**Model Answer:** "Defense in depth. If the model never sees dangerous examples during training, it's less likely to generate them. But training-time filtering alone isn't enough—V1 showed a 57% adversarial safe rate. That's why V2 adds runtime guardrails: the CommandRisk engine validates every generated command before execution."

### Q: How do you ensure reproducibility?

**Model Answer:** "Fixed seeds everywhere. The deduplication uses SHA256 for deterministic fingerprints. The split uses `random.seed(42)`. The provenance file records exact counts at each pipeline stage. Given the same input datasets and code, you get identical output."

### Q: What happens if Shellcheck isn't installed?

**Model Answer:** "The setup script installs it first: `apt-get install shellcheck`. If it's somehow missing during data preparation, the script catches the FileNotFoundError and logs a warning. On Windows development, I'd use `choco install shellcheck`. The pipeline is designed to fail loudly rather than silently skip validation."

### Q: How would you add a new dataset source?

**Model Answer:** "Create a new download function in `download_datasets.py` that normalizes to our schema—`instruction`, `output`, `input`, `tool`, `source`. Save to `data/raw/` as JSONL. The preparation script automatically picks up all JSONL files from that directory and merges them."

---

## 7. Key Takeaways

- ✅ One command (`setup_runpod.sh`) does everything
- ✅ 7-step pipeline: download → dedup → validate → filter → template → split → save
- ✅ Dangerous command filtering uses CommandRisk guardrail
- ✅ Provenance tracking for reproducibility and audit
- ✅ Fixed seeds ensure deterministic output

---

## 8. Next Steps

- Run setup on RunPod when GPUs are available
- Monitor training with W&B (SecureCLI-Training project)
- After training, evaluate with adversarial suite
