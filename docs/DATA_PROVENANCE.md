# Data Provenance & Methodology: SecureCLI-Tuner V2

## 1. Data Sources

The primary training dataset is derived from:

- **Source:** [pharaouk/dharma-1](https://huggingface.co/datasets/pharaouk/dharma-1)
- **Format:** JSONL (Instruction-following)
- **Original Size:** ~2,000 examples
- **Target Domains:** Bash, Git, Docker

## 2. Cleaning & Filtration (The Secure Pipeline)

To ensure high data quality and security alignment, we applied a **Zero-Tolerance Filtration Pipeline**:

### A. Syntax Validation

- Every generated shell command was passed through `ShellCheck`.
- Any command with syntax errors or structural invalidity was discarded.

### B. Security Decontamination

- **Dangerous Pattern Scan:** We scanned for 17 high-risk regex patterns (root deletions, fork bombs, etc.).
- **Result:** ~125 examples were discarded for containing potentially harmful instructions that contradicted the "Secure" intent of this model.

### C. Normalization

- All commands were normalized to a standard `Qwen-2.5-Chat` template.
- Assistant responses were masked to ensure the model focuses purely on command generation logic.

## 3. Dataset Statistics (Final)

| Split | Count | Percentage |
| :------- | :------- | :------- |
| **Train** | 1,000 | 81% |
| **Validation** | 100 | 8% |
| **Test** | 125 | 11% |
| **Total** | **1,225** | **100%** |

## 4. Reproducibility

The data pipeline is fully reproducible via:

```bash
python scripts/prepare_training_data.py
```

This script uses fixed random seeds and a SHA256 checksum to ensure that the exact same dataset is generated on any machine.
