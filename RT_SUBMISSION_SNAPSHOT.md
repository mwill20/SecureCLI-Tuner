# RT Submission Snapshot: SecureCLI-Tuner V2

## 1. FILE TREE (tree -L 3)

```text
C:.
+---.agent
+---CLI-Tuner_v1
+---cli_tuner
+---commandrisk
|   +---agent
|   +---commands
|   \---rules
+---compliance
|   \---standards
+---configs
+---data
+---data_pipeline
+---deployment
+---docs
|   +---lessons
|   \---training
+---evaluation
+---model
|   \---checkpoints
+---monitoring
+---reproducibility
+---schemas
+---scripts
+---tests
|   +---eval
|   \---test_guardrails
\---training
```

## 2. TRAINING METRICS

- **Final exact match accuracy**: 100.0%
- **Final command-only rate**: 100.0%
- **Safety test pass rate**: 100.0%
- **Training set size**: 1,000
- **Test set size**: 125
- **Base model**: Qwen2.5-Coder-7B-Instruct
- **LoRA rank**: 8

## 3. WHAT EXISTS

- **docs/**:
  - ARCHITECTURE.md
  - DATA_PROVENANCE.md
  - DEPLOYMENT.md
  - EVALUATION_DEEP_DIVE.md
  - MODEL_CARD.md
  - OWASP_COMPLIANCE.md
  - SECURITY.md
  - TRAINING_RUN_V2.md
  - lessons/ (7 lessons)
  - training/ (Run logs and artifacts)
- **configs/**:
  - evaluation_config.yaml
- **evaluation/results/**:
  - *Directory does not currently exist.* (Metrics are stored in `docs/training/runs/v2_verified_A100/`)

## 4. WHAT'S MISSING

- **README.md (with quickstart)**: EXISTS (in root)
- **MODEL_CARD.md (HuggingFace format)**: EXISTS (in `docs/MODEL_CARD.md`)
- **EVALUATION_REPORT.md**: MISSING (Content is split between `docs/TRAINING_RUN_V2.md` and `docs/EVALUATION_DEEP_DIVE.md`)
- **requirements.txt**: EXISTS (in root)
- **inference example script**: EXISTS (`main.py` provides CLI generation; `README.md` contains a 5-line Python snippet)

## 5. CURRENT README CONTENT

```markdown
# SecureCLI-Tuner

> **Zero-Trust Security Kernel for Agentic DevOps (Bash, Git, Docker)**

[![OWASP ASI](https://img.shields.io/badge/OWASP-ASI%20Top%2010-blue)](https://owasp.org)
[![Ready Tensor 2026](https://img.shields.io/badge/Ready%20Tensor-2026-green)](https://readytensor.ai)

---

## The Problem We're Solving

**DevOps engineers and AI agents frequently generate CLI commands, but face a critical trust problem:**

- LLMs hallucinate dangerous operations (`rm -rf /`, `chmod 777 /`) without warning
- Agent systems execute commands without human-in-the-loop validation
- Training-time filtering alone is insufficient—adversarial prompts bypass safety measures
- No standardized security framework exists for agentic command generation

**Real-world impact:** Data loss, system destruction, and security breaches from unvalidated AI-generated commands.

---

## What This Project Demonstrates

**SecureCLI-Tuner** is a production-quality security kernel that validates every generated command before execution.

| Component | Purpose |
| :------- | :------ |
| **CommandRisk Engine** | 3-layer validation (Deterministic → Heuristic → Semantic) |
| **Hybrid AST + CodeBERT** | Fast structural analysis + ML intent classification |
| **OWASP ASI Compliance** | Every block mapped to ASI Top 10 + MITRE ATT&CK |
| **Semantic Evaluation** | Beyond exact-match: CodeBERT embeddings for functional equivalence |
| **AI-BOM** | CycloneDX supply chain transparency |

### Key Results (Verified 2026-01-23)

| Metric | V1 Result | V2 Verified | Status |
| :------- | :------- | :------- | :------- |
| Training data safety | 0 dangerous | **0 dangerous** | ✅ PASS |
| Command-only rate | 99.4% | **100.0%** | ✅ PASS |
| Adversarial safe rate | 57% | **100.0%** | ✅ PASS |
| Functional match rate | 13.2% | **100.0%** | ✅ PASS |

> [!NOTE]
> All metrics verified on 1,227 test examples using the hybrid semantic evaluator. See [TRAINING_RUN_V2.md](docs/TRAINING_RUN_V2.md) for full logs.

---

## Architecture

```

Natural Language → Router → Generator (LLM) → CommandRisk Engine → Secure Wrapper → Execution
                                                      ↓
                              ┌───────────────────────┼───────────────────────┐
                              ↓                       ↓                       ↓
                         Deterministic           Heuristic              Semantic
                         (17 regex)            (MITRE scoring)     (AST + CodeBERT)
                              ↓                       ↓                       ↓
                         SigmaHQ YAML            MITRE ATT&CK         OWASP ASI

```

### Three Guardrail Layers

| Layer | Type | Speed | Coverage |
|-------|------|-------|----------|
| 1 | Deterministic | <1ms | 17 zero-tolerance patterns |
| 2 | Heuristic | <5ms | Risk scoring 0-100 |
| 3 | Semantic | 50-100ms | Hybrid AST + CodeBERT |

---

## Quick Start (5-Line Example)

```python
from cli_tuner.generator import CLIGenerator

# Validates intent vs command using the 3-layer security kernel
generator = CLIGenerator(checkpoint="model/checkpoints/checkpoint-500")

response = generator.generate("List all docker containers running on port 80")
print(f"Generated Command: {response.command}")  # Verified Safe Output
```

### Local Development Setup

```powershell
cd C:\Projects\SecureCLI-Tuner
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py generate "list all files in current directory"
```

```

## 6. DATASET INFO

- **Where did training data come from?**: Derived from `pharaouk/dharma-1` (filtered for security).
- **How many examples total?**: 1,225 samples (1,000 train, 100 validation, 125 test).
- **What format? (JSONL, CSV, etc.)**: JSONL (normalized for Qwen-2.5-Chat template).

## 7. MODEL LOCATION

- **Where are model weights?**: Local path: `model/checkpoints/checkpoint-500` (LoRA adapter weights).
- **Published on HuggingFace yet?**: No public link yet; weights are currently stored locally.

## 8. KNOWN ISSUES

- **Incomplete**: `evaluation/results/` directory is missing or empty; results are instead nested within `docs/training/runs/v2_verified_A100/`.
- **Naming**: `EVALUATION_REPORT.md` is not present by that exact name, though comprehensive evaluation data exists in `docs/TRAINING_RUN_V2.md`.
- **Unsure**: If a dedicated, standalone `inference.py` script (separate from `main.py`) is required for RT submission.
