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
|-----------|---------|
| **CommandRisk Engine** | 3-layer validation (Deterministic → Heuristic → Semantic) |
| **Hybrid AST + CodeBERT** | Fast structural analysis + ML intent classification |
| **OWASP ASI Compliance** | Every block mapped to ASI Top 10 + MITRE ATT&CK |
| **Semantic Evaluation** | Beyond exact-match: CodeBERT embeddings for functional equivalence |
| **AI-BOM** | CycloneDX supply chain transparency |

### Key Results

| Metric | V1 Result | V2 Target |
|--------|-----------|-----------|
| Training data safety | 0 dangerous commands | 0 dangerous commands |
| Command-only rate | 99.4% | 99%+ |
| Adversarial safe rate | 57% | **>95%** |
| Semantic match rate | N/A | **70-85%** |

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

## Quick Start

### Local Development

```powershell
cd C:\Projects\SecureCLI-Tuner
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run CLI
python main.py generate "list all docker containers"

# Run tests
python -m pytest tests/ -v
```

### RunPod Training

```bash
cd /workspace
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

# Create .env with keys
cat > .env << 'EOF'
WANDB_API_KEY=your_key
HF_TOKEN=your_token
EOF

# One-command setup
bash scripts/setup_runpod.sh

# Start training
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

---

## Project Structure

```
SecureCLI-Tuner/
├── cli_tuner/              # Command generation (router, generator, prompts)
├── commandrisk/            # Security engine
│   ├── types.py            # Shared types (avoid circular imports)
│   ├── engine.py           # 3-layer orchestrator
│   ├── wrapper.py          # Secure execution wrapper
│   └── guardrails/         # Deterministic, Semantic, Policy
├── data_pipeline/          # Training data preparation
├── evaluation/             # Semantic equivalence evaluator
├── scripts/                # Setup and utility scripts
│   ├── setup_runpod.sh     # One-command RunPod setup
│   ├── download_datasets.py
│   ├── prepare_training_data.py
│   ├── evaluate_semantic.py
│   └── demo_v1_security_gap.py
├── schemas/                # Pydantic validation schemas
├── training/               # QLoRA fine-tuning (axolotl configs)
├── compliance/             # OWASP & AI-BOM tools
├── docs/                   # SECURITY.md, OWASP_COMPLIANCE.md
│   └── lessons/            # 6 educational lessons (~4 hours)
└── tests/                  # Adversarial suite (>95% threshold)
```

---

## Security Philosophy

### Zero-Trust Architecture

Every component is untrusted until validated:

- **User input** → Prompt injection detection
- **LLM output** → CommandRisk 3-layer validation
- **Training data** → Zero-tolerance dangerous pattern filtering
- **Execution** → Secure wrapper with audit logging

### OWASP ASI Top 10 Coverage

| ASI ID | Vulnerability | Mitigation |
|--------|---------------|------------|
| ASI01 | Goal Hijack | Semantic Guardrail |
| ASI02 | Tool Misuse | Deterministic + Heuristic |
| ASI03 | Privilege Abuse | Deterministic Guardrail |
| ASI05 | Unexpected Execution | Secure Wrapper |

---

## Educational Materials

Comprehensive lessons for AI/ML engineers and security practitioners (~4 hours total):

| Lesson | Topic | Duration |
|--------|-------|----------|
| [Lesson 1](docs/lessons/Lesson_01_Data_Pipeline.md) | Security-First Data Pipelines | 45 min |
| [Lesson 2](docs/lessons/Lesson_02_Training.md) | QLoRA Fine-Tuning | 30 min |
| [Lesson 3](docs/lessons/Lesson_03_Evaluation.md) | Safety Evaluation | 30 min |
| [Lesson 4](docs/lessons/Lesson_04_CommandRisk.md) | CommandRisk Engine | 60 min |
| [Lesson 5](docs/lessons/Lesson_05_RunPod_Setup.md) | RunPod Setup & Data Prep | 45 min |
| [Lesson 6](docs/lessons/Lesson_06_Semantic_Evaluation.md) | Semantic Evaluation | 30 min |

---

## Dependencies

- Python 3.11+
- transformers >= 4.40.0
- pysigma >= 0.11.0 (SigmaHQ)
- mrm8488/codebert-base-finetuned-detect-insecure-code (Semantic layer)
- axolotl (QLoRA training)
- wandb (experiment tracking)

---

## License

MIT License - See LICENSE file

---

**Repository:** [https://github.com/mwill20/SecureCLI-Tuner](https://github.com/mwill20/SecureCLI-Tuner)
