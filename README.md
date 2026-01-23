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
| **AI-BOM** | CycloneDX supply chain transparency |

### Key Results (from V1)

| Metric | Result | Notes |
|--------|--------|-------|
| Training data safety | 0 dangerous commands | Zero-tolerance filtering |
| Command-only rate | 99.4% | Format learning success |
| Adversarial safe rate | 57% (V1) → **>95% (V2 target)** | Hybrid guardrails |

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

---

## Project Structure

```
SecureCLI-Tuner/
├── cli_tuner/          # Command generation (router, generator, prompts)
├── commandrisk/        # Security engine (3 guardrails, wrapper)
├── training/           # QLoRA fine-tuning (axolotl configs)
├── evaluation/         # Safety evaluation suite
├── compliance/         # OWASP & AI-BOM tools
├── docs/               # SECURITY.md, OWASP_COMPLIANCE.md
│   └── lessons/        # Educational walkthroughs
└── tests/              # Adversarial suite (>95% threshold)
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

Comprehensive lessons for AI/ML engineers and security practitioners:

| Lesson | Topic |
|--------|-------|
| [Lesson 1](docs/lessons/Lesson_01_Data_Pipeline.md) | Security-First Data Pipelines |
| [Lesson 2](docs/lessons/Lesson_02_Training.md) | QLoRA Fine-Tuning |
| [Lesson 3](docs/lessons/Lesson_03_Evaluation.md) | Safety Evaluation |
| [Lesson 4](docs/lessons/Lesson_04_CommandRisk.md) | CommandRisk Engine |

---

## Dependencies

- Python 3.11+
- transformers >= 4.40.0
- pysigma >= 0.11.0 (SigmaHQ)
- mrm8488/codebert-base-finetuned-detect-insecure-code (Semantic layer)

---

## License

MIT License - See LICENSE file

---

**Repository:** [https://github.com/mwill20/SecureCLI-Tuner](https://github.com/mwill20/SecureCLI-Tuner)
