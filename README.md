# SecureCLI-Tuner

![SecureCLI-Tuner Banner](docs/assets/banner.png)

> **Zero-Trust Security Kernel for Agentic DevOps (Bash, Git, Docker)**

[![OWASP ASI](https://img.shields.io/badge/OWASP-ASI%20Top%2010-blue)](https://owasp.org)
[![Ready Tensor 2026](https://img.shields.io/badge/Ready%20Tensor-2026-green)](https://readytensor.ai)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-SecureCLI--Tuner--V2-yellow)](https://huggingface.co/mwill-AImission/SecureCLI-Tuner-V2)

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

### Key Results (Verified — Run `honest-music-2`)

| Metric | Target | Result | Status |
| :------- | :------- | :------- | :------- |
| Command-only rate | ≥90% | **99.0%** | ✅ VERIFIED |
| Safety (dangerous removed) | 0 in training | 95 removed | ✅ VERIFIED |
| Adversarial pass rate | ≥95% | 100% (9/9) | ✅ VERIFIED |
| Final train loss | < 1.0 | 0.813 | ✅ VERIFIED |
| Final eval loss | < 1.0 | 0.861 | ✅ VERIFIED |
| Exact match rate* | ≥70% | 9.1% | ⚠️ See Note |

> [!NOTE]
> *Exact match is a conservative metric—`ls -la` vs `ls -al` are functionally identical but fail exact match. Command-only rate (99%) and adversarial pass rate (100%) are the primary quality indicators. See [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md) for details.
>
> Training completed on RunPod A100 (44.5 min, 500 steps). Model: [🤗 HuggingFace](https://huggingface.co/mwill-AImission/SecureCLI-Tuner-V2)

---

## Architecture

![SecureCLI-Tuner Architecture](docs/architecture.png)

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

---

## Ready Tensor Certification Documentation

This repository is structurally aligned with the Ready Tensor (RT) LLM Engineering & Deployment certification.

| Document | Purpose |
| :------- | :------- |
| [**Model Card**](docs/MODEL_CARD.md) | Standardized metadata, training config, and intended use. |
| [**Architecture**](docs/ARCHITECTURE.md) | System design, 3-layer guardrail logic, and OWASP mapping. |
| [**Evaluation**](docs/EVALUATION_DEEP_DIVE.md) | Failure analysis, baseline comparison (Base vs V2), and rigor. |
| [**Deployment**](docs/DEPLOYMENT.md) | Operational guide for local and cloud (RunPod) inference. |

---

## Educational Materials

Comprehensive lessons for AI/ML engineers and security practitioners (~4 hours total):

| Lesson | Topic |
| :------- | :------- |
| [**Lesson 1**](docs/lessons/Lesson_01_Data_Pipeline.md) | Security-First Data Pipelines |
| [**Lesson 2**](docs/lessons/Lesson_02_Training.md) | QLoRA Fine-Tuning |
| [**Lesson 3**](docs/lessons/Lesson_03_Evaluation.md) | Safety Evaluation |
| [**Lesson 4**](docs/lessons/Lesson_04_CommandRisk.md) | CommandRisk Engine |
| [**Lesson 5**](docs/lessons/Lesson_05_RunPod_Setup.md) | RunPod Setup & Data Prep |
| [**Lesson 6**](docs/lessons/Lesson_06_Semantic_Evaluation.md) | Semantic Evaluation |
| [**Lesson 7**](docs/lessons/Lesson_07_Inference_and_Use_Cases.md) | Inference & Use Cases |

---

## Citation & Professional Attribution

```bibtex
@misc{securecli_tuner_v2,
  author = { mwill-itmission },
  title = {SecureCLI-Tuner V2: A Security-First LLM for Agentic DevOps},
  year = {2026},
  publisher = {Ready Tensor Certification Portfolio}
}
```

---

**License:** MIT
**Repository:** [https://github.com/mwill20/SecureCLI-Tuner](https://github.com/mwill20/SecureCLI-Tuner)
