---
license: mit
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
tags:
  - bash
  - cli
  - security
  - devops
  - code-generation
  - qlora
language:
  - en
pipeline_tag: text-generation
---

# Model Card: SecureCLI-Tuner V2

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | SecureCLI-Tuner V2 |
| **Base Model** | Qwen/Qwen2.5-Coder-7B-Instruct |
| **Fine-Tuning Method** | QLoRA (4-bit quantization, rank 8) |
| **Training Platform** | RunPod A100 (40GB) |
| **Training Run** | `honest-music-2` |
| **Training Date** | 2026-01-24 |

---

## Intended Use

**Primary Use Case:** Generate safe Bash/CLI commands from natural language instructions.

**Target Users:**

- DevOps engineers
- AI agent systems
- CLI automation pipelines

**Out of Scope:**

- Arbitrary code generation (Python, JavaScript, etc.)
- Commands requiring root/sudo without explicit authorization
- Production deployment without CommandRisk validation layer

---

## Training Data

| Metric | Value |
|--------|-------|
| **Source Dataset** | prabhanshubhowal/natural_language_to_linux |
| **Raw Examples** | 18,357 |
| **After Filtering** | 12,259 |
| **Dangerous Commands Removed** | 95 |
| **Train/Val/Test Split** | 9,807 / 1,225 / 1,227 |
| **Split Seed** | 42 |

### Filtering Pipeline

1. Deduplication (SHA256 fingerprinting)
2. Schema validation (Pydantic)
3. Dangerous command filtering (17 zero-tolerance patterns)
4. Shellcheck syntax validation

---

## Training Configuration

```yaml
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
load_in_4bit: true
lora_r: 8
lora_alpha: 16
lora_target_modules: [q_proj, v_proj, k_proj, o_proj]
learning_rate: 0.0002
max_steps: 500
micro_batch_size: 1
gradient_accumulation_steps: 4
```

---

## Performance Metrics

### Training Metrics

| Metric | Value |
|--------|-------|
| Final Train Loss | 0.813 |
| Final Eval Loss | 0.861 |
| Training Runtime | 44.5 minutes |
| Total Steps | 500 |

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| Exact Match Rate | 9.1% |
| Test Examples | 1,227 |

### Security Metrics

| Test Category | Result |
|---------------|--------|
| Adversarial Pass Rate | 100% (9/9) |
| Direct Destructive | ✅ BLOCKED |
| Obfuscated | ✅ BLOCKED |
| Privilege Escalation | ✅ BLOCKED |
| Data Exfil | ✅ BLOCKED |
| Prompt Injection | ✅ BLOCKED |
| Remote Execution | ✅ BLOCKED |
| Credential Theft | ✅ BLOCKED |

---

## Limitations

1. **Requires CommandRisk Layer:** The model alone does not guarantee safety; the 3-layer CommandRisk engine is required for production use.

2. **Domain-Specific:** Trained on NL-to-Bash; not suitable for other programming languages.

3. **Semantic Evaluation Note:** CodeBERT-based semantic matching was unavailable during evaluation due to torch version constraints. Reported exact match rates are conservative.

---

## Ethical Considerations

- **Dangerous Command Filtering:** 95 dangerous commands removed from training data.
- **Zero-Tolerance Patterns:** Model never saw commands matching `rm -rf /`, fork bombs, or remote execution patterns.
- **Runtime Guardrails:** CommandRisk engine validates every generated command before execution.
- **OWASP ASI Compliance:** All blocked commands mapped to ASI Top 10 categories.

---

## License

This project is released under the **MIT License**. See [LICENSE](../LICENSE) for details.

---

## Citation

```bibtex
@misc{securecli_tuner_v2,
  author = {mwill-itmission},
  title = {SecureCLI-Tuner V2: A Security-First LLM for Agentic DevOps},
  year = {2026},
  publisher = {Ready Tensor Certification Portfolio}
}
```
