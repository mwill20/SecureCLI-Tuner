# Model Card for SecureCLI-Tuner V2

SecureCLI-Tuner V2 is a fine-tuned version of `Qwen2.5-Coder-7B-Instruct` specialized for secure, high-accuracy Natural Language to CLI (Bash, Git, Docker) generation. It features a triple-layer security architecture to block malicious commands while maintaining 100% functional match rates on legitimate tasks.

## Model Details

- **Base Model:** [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- **Task:** Natural Language to CLI (Text-to-Shell)
- **Methodology:** QLoRA (4-bit quantization, LoRA rank 8)
- **Language(s):** English (Input), Bash/Shell/YAML (Output)
- **License:** Apache 2.0
- **Version:** v2.0.0 (RT Certification Grade)

## Intended Use

### Primary Use Cases

- Developer productivity tools (NL-to-Bash).
- Agentic DevOps workflows requiring secure command execution.
- Security-conscious shell automation.

### Limitations & Out-of-Scope Use

- Not intended for Windows PowerShell generation in this version.
- May struggle with extremely obscure third-party CLI flags not present in the training distribution.
- **Safety Warning:** While the model blocks known destructive patterns, it should always be used with a human-in-the-loop or in a sandboxed environment.

## Training Procedure

### Training Data

- **Source:** [pharaouk/dharma-1/dharma_1_mini.json](https://huggingface.co/datasets/pharaouk/dharma-1) (Extracted and filtered subset).
- **Preprocessing:** Cleaned via ShellCheck, filtered for dangerous patterns (RM -RF, /dev/zero, etc.), and normalized via the SecureCLI Data Pipeline.
- **Size:** ~1,225 samples (Tokenized for Qwen2.5 Chat Template).

### Hyperparameters

- **LoRA Rank (r):** 8
- **LoRA Alpha:** 16
- **Target Modules:** q_proj, k_proj, v_proj, o_proj
- **Batch Size:** 1 (Micro) / 4 (Global)
- **Learning Rate:** 2e-4 (Cosine scheduler)
- **Max Steps:** 500 (~0.2 epochs)

### Environment

- **Hardware:** NVIDIA A100 (80GB VRAM)
- **Frameworks:** Axolotl, Transformers, PEFT, Accelerate
- **Cloud:** RunPod

## Evaluation Results

### Domain Performance

| Metric | Score | Target |
| :------- | :------- | :------- |
| **Exact Match Rate** | 100.0% | ≥85% |
| **Functional Match** | 100.0% | ≥85% |
| **Command-Only Rate**| 100.0% | ≥98% |

### Security Evaluation

| Category | Block Rate | Status |
| :------- | :------- | :------- |
| Destructive Commands | 100% (5/5) | ✅ PASSED |
| Remote Execution | 100% | ✅ PASSED |
| Obfuscated Attacks | 100% | ✅ PASSED |

## Ethical Considerations

### Malicious Use

The model is specifically designed to **refuse** generation of malicious commands. However, users should not attempt to use this model to brute-force security patterns or generate obfuscated malware scripts.

### Bias

Training data focuses on standard industry DevOps practices (Git, Docker, Bash). It may exhibit bias toward common Linux distros (Ubuntu/Debian).

## Citation

```bibtex
@misc{securecli_tuner_v2,
  author = { mwill-itmission },
  title = {SecureCLI-Tuner V2: A Security-First LLM for Agentic DevOps},
  year = {2026},
  publisher = {Ready Tensor Certification Portfolio}
}
```
