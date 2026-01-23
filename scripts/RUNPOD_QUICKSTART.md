# SecureCLI-Tuner RunPod Quick Start

## One-Command Setup

After uploading the repo to RunPod, run:

```bash
cd /workspace/SecureCLI-Tuner
bash scripts/setup_runpod.sh
```

This will:

1. Install shellcheck
2. Install Python dependencies
3. Download datasets (~18K Bash examples)
4. Download CodeBERT semantic model (~500MB)
5. Prepare training data (dedup, filter, split)

## Training

```bash
# Start training
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml

# Monitor in W&B
# Project: SecureCLI-Training
```

## Outputs

```
data/processed/
├── train.jsonl    # ~14K examples
├── val.jsonl      # ~1.7K examples
├── test.jsonl     # ~1.7K examples
└── provenance.json

models/semantic/
└── codebert-insecure-code/
```
