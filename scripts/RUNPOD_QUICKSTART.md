# SecureCLI-Tuner RunPod Quick Reference

## Prerequisites

Create `.env` file with your keys:

```bash
WANDB_API_KEY=your_wandb_key
HF_TOKEN=your_huggingface_token
```

---

## Complete RunPod Commands (Copy/Paste)

### Step 1: Clone and Setup

```bash
cd /workspace
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

# Create .env with your keys
cat > .env << 'EOF'
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_hf_token_here
EOF
```

### Step 2: Run Setup

```bash
bash scripts/setup_runpod.sh
```

### Step 3: Start Training

```bash
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

### Step 4: Monitor (open in browser)

```
https://wandb.ai/YOUR_USERNAME/SecureCLI-Training
```

---

## If Setup Fails

### Fix torch/torchvision conflict manually

```bash
pip install --no-deps torchvision==0.19.1+cu124
```

### Skip CodeBERT download (not needed for training)

```bash
python scripts/download_datasets.py
python scripts/prepare_training_data.py
```

### Manual W&B login

```bash
wandb login YOUR_API_KEY
```

---

## Expected Output

After `setup_runpod.sh`:

```
data/processed/
├── train.jsonl    (~14K examples)
├── val.jsonl      (~1.7K examples)  
├── test.jsonl     (~1.7K examples)
└── provenance.json
```

Training takes ~45 min on A100 for 500 steps.
