# SecureCLI-Tuner RunPod Lifecycle Guide

Follow these steps in order to execute a full, safety-verified training run that logs metrics to **Weights & Biases**.

## 1. Setup Environment

Launch a RunPod instance (A100 80GB or RTX 4090 recommended). Once connected via terminal:

```bash
# Clone the repository
cd /workspace
git clone https://github.com/mwill20/SecureCLI-Tuner.git
cd SecureCLI-Tuner

# Create your authentication file
# Replace 'your_key' with your actual keys from wandb.ai and huggingface.co
cat > .env << 'EOF'
WANDB_API_KEY=your_wandb_api_key_here
HF_TOKEN=your_huggingface_token_here
EOF
```

## 2. Automated Setup (Recommended)

Run the master setup script. This script handles:

- Fixing `torch` version conflicts
- Installing `accelerate`, `bitsandbytes`, and `axolotl`
- **WandB Login**: Automatically logs you in using the key from `.env`
- **Data Download**: Fetches the raw Bash datasets
- **Safety Preparation**: Scrubs dangerous commands and applies templates

```bash
bash scripts/setup_runpod.sh
```

## 3. Manual Data Controls (If needed)

If you need to re-run the data pipeline without a full setup:

```bash
# Step A: Download raw data from HF
python scripts/download_datasets.py

# Step B: Filter dangerous commands & Prepare for training
python scripts/prepare_training_data.py
```

## 4. Launch Training

Start the fine-tuning process. This will automatically sync your loss, learning rate, and safety benchmarks to WandB.

```bash
AXOLOTL_DO_NOT_TRACK=1 accelerate launch -m axolotl.cli.train training/axolotl_config.yaml
```

## 5. Monitor Progress

Open the URL provided in the terminal (usually `https://wandb.ai/[your-username]/SecureCLI-Training`) to monitor:

- **Training Loss**: Downward trend indicates learning.
- **Eval Loss**: Check for overfitting.
- **Resource Usage**: Track GPU memory and power.

---

### Troubleshooting

**WandB Login Issues:**
If the setup script fails to log you in, run:

```bash
wandb login [your_api_key]
```

**Missing Directories:**
The data scripts will automatically create `data/raw` and `data/processed`. If you see errors about missing folders, the setup script (Step 2) likely didn't finish.
