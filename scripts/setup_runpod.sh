#!/bin/bash
# =============================================================================
# SecureCLI-Tuner RunPod Setup Script (v2 - Fixed)
# 
# This script handles version conflicts and sets up W&B properly.
# 
# Prerequisites:
#   1. Create .env file with your keys (or export them directly)
#   2. Run this script
#
# Usage:
#     bash scripts/setup_runpod.sh
# =============================================================================

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SecureCLI-Tuner RunPod Setup (v2)                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Load environment variables
# -----------------------------------------------------------------------------
echo "Step 0/6: Loading environment variables..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "  ✓ Loaded .env file"
else
    echo "  ⚠ No .env file found. Make sure WANDB_API_KEY is set."
fi

# -----------------------------------------------------------------------------
# Step 1: System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Step 1/6: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq shellcheck
echo "  ✓ Shellcheck installed"

# -----------------------------------------------------------------------------
# Step 2: Fix Python environment (handle version conflicts)
# -----------------------------------------------------------------------------
echo ""
echo "Step 2/6: Setting up Python environment..."

# Upgrade pip
pip install -q --upgrade pip

# Fix torchvision compatibility (don't upgrade torch, keep what RunPod has)
pip install -q --no-deps torchvision==0.19.1+cu124 || true

# Install core dependencies (without changing torch)
pip install -q datasets transformers pydantic pysigma PyYAML

# Install training dependencies
pip install -q accelerate bitsandbytes wandb

# Install axolotl (for training)
pip install -q axolotl

echo "  ✓ Dependencies installed"

# -----------------------------------------------------------------------------
# Step 3: Configure Weights & Biases
# -----------------------------------------------------------------------------
echo ""
echo "Step 3/6: Configuring Weights & Biases..."

if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
    echo "  ✓ W&B logged in"
else
    echo "  ⚠ WANDB_API_KEY not set. Run: wandb login"
fi

# Set W&B project
export WANDB_PROJECT="SecureCLI-Training"
export WANDB_ENTITY=""  # Will use default
echo "  W&B Project: $WANDB_PROJECT"

# -----------------------------------------------------------------------------
# Step 4: Download datasets
# -----------------------------------------------------------------------------
echo ""
echo "Step 4/6: Downloading datasets..."
python scripts/download_datasets.py

# -----------------------------------------------------------------------------
# Step 5: Prepare training data
# -----------------------------------------------------------------------------
echo ""
echo "Step 5/6: Preparing training data..."
python scripts/prepare_training_data.py

# -----------------------------------------------------------------------------
# Step 6: Skip CodeBERT for now (can download separately)
# -----------------------------------------------------------------------------
echo ""
echo "Step 6/6: Verifying setup..."
echo "  ✓ Datasets ready: $(ls -la data/processed/*.jsonl 2>/dev/null | wc -l) files"
echo "  ✓ Training config: training/axolotl_config.yaml"

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                     ✓ Setup Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Training data ready at: data/processed/"
echo "  - train.jsonl"
echo "  - val.jsonl"  
echo "  - test.jsonl"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Start training:"
echo "   accelerate launch -m axolotl.cli.train training/axolotl_config.yaml"
echo ""
echo "2. Monitor in W&B:"
echo "   https://wandb.ai/YOUR_USERNAME/SecureCLI-Training"
echo ""
