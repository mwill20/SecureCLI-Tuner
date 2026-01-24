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
# Step 2: Fix Python environment (Jan 2026 Stability Stack)
# -----------------------------------------------------------------------------
echo ""
echo "Step 2/6: Setting up Python environment (Jan 2026 Stability)..."

# Upgrade pip
pip install -q --upgrade pip

# Purge any existing problematic installations
echo "  Purging existing packages to ensure clean slate..."
pip uninstall -q -y torch torchvision torchaudio transformers axolotl peft bitsandbytes accelerate datasets 2>/dev/null || true

# Install specific, validated stack for cu124 (Torch 2.5.1 + Torchvision 0.20.1)
# We use the explicit index-url to ensure we get the CUDA-enabled versions
echo "  Installing validated Torch/Torchvision (CUDA 12.4)..."
pip install -q --root-user-action=ignore \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install LLM core stack (Pinned for Jan 2026 compatibility)
echo "  Installing LLM core (Transformers/PEFT/Accelerate)..."
pip install -q \
    transformers>=4.57.6 \
    peft>=0.17.1 \
    accelerate>=1.2.1 \
    datasets>=3.2.0 \
    bitsandbytes>=0.45.0 \
    pydantic pysigma PyYAML

# Install axolotl (Pinning to 0.10.0 for known stability in this pipeline)
echo "  Installing Axolotl..."
pip install -q axolotl==0.10.0 --no-deps

# Step 2b: Resolve the "Packaging" Conflict
# Axolotl wants 26.0, PySigma wants <26.0. We force 25.0 to satisfy both as much as possible.
echo "  Resolving dependency conflicts (pinning packaging==25.0)..."
pip install -q "packaging==25.0" --force-reinstall

# Fix Axolotl Telemetry Bug (Missing whitelist.yaml)
AXOLOTL_PATH=$(python -c "import axolotl; import os; print(os.path.dirname(axolotl.__file__))")
mkdir -p "$AXOLOTL_PATH/telemetry"
echo "[]" > "$AXOLOTL_PATH/telemetry/whitelist.yaml"

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
echo "1. Start Training (Standard):"
echo "   accelerate launch -m axolotl.cli.train training/axolotl_config.yaml"
echo ""
echo "2. Start Training (Safe/Private Mode - Recommended):"
echo "   AXOLOTL_DO_NOT_TRACK=1 accelerate launch -m axolotl.cli.train training/axolotl_config.yaml"
echo "   → Use this to prevent telemetry crashes and ensure maximum data privacy."
echo ""
echo "3. Monitor in Weights & Biases:"
echo "   https://wandb.ai/YOUR_USERNAME/SecureCLI-Training"
echo "   → View real-time loss curves, GPU usage, and safety benchmarks."
echo ""
