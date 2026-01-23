#!/bin/bash
# =============================================================================
# SecureCLI-Tuner RunPod Setup Script
# 
# Run this script on RunPod to set up everything for training.
# 
# Usage:
#     bash scripts/setup_runpod.sh
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "SecureCLI-Tuner RunPod Setup"
echo "============================================================"

# -----------------------------------------------------------------------------
# Step 1: System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Step 1/5: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq shellcheck

# -----------------------------------------------------------------------------
# Step 2: Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Step 2/5: Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Additional training dependencies
pip install -q axolotl accelerate bitsandbytes wandb

# -----------------------------------------------------------------------------
# Step 3: Download datasets
# -----------------------------------------------------------------------------
echo ""
echo "Step 3/5: Downloading datasets..."
python scripts/download_datasets.py

# -----------------------------------------------------------------------------
# Step 4: Download semantic model
# -----------------------------------------------------------------------------
echo ""
echo "Step 4/5: Downloading CodeBERT semantic model..."
python scripts/download_semantic_model.py

# -----------------------------------------------------------------------------
# Step 5: Prepare training data
# -----------------------------------------------------------------------------
echo ""
echo "Step 5/5: Preparing training data..."
python scripts/prepare_training_data.py

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "✓ Setup complete!"
echo "============================================================"
echo ""
echo "Training data ready at: data/processed/"
echo "Semantic model ready at: models/semantic/"
echo ""
echo "To start training:"
echo "  accelerate launch -m axolotl.cli.train training/axolotl_config.yaml"
echo ""
echo "Or for a smoke test first:"
echo "  python -c \"from commandrisk import CommandRiskEngine; e = CommandRiskEngine(); print(e.validate('ls -la'))\""
echo ""
