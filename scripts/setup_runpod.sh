#!/bin/bash
# =============================================================================
# SecureCLI-Tuner RunPod Setup Script (v3 - UV Edition)
# 
# Uses 'uv' for high-speed, reliable dependency resolution.
# =============================================================================

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         SecureCLI-Tuner RunPod Setup (v3 - UV)                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Step 0: Load environment variables
# -----------------------------------------------------------------------------
echo "Step 0/6: Loading environment variables..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "  ✓ Loaded .env file"
fi

# -----------------------------------------------------------------------------
# Step 1: Install UV (The Fast Package Manager)
# -----------------------------------------------------------------------------
echo ""
echo "Step 1/6: Installing 'uv' package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
# Fix: Use the correct path for uv on RunPod
export PATH="$HOME/.local/bin:$PATH"
echo "  ✓ uv installed and added to path"

# -----------------------------------------------------------------------------
# Step 2: System dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Step 2/6: Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq shellcheck git
echo "  ✓ System tools ready"

# -----------------------------------------------------------------------------
# Step 3: Fast Environment Setup (Deep Cleansing Mode)
# -----------------------------------------------------------------------------
echo ""
echo "Step 3/6: Deep cleansing Python environment..."

# 1. Purge via pip
pip uninstall -q -y torch torchvision transformers axolotl peft bitsandbytes accelerate datasets 2>/dev/null || true

# 2. Nuclear Strike: Manually remove suspected corrupted directories to avoid shadowing
echo "  Removing corrupted package traces..."
rm -rf /usr/local/lib/python3.11/dist-packages/transformers*
rm -rf /usr/local/lib/python3.11/dist-packages/peft*
rm -rf /usr/local/lib/python3.11/dist-packages/accelerate*
rm -rf /usr/local/lib/python3.11/dist-packages/axolotl*

# 3. Fresh installation using UV's lightning-fast resolver with NO CACHE
echo "  Performing clean-room installation of validated stack..."
# We relax the packaging constraint to let UV find the best match
# Use --extra-index-url for PyTorch so regular PyPI is still available for transformers
uv pip install --system --force-reinstall --no-cache \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --index-strategy unsafe-best-match \
    "torch==2.5.1+cu124" \
    "torchvision==0.20.1+cu124" \
    "transformers>=4.57.6" \
    "peft>=0.17.1" \
    "accelerate>=1.2.1" \
    "datasets>=3.2.0" \
    "bitsandbytes>=0.45.0" \
    "pydantic" "pysigma" "PyYAML" "wandb" "colorama" \
    "packaging<26.0,>=24.0"

# Install axolotl separately to ensure it doesn't downgrade transformers
uv pip install --system --no-deps "axolotl==0.10.0"

# Fix Axolotl Telemetry Bug
AXOLOTL_PATH=$(python -c "import axolotl; import os; print(os.path.dirname(axolotl.__file__))")
mkdir -p "$AXOLOTL_PATH/telemetry"
echo "[]" > "$AXOLOTL_PATH/telemetry/whitelist.yaml"

echo "  ✓ LLM Stack verified and installed"

# -----------------------------------------------------------------------------
# Step 4: Configure Weights & Biases
# -----------------------------------------------------------------------------
echo ""
echo "Step 4/6: Configuring Weights & Biases..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
    echo "  ✓ W&B logged in"
fi
export WANDB_PROJECT="SecureCLI-Training"

# -----------------------------------------------------------------------------
# Step 5: Data Pipeline (Automatic Re-run)
# -----------------------------------------------------------------------------
echo ""
echo "Step 5/6: Refreshing safety-verified datasets..."
python scripts/download_datasets.py
python scripts/prepare_training_data.py

# -----------------------------------------------------------------------------
# Step 6: Final Verification (with self-healing)
# -----------------------------------------------------------------------------
echo ""
echo "Step 6/6: Verifying final environment..."

# Test the specific import that was failing
if python -c "from transformers import PreTrainedModel; from peft import PeftModel; print('  ✓ All critical imports successful')"; then
    echo "  ✓ Environment verified!"
else
    echo "  ⚠ Import failed, attempting reinstall..."
    pip uninstall -y transformers peft 2>/dev/null || true
    rm -rf /usr/local/lib/python3.11/dist-packages/transformers* /usr/local/lib/python3.11/dist-packages/peft*
    uv pip install --system --force-reinstall --no-cache "transformers>=4.57.6" "peft>=0.17.1"
    python -c "from transformers import PreTrainedModel; from peft import PeftModel; print('  ✓ All critical imports successful after reinstall')"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✓ UV Setup Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Start Training (Private Mode):"
echo "   AXOLOTL_DO_NOT_TRACK=1 accelerate launch -m axolotl.cli.train training/axolotl_config.yaml"
echo ""
echo "2. Monitor in WandB:"
echo "   https://wandb.ai/YOUR_USERNAME/SecureCLI-Training"
echo ""
