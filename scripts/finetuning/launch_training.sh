#!/bin/bash
# =============================================================================
# Multi-GPU Fine-Tuning Launch Script
# =============================================================================
#
# Usage:
#   chmod +x launch_training.sh
#   ./launch_training.sh
#
# Or manually:
#   accelerate launch --config_file accelerate_config.yaml finetune_optimized.py
#
# =============================================================================

echo "============================================================"
echo "🚀 MULTI-GPU FINE-TUNING"
echo "============================================================"
echo ""

# Check for accelerate
if ! command -v accelerate &> /dev/null; then
    echo "❌ Accelerate not found. Installing..."
    pip install accelerate
fi

# Show GPU info
echo "🖥️  Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Launch training
echo "🚀 Launching training on all GPUs..."
echo ""

accelerate launch \
    --config_file "${SCRIPT_DIR}/accelerate_config.yaml" \
    "${SCRIPT_DIR}/finetune_optimized.py"

echo ""
echo "============================================================"
echo "✅ Training complete!"
echo "============================================================"
