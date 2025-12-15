#!/bin/bash
# ShapeForge RunPod Setup Script
# Run this after cloning the repo on a RunPod instance

set -e

echo "🚀 ShapeForge RunPod Setup"
echo "=========================="

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ No GPU detected. Training will be slow."
else
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify PyTorch CUDA
echo ""
echo "🔍 Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Setup git for pushing results
echo ""
echo "📝 Configuring git..."
git config --global user.email "runpod@shapeforge.training"
git config --global user.name "ShapeForge RunPod"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Prepare data:  bash scripts/train_runpod.sh prepare"
echo "  2. Train model:   bash scripts/train_runpod.sh train"
echo "  3. Push results:  bash scripts/train_runpod.sh push"
echo ""
echo "  Or run all at once: bash scripts/train_runpod.sh all"
