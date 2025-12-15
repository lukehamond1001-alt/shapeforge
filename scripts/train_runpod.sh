#!/bin/bash
# ShapeForge Training Script for RunPod
# Usage: bash scripts/train_runpod.sh [prepare|train|push|all]

set -e

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_PROCEDURAL=${NUM_PROCEDURAL:-100}

prepare_data() {
    echo "📦 Preparing training data..."
    
    # Generate procedural chairs to augment the Imagen Apex PLY files
    echo "   Generating $NUM_PROCEDURAL procedural chairs..."
    python data/download.py --source procedural --limit $NUM_PROCEDURAL --output data/raw
    
    # Copy existing PLY files from examples
    if [ -d "examples/imagen_apex" ]; then
        echo "   Adding Imagen Apex PLY files..."
        mkdir -p data/raw/imagen_apex_01
        mkdir -p data/raw/imagen_apex_02
        cp examples/imagen_apex/chair_imagen_apex_01.ply data/raw/imagen_apex_01/model.ply 2>/dev/null || true
        cp examples/imagen_apex/chair_imagen_apex_02.ply data/raw/imagen_apex_02/model.ply 2>/dev/null || true
    fi
    
    # Preprocess all to point clouds
    echo "   Converting meshes to point clouds..."
    python data/preprocess.py --input data/raw --output data/processed --num-points 4096
    
    echo "✅ Data preparation complete!"
}

train_model() {
    echo "🏋️ Starting training..."
    echo "   Epochs: $EPOCHS"
    echo "   Batch size: $BATCH_SIZE"
    echo ""
    
    python model/train.py \
        --data data/processed \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output checkpoints/
    
    echo ""
    echo "✅ Training complete!"
}

push_results() {
    echo "📤 Pushing results to GitHub..."
    
    # Generate sample outputs for showcase
    if [ -f "checkpoints/shapeforge-best.pt" ]; then
        echo "   Generating sample outputs..."
        python inference/generate.py \
            --checkpoint checkpoints/shapeforge-best.pt \
            --num-samples 3 \
            --output outputs/samples/
    fi
    
    # Add and commit
    git add checkpoints/ outputs/
    git commit -m "Add trained ShapeForge model ($(date +%Y-%m-%d))" || echo "Nothing to commit"
    
    # Push (may need authentication)
    git push origin main || {
        echo ""
        echo "⚠️ Push failed. You may need to authenticate:"
        echo "   git remote set-url origin https://YOUR_TOKEN@github.com/lukehamond1001-alt/shapeforge.git"
        echo "   git push origin main"
    }
    
    echo ""
    echo "✅ Results pushed!"
}

case "${1:-all}" in
    prepare)
        prepare_data
        ;;
    train)
        train_model
        ;;
    push)
        push_results
        ;;
    all)
        prepare_data
        train_model
        push_results
        echo ""
        echo "🎉 All done! Remember to terminate your RunPod instance."
        ;;
    *)
        echo "Usage: $0 [prepare|train|push|all]"
        exit 1
        ;;
esac
