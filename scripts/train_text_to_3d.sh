#!/bin/bash
# ShapeForge Text-to-3D Training for RunPod
# Usage: bash scripts/train_text_to_3d.sh [prepare|train|push|all]

set -e

EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_CHAIRS=${NUM_CHAIRS:-100}

prepare_data() {
    echo "📦 Preparing training data..."
    
    # Generate procedural chairs
    echo "   Generating $NUM_CHAIRS procedural chairs..."
    python data/download.py --source procedural --limit $NUM_CHAIRS --output data/raw
    
    # Add Imagen Apex PLY files
    if [ -d "examples/imagen_apex" ]; then
        echo "   Adding Imagen Apex PLY files..."
        mkdir -p data/raw/imagen_apex_01
        mkdir -p data/raw/imagen_apex_02
        cp examples/imagen_apex/chair_imagen_apex_01.ply data/raw/imagen_apex_01/model.ply 2>/dev/null || true
        cp examples/imagen_apex/chair_imagen_apex_02.ply data/raw/imagen_apex_02/model.ply 2>/dev/null || true
    fi
    
    # Preprocess to point clouds
    echo "   Converting to point clouds..."
    python data/preprocess.py --input data/raw --output data/processed --num-points 4096
    
    echo "✅ Data ready!"
}

train_model() {
    echo "🏋️ Training Text-to-3D model..."
    
    python model/train_text_to_3d.py \
        --data data/processed \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --output checkpoints/
}

test_generation() {
    echo "🔮 Testing generation..."
    
    python inference/generate_from_text.py \
        --checkpoint checkpoints/shapeforge-text2shape-best.pt \
        --prompt "modern wooden chair" \
        --num-samples 3 \
        --output outputs/
}

push_results() {
    echo "📤 Pushing to GitHub..."
    
    git add checkpoints/ outputs/
    git commit -m "Add trained Text-to-3D model ($(date +%Y-%m-%d))" || true
    git push origin main || echo "⚠️ Push failed - authenticate and retry"
}

case "${1:-all}" in
    prepare) prepare_data ;;
    train) train_model ;;
    test) test_generation ;;
    push) push_results ;;
    all)
        prepare_data
        train_model
        test_generation
        push_results
        echo "🎉 Done! Remember to terminate RunPod."
        ;;
    *) echo "Usage: $0 [prepare|train|test|push|all]" ;;
esac
