# ShapeForge

<div align="center">

![ShapeForge Banner](assets/banner.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Shap-E](https://img.shields.io/badge/OpenAI-Shap--E-412991)](https://github.com/openai/shap-e)

**Train your own 3D generative model on ShapeNet**

[Quick Start](#-quick-start) • [Training](#-training) • [Inference](#-inference) • [Results](#-results)

</div>

---

## ✨ What is ShapeForge?

ShapeForge is a 3D generative model fine-tuned on the **ShapeNet chairs dataset**. It demonstrates:

- 🎓 **End-to-end ML pipeline** — Data preprocessing → Training → Inference
- 🪑 **Domain-specific generation** — Specializes in generating chair 3D models  
- ⚡ **Cloud-ready training** — Optimized for RunPod/Lambda GPUs
- 🔄 **Comparison with Imagen Apex** — Side-by-side with text-to-3D pipeline

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   ShapeNet      │ ──▶ │  Point Cloud    │ ──▶ │   Shap-E        │
│   Chairs (OBJ)  │     │  Preprocessing  │     │   Fine-tuning   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Novel Chair   │ ◀── │   3D Decoder    │ ◀── │  Trained Model  │
│   (PLY/OBJ)     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (for training) or Apple Silicon Mac (for inference)
- ~10GB disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/lukehamond1001-alt/shapeforge.git
cd shapeforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Inference (Using Pretrained)

```bash
# Generate a chair using pretrained Shap-E
python inference/generate.py --output outputs/chair.ply

# View in any 3D viewer (MeshLab, Blender, etc.)
```

---

## 📦 Data Pipeline

### Download ShapeNet Chairs

```bash
# Download chair subset (~2,000 models, ~2GB)
python data/download.py --category chair --output data/raw

# Or use the curated subset (recommended for quick training)
python data/download.py --curated --output data/raw
```

### Preprocess to Point Clouds

```bash
# Convert OBJ/PLY meshes to normalized point clouds
python data/preprocess.py \
    --input data/raw \
    --output data/processed \
    --num-points 4096
```

---

## 🏋️ Training

### Local Training (GPU required)

```bash
python model/train.py \
    --data data/processed \
    --output checkpoints/ \
    --epochs 50 \
    --batch-size 8
```

### Cloud Training (RunPod/Lambda)

1. **Launch GPU instance** — RTX 4090 recommended (~$0.50/hr)
2. **Clone and setup:**
   ```bash
   git clone https://github.com/lukehamond1001-alt/shapeforge.git
   cd shapeforge && pip install -r requirements.txt
   ```
3. **Download data and train:**
   ```bash
   python data/download.py --curated
   python data/preprocess.py
   python model/train.py --epochs 100
   ```
4. **Download checkpoint** back to local machine

**Estimated cost:** ~$2-5 for full training

---

## 🔮 Inference

### Generate New Chairs

```bash
# Using fine-tuned model
python inference/generate.py \
    --checkpoint checkpoints/shapeforge-v1.pt \
    --num-samples 5 \
    --output outputs/

# Using pretrained Shap-E (no training needed)
python inference/generate.py --pretrained --output outputs/
```

### Compare with Imagen Apex

```bash
# Generate comparison image
python inference/compare.py \
    --shapeforge-checkpoint checkpoints/shapeforge-v1.pt \
    --imagen-apex-endpoint https://your-endpoint/predict \
    --prompt "modern wooden chair" \
    --output comparison.png
```

---

## 📊 Results

| Model | Dataset | Training Time | Quality |
|-------|---------|--------------|---------|
| Shap-E (pretrained) | Objaverse | N/A | General |
| **ShapeForge** | ShapeNet Chairs | ~2-4 hrs | Chair-specialized |

### Sample Outputs

Coming soon — Generated chair examples

---

## 📁 Project Structure

```
shapeforge/
├── data/
│   ├── download.py         # Download ShapeNet
│   └── preprocess.py       # Convert to point clouds
├── model/
│   ├── config.yaml         # Training config
│   └── train.py            # Training script
├── inference/
│   ├── generate.py         # Generate 3D shapes
│   └── compare.py          # Compare with Imagen Apex
├── outputs/                # Generated models
├── checkpoints/            # Trained weights
└── assets/                 # Documentation assets
```

---

## 🔗 Related Projects

- [Imagen Apex](https://github.com/lukehamond1001-alt/imagen-apex) — Text-to-3D pipeline using Gemini + SAM 3D
- [OpenAI Shap-E](https://github.com/openai/shap-e) — Base model for ShapeForge
- [ShapeNet](https://shapenet.org/) — Training dataset source

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

<div align="center">
  <strong>Built with ❤️ using PyTorch and Shap-E</strong>
</div>
