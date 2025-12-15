"""
Training script for Text-to-3D ShapeForge model.

Trains a CLIP-conditioned point cloud generator on chair datasets.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import click
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.text_to_3d import TextConditionedGenerator

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


# Chair text labels for training
CHAIR_PROMPTS = [
    "wooden chair", "modern chair", "office chair", "dining chair",
    "armchair", "rocking chair", "bar stool", "lounge chair",
    "folding chair", "plastic chair", "metal chair", "leather chair",
    "vintage chair", "minimalist chair", "ergonomic chair", "accent chair",
]


class TextPointCloudDataset(Dataset):
    """Dataset pairing point clouds with text descriptions."""
    
    def __init__(self, data_dir: str, num_points: int = 4096):
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        
        # Load all .npy point cloud files
        self.files = list(self.data_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files in {data_dir}")
        
        # Assign text labels (cycle through prompts)
        self.labels = [CHAIR_PROMPTS[i % len(CHAIR_PROMPTS)] for i in range(len(self.files))]
        
        print(f"📦 Loaded {len(self.files)} samples with text labels")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Load point cloud
        points = np.load(self.files[idx])
        
        # Ensure correct point count
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            indices = np.random.choice(len(points), self.num_points - len(points))
            points = np.vstack([points, points[indices]])
        
        # Apply random augmentation
        points = self._augment(points)
        
        return torch.from_numpy(points).float(), self.labels[idx]
    
    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Random rotation and jitter."""
        # Random Y-axis rotation
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        points = points @ rot
        
        # Random jitter
        points += np.random.normal(0, 0.01, points.shape)
        
        return points.astype(np.float32)


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between two point clouds."""
    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    dist = torch.sum(diff ** 2, dim=-1)
    
    min_pred = torch.min(dist, dim=2)[0].mean()
    min_target = torch.min(dist, dim=1)[0].mean()
    
    return min_pred + min_target


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@click.command()
@click.option('--data', '-d', default='data/processed', help='Training data directory')
@click.option('--output', '-o', default='checkpoints', help='Checkpoint output directory')
@click.option('--epochs', '-e', default=100, help='Number of epochs')
@click.option('--batch-size', '-b', default=8, help='Batch size')
@click.option('--lr', default=1e-4, help='Learning rate')
def main(data: str, output: str, epochs: int, batch_size: int, lr: float):
    """Train text-conditioned 3D chair generator."""
    
    print("🏋️ ShapeForge Text-to-3D Training")
    print("=" * 50)
    
    device = get_device()
    print(f"🖥️ Device: {device}")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\n📦 Loading data from {data}")
    try:
        dataset = TextPointCloudDataset(data)
    except ValueError as e:
        print(f"❌ {e}")
        print("   Run: python data/download.py && python data/preprocess.py")
        sys.exit(1)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Load CLIP
    if HAS_CLIP:
        print("\n🔤 Loading CLIP text encoder...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        clip.to(device).eval()
        clip_dim = 512
    else:
        print("⚠️ CLIP not available, using random embeddings")
        tokenizer, clip = None, None
        clip_dim = 512
    
    # Create model
    print("\n🧠 Initializing model...")
    model = TextConditionedGenerator(clip_dim=clip_dim).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training loop
    print(f"\n🚀 Training for {epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for points, texts in pbar:
            points = points.to(device)
            
            # Encode text
            if clip is not None:
                inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_emb = clip(**inputs).pooler_output
            else:
                text_emb = torch.randn(len(texts), clip_dim, device=device)
            
            # Forward
            pred_points = model(text_emb)
            loss = chamfer_distance(pred_points, points)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, output_path / "shapeforge-text2shape-best.pt")
        
        # Save periodic
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
            }, output_path / f"shapeforge-text2shape-epoch{epoch}.pt")
    
    # Save final
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'loss': avg_loss,
    }, output_path / "shapeforge-text2shape-final.pt")
    
    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints: {output_path}/")
    print("\n🎯 Generate with:")
    print("   python inference/generate_from_text.py --prompt 'modern chair'")


if __name__ == "__main__":
    main()
