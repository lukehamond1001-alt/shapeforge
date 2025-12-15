"""
ShapeForge Training Script

Fine-tunes OpenAI's Shap-E model on ShapeNet chairs for specialized 3D generation.
Designed to run on cloud GPUs (RunPod, Lambda, etc.).
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import click
from tqdm import tqdm


class PointCloudDataset(Dataset):
    """Dataset for preprocessed point clouds."""
    
    def __init__(
        self,
        data_dir: str,
        num_points: int = 4096,
        augment: bool = True,
        config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.augment = augment
        self.config = config or {}
        
        # Find all .npy files
        self.files = list(self.data_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {data_dir}")
        
        print(f"📦 Loaded dataset with {len(self.files)} samples")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load point cloud
        points = np.load(self.files[idx])
        
        # Ensure correct number of points
        if len(points) > self.num_points:
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # Pad with duplicates if needed
            indices = np.random.choice(len(points), self.num_points - len(points))
            points = np.vstack([points, points[indices]])
        
        # Apply augmentation
        if self.augment:
            points = self._augment(points)
        
        return torch.from_numpy(points).float()
    
    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        aug_config = self.config.get('data', {}).get('augmentation', {})
        
        # Random rotation around Y axis
        if aug_config.get('random_rotation', True):
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            points = points @ rotation_matrix
        
        # Random scale
        scale_range = aug_config.get('random_scale', [0.9, 1.1])
        if scale_range:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            points = points * scale
        
        # Random jitter
        jitter = aug_config.get('random_jitter', 0.01)
        if jitter > 0:
            points = points + np.random.normal(0, jitter, points.shape)
        
        return points.astype(np.float32)


class ShapeForgeModel(nn.Module):
    """
    ShapeForge: Point cloud encoder for 3D generation.
    
    This is a simplified encoder that learns to embed point clouds.
    For production, integrate with full Shap-E pipeline.
    """
    
    def __init__(self, num_points: int = 4096, latent_dim: int = 256):
        super().__init__()
        
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # PointNet-style encoder
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        # Global feature aggregation
        self.global_feat = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 3),
        )
    
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to latent vector."""
        # points: (B, N, 3)
        features = self.encoder(points)  # (B, N, 256)
        global_features = torch.max(features, dim=1)[0]  # (B, 256)
        latent = self.global_feat(global_features)  # (B, latent_dim)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to point cloud."""
        # latent: (B, latent_dim)
        points_flat = self.decoder(latent)  # (B, N*3)
        points = points_flat.view(-1, self.num_points, 3)
        return points
    
    def forward(self, points: torch.Tensor) -> tuple:
        """Forward pass with reconstruction."""
        latent = self.encode(points)
        reconstructed = self.decode(latent)
        return reconstructed, latent


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between two point clouds.
    
    Args:
        pred: (B, N, 3) predicted points
        target: (B, N, 3) target points
        
    Returns:
        Chamfer distance (scalar)
    """
    # Compute pairwise distances
    # pred: (B, N, 1, 3), target: (B, 1, M, 3)
    pred_expand = pred.unsqueeze(2)
    target_expand = target.unsqueeze(1)
    
    # (B, N, M)
    distances = torch.sum((pred_expand - target_expand) ** 2, dim=-1)
    
    # For each point in pred, find nearest in target
    min_dist_pred = torch.min(distances, dim=2)[0]  # (B, N)
    
    # For each point in target, find nearest in pred
    min_dist_target = torch.min(distances, dim=1)[0]  # (B, M)
    
    # Chamfer distance
    chamfer = torch.mean(min_dist_pred) + torch.mean(min_dist_target)
    
    return chamfer


def get_device(config: Dict) -> torch.device:
    """Get the appropriate device for training."""
    device_str = config.get('hardware', {}).get('device', 'auto')
    
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    return torch.device(device_str)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    log_every = config.get('logging', {}).get('log_every', 100)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, points in enumerate(pbar):
        points = points.to(device)
        
        # Forward pass
        reconstructed, latent = model(points)
        
        # Compute loss (Chamfer distance)
        loss = chamfer_distance(reconstructed, points)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict,
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")


@click.command()
@click.option('--config', '-c', default='model/config.yaml', help='Path to config file')
@click.option('--data', '-d', default=None, help='Override data directory')
@click.option('--output', '-o', default='checkpoints', help='Output directory for checkpoints')
@click.option('--epochs', '-e', type=int, default=None, help='Override number of epochs')
@click.option('--batch-size', '-b', type=int, default=None, help='Override batch size')
@click.option('--resume', type=str, default=None, help='Resume from checkpoint')
def main(
    config: str,
    data: Optional[str],
    output: str,
    epochs: Optional[int],
    batch_size: Optional[int],
    resume: Optional[str]
):
    """
    Train ShapeForge on preprocessed point clouds.
    
    Examples:
        python train.py --epochs 50 --batch-size 8
        python train.py --resume checkpoints/latest.pt
    """
    print("🏋️ ShapeForge Training")
    print("=" * 50)
    
    # Load config
    config_path = Path(config)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        print(f"📋 Loaded config from {config}")
    else:
        cfg = {}
        print("⚠️ No config file found, using defaults")
    
    # Apply overrides
    if data:
        cfg.setdefault('data', {})['train_dir'] = data
    if epochs:
        cfg.setdefault('training', {})['epochs'] = epochs
    if batch_size:
        cfg.setdefault('training', {})['batch_size'] = batch_size
    
    # Extract config values
    data_dir = cfg.get('data', {}).get('train_dir', 'data/processed')
    num_points = cfg.get('data', {}).get('num_points', 4096)
    num_epochs = cfg.get('training', {}).get('epochs', 50)
    batch_sz = cfg.get('training', {}).get('batch_size', 8)
    lr = cfg.get('training', {}).get('learning_rate', 1e-4)
    save_every = cfg.get('training', {}).get('save_every', 10)
    
    # Setup device
    device = get_device(cfg)
    print(f"🖥️ Using device: {device}")
    
    # Create output directory
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    print(f"\n📦 Loading data from {data_dir}")
    try:
        dataset = PointCloudDataset(data_dir, num_points=num_points, config=cfg)
    except ValueError as e:
        print(f"❌ {e}")
        print("\n   Run preprocessing first:")
        print("   python data/download.py --test --limit 100")
        print("   python data/preprocess.py")
        sys.exit(1)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_sz,
        shuffle=True,
        num_workers=cfg.get('hardware', {}).get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print(f"\n🧠 Initializing ShapeForge model")
    model = ShapeForgeModel(num_points=num_points).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.get('training', {}).get('weight_decay', 0.01)
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if resume:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"📂 Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs")
    print(f"   Batch size: {batch_sz}")
    print(f"   Learning rate: {lr}")
    print()
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, device, epoch, cfg)
        
        print(f"   Epoch {epoch}/{num_epochs} - Loss: {loss:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 0:
            checkpoint_path = output_path / f"shapeforge-epoch{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, loss, cfg, str(checkpoint_path))
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = output_path / "shapeforge-best.pt"
            save_checkpoint(model, optimizer, epoch, loss, cfg, str(best_path))
    
    # Save final model
    final_path = output_path / "shapeforge-final.pt"
    save_checkpoint(model, optimizer, num_epochs, loss, cfg, str(final_path))
    
    print(f"\n✅ Training complete!")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"   Checkpoints: {output_path}/")
    
    print("\n🎯 Next step: Generate samples")
    print("   python inference/generate.py --checkpoint checkpoints/shapeforge-best.pt")


if __name__ == "__main__":
    main()
