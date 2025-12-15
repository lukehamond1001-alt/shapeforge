"""
ShapeForge Inference - Generate Novel 3D Shapes

Generate new chair models using a trained ShapeForge model
or the pretrained Shap-E base model.
"""

import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import click
from tqdm import tqdm

# Import the model from train.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.train import ShapeForgeModel


def save_point_cloud_ply(points: np.ndarray, path: str) -> None:
    """Save point cloud as PLY file."""
    with open(path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points with color (gradient based on height)
        for point in points:
            x, y, z = point
            # Color based on height (y-axis), blue to red gradient
            t = (y + 1) / 2  # Normalize to [0, 1]
            r = int(255 * t)
            g = int(100)
            b = int(255 * (1 - t))
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def load_model(checkpoint_path: str, device: torch.device) -> ShapeForgeModel:
    """Load a trained ShapeForge model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    num_points = config.get('data', {}).get('num_points', 4096)
    
    # Create model
    model = ShapeForgeModel(num_points=num_points)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_random_latent(latent_dim: int = 256, device: torch.device = None) -> torch.Tensor:
    """Generate a random latent vector."""
    z = torch.randn(1, latent_dim)
    if device:
        z = z.to(device)
    return z


def generate_with_shapeforge(
    model: ShapeForgeModel,
    device: torch.device,
    num_samples: int = 1,
    seed: Optional[int] = None
) -> list:
    """Generate samples using ShapeForge model."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    samples = []
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate random latent vector
            z = generate_random_latent(model.latent_dim, device)
            
            # Decode to point cloud
            points = model.decode(z)
            points = points.squeeze(0).cpu().numpy()
            
            samples.append(points)
    
    return samples


def generate_with_pretrained_shape() -> np.ndarray:
    """
    Generate using pretrained Shap-E model.
    
    Note: This requires the shap-e package to be installed.
    """
    try:
        import shap_e
        from shap_e.diffusion.sample import sample_latents
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_model, load_config
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("📥 Loading pretrained Shap-E model...")
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        
        print("🎨 Generating with prompt: 'a chair'")
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=15.0,
            model_kwargs=dict(texts=['a chair']),
            progress=True,
            clip_denoised=True,
        )
        
        # Decode to point cloud
        pc = xm.decode_to_points(latents[0:1])
        points = pc[0].cpu().numpy()
        
        return points
        
    except ImportError:
        print("⚠️ shap-e not installed. Install with:")
        print("   pip install git+https://github.com/openai/shap-e.git")
        print("\n   Using random point cloud as fallback...")
        
        # Generate a simple random "chair-like" shape
        return generate_placeholder_chair()


def generate_placeholder_chair() -> np.ndarray:
    """Generate a placeholder chair-like point cloud for testing."""
    points = []
    
    # Seat (rectangular)
    for _ in range(1000):
        x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(0.3, 0.35)
        z = np.random.uniform(-0.4, 0.4)
        points.append([x, y, z])
    
    # Backrest
    for _ in range(800):
        x = np.random.uniform(-0.4, 0.4)
        y = np.random.uniform(0.35, 0.9)
        z = np.random.uniform(0.35, 0.4)
        points.append([x, y, z])
    
    # Four legs
    for leg_x, leg_z in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]:
        for _ in range(300):
            x = leg_x + np.random.uniform(-0.03, 0.03)
            y = np.random.uniform(-0.5, 0.3)
            z = leg_z + np.random.uniform(-0.03, 0.03)
            points.append([x, y, z])
    
    points = np.array(points)
    
    # Normalize
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points = points / max_dist
    
    return points.astype(np.float32)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@click.command()
@click.option('--checkpoint', '-c', default=None, help='Path to ShapeForge checkpoint')
@click.option('--pretrained', is_flag=True, help='Use pretrained Shap-E model')
@click.option('--num-samples', '-n', default=1, help='Number of samples to generate')
@click.option('--output', '-o', default='outputs', help='Output directory')
@click.option('--seed', '-s', type=int, default=None, help='Random seed for reproducibility')
def main(
    checkpoint: Optional[str],
    pretrained: bool,
    num_samples: int,
    output: str,
    seed: Optional[int]
):
    """
    Generate novel 3D chair shapes.
    
    Examples:
        python generate.py --checkpoint checkpoints/shapeforge-best.pt
        python generate.py --pretrained --num-samples 5
        python generate.py --output my_chairs/
    """
    print("🔮 ShapeForge Generator")
    print("=" * 50)
    
    # Setup
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"🎲 Seed: {seed}")
    
    # Generate samples
    if checkpoint:
        print(f"\n📂 Loading checkpoint: {checkpoint}")
        model = load_model(checkpoint, device)
        
        print(f"⚡ Generating {num_samples} samples...")
        samples = generate_with_shapeforge(model, device, num_samples, seed)
        
    elif pretrained:
        print("\n🌐 Using pretrained Shap-E model...")
        samples = []
        for i in tqdm(range(num_samples), desc="Generating"):
            points = generate_with_pretrained_shape()
            samples.append(points)
            
    else:
        print("\n📦 No checkpoint specified, generating placeholder chairs...")
        samples = []
        for i in range(num_samples):
            points = generate_placeholder_chair()
            samples.append(points)
    
    # Save outputs
    print(f"\n💾 Saving {len(samples)} samples to {output_path}/")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, points in enumerate(samples):
        filename = f"chair_{timestamp}_{i:04d}.ply"
        filepath = output_path / filename
        save_point_cloud_ply(points, str(filepath))
        print(f"   ✅ {filename} ({len(points)} points)")
    
    print(f"\n🎉 Generation complete!")
    print(f"   Output: {output_path}/")
    print("\n   View in any 3D viewer (MeshLab, Blender, etc.)")


if __name__ == "__main__":
    main()
