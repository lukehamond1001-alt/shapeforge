"""
ShapeForge Text-to-3D Model

A minimal CLIP-conditioned point cloud generator for chairs.
Input: Text prompt (e.g., "modern wooden chair")
Output: PLY file of a 3D chair
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("⚠️ Install transformers for text conditioning: pip install transformers")


class TextConditionedGenerator(nn.Module):
    """
    Text-to-3D point cloud generator using CLIP embeddings.
    
    Architecture:
        Text → CLIP → Embedding → MLP Decoder → Point Cloud (N x 3)
    """
    
    def __init__(
        self, 
        num_points: int = 4096,
        latent_dim: int = 512,
        clip_dim: int = 512,  # CLIP embedding size
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Text embedding projection (CLIP → latent)
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Noise embedding for variation
        self.noise_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )
        
        # Combined decoder: latent → point cloud
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_points * 3),
            nn.Tanh(),  # Output in [-1, 1] range
        )
    
    def forward(self, text_embedding: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate point cloud from text embedding.
        
        Args:
            text_embedding: (B, clip_dim) CLIP text embedding
            noise: (B, latent_dim) random noise for variation
            
        Returns:
            points: (B, num_points, 3) generated point cloud
        """
        batch_size = text_embedding.shape[0]
        device = text_embedding.device
        
        # Project text embedding
        text_latent = self.text_proj(text_embedding)
        
        # Add noise for variation
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=device)
        noise_latent = self.noise_proj(noise)
        
        # Combine and decode
        combined = torch.cat([text_latent, noise_latent], dim=-1)
        points_flat = self.decoder(combined)
        points = points_flat.view(batch_size, self.num_points, 3)
        
        return points


class ShapeForgeTextTo3D:
    """
    Main interface for text-to-3D generation.
    
    Usage:
        model = ShapeForgeTextTo3D()
        points = model.generate("modern wooden chair")
        model.save_ply(points, "chair.ply")
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        # Set device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load CLIP
        if HAS_CLIP:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip.to(self.device)
            self.clip.eval()
            clip_dim = 512
        else:
            self.tokenizer = None
            self.clip = None
            clip_dim = 512
        
        # Load generator
        self.generator = TextConditionedGenerator(clip_dim=clip_dim)
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        
        self.generator.to(self.device)
        self.generator.eval()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Convert text to CLIP embedding."""
        if self.clip is None:
            # Fallback: random embedding for testing
            return torch.randn(1, 512, device=self.device)
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip(**inputs)
            # Use pooled output (CLS token)
            embedding = outputs.pooler_output
        
        return embedding
    
    @torch.no_grad()
    def generate(self, prompt: str, num_samples: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate 3D point cloud from text prompt.
        
        Args:
            prompt: Text description (e.g., "modern wooden chair")
            num_samples: Number of variations to generate
            seed: Random seed for reproducibility
            
        Returns:
            points: (num_samples, num_points, 3) numpy array
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        # Encode text
        text_embedding = self.encode_text(prompt)
        text_embedding = text_embedding.expand(num_samples, -1)
        
        # Generate
        points = self.generator(text_embedding)
        
        return points.cpu().numpy()
    
    @staticmethod
    def save_ply(points: np.ndarray, path: str, colors: bool = True):
        """Save point cloud as PLY file."""
        if points.ndim == 3:
            points = points[0]  # Take first sample if batched
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for point in points:
                x, y, z = point
                if colors:
                    # Blue to red gradient based on height
                    t = (y + 1) / 2
                    r, g, b = int(255 * t), 100, int(255 * (1 - t))
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        
        print(f"💾 Saved: {path}")


if __name__ == "__main__":
    # Quick test
    print("🔧 ShapeForge Text-to-3D")
    print("=" * 40)
    
    model = ShapeForgeTextTo3D()
    
    prompts = ["modern wooden chair", "office chair", "rocking chair"]
    
    for prompt in prompts:
        print(f"\n📝 Generating: '{prompt}'")
        points = model.generate(prompt, seed=42)
        
        output_path = f"outputs/chair_{prompt.replace(' ', '_')}.ply"
        Path("outputs").mkdir(exist_ok=True)
        model.save_ply(points, output_path)
