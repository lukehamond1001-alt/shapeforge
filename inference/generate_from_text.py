"""
Generate 3D chairs from text prompts.

Usage:
    python generate_from_text.py --prompt "modern wooden chair"
    python generate_from_text.py --prompt "office chair" --num-samples 5
"""

import sys
from pathlib import Path

import numpy as np
import torch
import click

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.text_to_3d import ShapeForgeTextTo3D


@click.command()
@click.option('--checkpoint', '-c', default='checkpoints/shapeforge-text2shape-best.pt', help='Model checkpoint')
@click.option('--prompt', '-p', required=True, help='Text prompt (e.g., "modern wooden chair")')
@click.option('--num-samples', '-n', default=1, help='Number of samples to generate')
@click.option('--output', '-o', default='outputs', help='Output directory')
@click.option('--seed', '-s', type=int, default=None, help='Random seed')
def main(checkpoint: str, prompt: str, num_samples: int, output: str, seed: int):
    """Generate 3D chairs from text descriptions."""
    
    print("🔮 ShapeForge Text-to-3D Generator")
    print("=" * 50)
    print(f"📝 Prompt: '{prompt}'")
    print(f"🔢 Samples: {num_samples}")
    
    # Create output dir
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint_path = checkpoint if Path(checkpoint).exists() else None
    if checkpoint_path:
        print(f"📂 Loading: {checkpoint}")
    else:
        print("⚠️ No checkpoint found, using untrained model (for testing)")
    
    model = ShapeForgeTextTo3D(checkpoint_path=checkpoint_path)
    
    # Generate
    print(f"\n⚡ Generating...")
    points = model.generate(prompt, num_samples=num_samples, seed=seed)
    
    # Save
    prompt_slug = prompt.replace(' ', '_').replace('"', '').replace("'", '')[:30]
    
    for i in range(num_samples):
        filename = f"chair_{prompt_slug}_{i+1:02d}.ply"
        filepath = output_path / filename
        model.save_ply(points[i] if num_samples > 1 else points, str(filepath))
    
    print(f"\n🎉 Done! Generated {num_samples} chair(s)")
    print(f"   Output: {output_path}/")
    print("\n   View with: MeshLab, Blender, or online PLY viewer")


if __name__ == "__main__":
    main()
