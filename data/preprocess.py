"""
Point Cloud Preprocessing for ShapeForge

Converts ShapeNet mesh files (OBJ/PLY) to normalized point clouds
suitable for training with Shap-E.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import json

import numpy as np
import click
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    print("❌ trimesh not installed. Run: pip install trimesh")
    sys.exit(1)


def load_mesh(mesh_path: str) -> Optional[trimesh.Trimesh]:
    """Load a mesh file (OBJ, PLY, OFF, etc.)."""
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # If it's a scene, try to get the geometry
            if len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            else:
                return None
        return mesh
    except Exception as e:
        print(f"   ⚠️ Failed to load {mesh_path}: {e}")
        return None


def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int = 4096) -> np.ndarray:
    """Sample points from mesh surface."""
    try:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points
    except Exception as e:
        print(f"   ⚠️ Sampling failed: {e}")
        return None


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Normalize point cloud to unit sphere centered at origin.
    
    Args:
        points: (N, 3) array of points
        
    Returns:
        Normalized points in [-1, 1] range
    """
    # Center at origin
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to fit in unit sphere
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return points.astype(np.float32)


def find_mesh_files(directory: str) -> list:
    """Find all mesh files in a directory."""
    mesh_files = []
    supported_ext = ('.obj', '.ply', '.off', '.stl')
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(supported_ext):
                mesh_files.append(os.path.join(root, f))
    
    return mesh_files


def process_single_mesh(
    mesh_path: str,
    output_dir: str,
    num_points: int = 4096
) -> Optional[str]:
    """
    Process a single mesh file to point cloud.
    
    Returns output path if successful, None otherwise.
    """
    # Load mesh
    mesh = load_mesh(mesh_path)
    if mesh is None:
        return None
    
    # Sample points
    points = sample_point_cloud(mesh, num_points)
    if points is None:
        return None
    
    # Normalize
    points = normalize_point_cloud(points)
    
    # Generate output filename
    mesh_path = Path(mesh_path)
    # Use parent directory name as the model ID
    model_id = mesh_path.parent.name
    if model_id in ('', '.'):
        model_id = mesh_path.stem
    
    output_path = Path(output_dir) / f"{model_id}.npy"
    
    # Save
    np.save(str(output_path), points)
    
    return str(output_path)


@click.command()
@click.option('--input', '-i', 'input_dir', default='data/raw', help='Input directory with mesh files')
@click.option('--output', '-o', 'output_dir', default='data/processed', help='Output directory for point clouds')
@click.option('--num-points', '-n', default=4096, help='Number of points per cloud')
@click.option('--limit', type=int, default=None, help='Limit number of files to process')
def main(input_dir: str, output_dir: str, num_points: int, limit: Optional[int]):
    """
    Preprocess ShapeNet meshes to normalized point clouds.
    
    Examples:
        python preprocess.py --input data/raw --output data/processed
        python preprocess.py --limit 100  # Process first 100 only
    """
    print("🔧 ShapeForge Preprocessor")
    print("=" * 50)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all mesh files
    print(f"\n📂 Scanning {input_dir} for mesh files...")
    mesh_files = find_mesh_files(input_dir)
    
    if not mesh_files:
        print(f"❌ No mesh files found in {input_dir}")
        print("   Run data/download.py first, or check the input path")
        sys.exit(1)
    
    print(f"   Found {len(mesh_files)} mesh files")
    
    # Apply limit if specified
    if limit:
        mesh_files = mesh_files[:limit]
        print(f"   Processing first {limit} files")
    
    # Process each mesh
    print(f"\n⚙️ Processing to {num_points}-point clouds...")
    
    successful = 0
    failed = 0
    
    for mesh_path in tqdm(mesh_files, desc="Processing"):
        result = process_single_mesh(mesh_path, output_dir, num_points)
        if result:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n✅ Processing complete!")
    print(f"   Successful: {successful}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_dir}/")
    
    # Save metadata
    metadata = {
        "num_samples": successful,
        "num_points": num_points,
        "source_dir": input_dir,
        "files_processed": successful + failed
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {metadata_path}")
    
    print("\n🎯 Next step: Train the model")
    print("   python model/train.py --data data/processed")


if __name__ == "__main__":
    main()
