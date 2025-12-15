"""
ShapeNet Chairs Dataset Downloader

Downloads the ShapeNet chairs subset for training ShapeForge.
Uses the ShapeNet public URLs or a curated subset.
"""

import os
import sys
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve
from urllib.error import URLError

import click
from tqdm import tqdm


# Curated chair models from ShapeNet (public subset)
CURATED_CHAIRS_URL = "https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/03001627.zip"
SHAPENET_CHAIR_SYNSET = "03001627"  # ShapeNet synset ID for chairs


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: str) -> str:
    """Download a file with progress bar."""
    print(f"📥 Downloading from {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urlretrieve(url, output_path, reporthook=t.update_to)
    
    return output_path


def extract_zip(zip_path: str, output_dir: str) -> None:
    """Extract a zip file with progress."""
    print(f"📦 Extracting to {output_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        for member in tqdm(members, desc="Extracting"):
            zf.extract(member, output_dir)


def count_models(directory: str) -> int:
    """Count the number of 3D model files in a directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(('.obj', '.ply', '.off')):
                count += 1
    return count


def create_sample_data(output_dir: str, num_samples: int = 10) -> None:
    """Create sample placeholder data for testing."""
    print(f"📝 Creating {num_samples} sample placeholders for testing...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create simple placeholder files
    for i in range(num_samples):
        sample_dir = output_path / f"chair_{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Create a minimal OBJ file (just a cube as placeholder)
        obj_content = """# Sample chair placeholder
v -0.5 -0.5 -0.5
v 0.5 -0.5 -0.5
v 0.5 0.5 -0.5
v -0.5 0.5 -0.5
v -0.5 -0.5 0.5
v 0.5 -0.5 0.5
v 0.5 0.5 0.5
v -0.5 0.5 0.5
f 1 2 3 4
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
        obj_path = sample_dir / "model.obj"
        obj_path.write_text(obj_content)
    
    print(f"✅ Created {num_samples} sample models in {output_dir}")


@click.command()
@click.option('--output', '-o', default='data/raw', help='Output directory for downloaded data')
@click.option('--curated', is_flag=True, help='Download curated subset (recommended)')
@click.option('--test', is_flag=True, help='Create test data only (no download)')
@click.option('--limit', type=int, default=None, help='Limit number of samples (for testing)')
def main(output: str, curated: bool, test: bool, limit: Optional[int]):
    """
    Download ShapeNet chairs dataset for ShapeForge training.
    
    Examples:
        python download.py --curated --output data/raw
        python download.py --test --limit 10
    """
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if test:
        # Create sample data for testing
        num_samples = limit or 10
        create_sample_data(output, num_samples)
        return
    
    print("🪑 ShapeForge Data Downloader")
    print("=" * 50)
    
    if curated:
        # Download curated subset from HuggingFace
        print("\n📦 Downloading curated ShapeNet chairs subset...")
        print("   Note: This requires HuggingFace access to ShapeNet dataset")
        print("   You may need to accept the dataset terms at:")
        print("   https://huggingface.co/datasets/ShapeNet/ShapeNetCore")
        print()
        
        try:
            zip_path = output_path / "chairs.zip"
            download_with_progress(CURATED_CHAIRS_URL, str(zip_path))
            extract_zip(str(zip_path), str(output_path))
            
            # Clean up zip
            zip_path.unlink()
            
            num_models = count_models(str(output_path))
            print(f"\n✅ Downloaded {num_models} chair models to {output_path}")
            
        except URLError as e:
            print(f"\n❌ Download failed: {e}")
            print("\n📋 Alternative: Manual download instructions")
            print("   1. Visit https://shapenet.org/ and create an account")
            print("   2. Download the 'ShapeNetCore.v2' dataset")
            print("   3. Extract the '03001627' folder (chairs) to data/raw/")
            print("\n   Or use --test flag to create sample data for development")
            sys.exit(1)
    else:
        print("\n📋 ShapeNet Download Instructions")
        print("=" * 50)
        print("""
ShapeNet requires registration for full access.

Option 1: Use curated subset (recommended)
    python download.py --curated

Option 2: Full ShapeNet access
    1. Register at https://shapenet.org/
    2. Download ShapeNetCore.v2
    3. Extract '03001627' (chairs) to data/raw/

Option 3: Test with sample data
    python download.py --test --limit 100
""")
    
    print("\n🎯 Next step: Run preprocessing")
    print("   python data/preprocess.py --input data/raw --output data/processed")


if __name__ == "__main__":
    main()
