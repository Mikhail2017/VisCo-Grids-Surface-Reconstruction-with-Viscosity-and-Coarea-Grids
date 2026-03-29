"""
Example script demonstrating VisCo Grids with real datasets.
"""

import torch
import numpy as np
from pathlib import Path
from train import train_visco_grids, estimate_normals
from visco_grids import VisCoGrids
from datasets import (
    load_point_cloud_from_ply,
    load_point_cloud_from_obj,
    normalize_point_cloud,
    downsample_point_cloud,
    print_dataset_info,
    DATASET_INFO
)
from mesh_utils import save_mesh_to_obj


def example_stanford_bunny():
    """Example using Stanford Bunny."""
    print("=" * 60)
    print("Example: Stanford Bunny")
    print("=" * 60)
    
    # Load point cloud
    try:
        from datasets import load_stanford_bunny
        points, normals = load_stanford_bunny(data_dir="data")
        print(f"Loaded {len(points)} points")
    except FileNotFoundError as e:
        print("Stanford Bunny not found.")
        print(str(e))
        print("\nInstructions:")
        print("1. Download bunny.tar.gz from: http://graphics.stanford.edu/data/3Dscanrep/")
        print("2. Extract the archive:")
        print("   tar -xzf bunny.tar.gz")
        print("3. Place the extracted bunny/ folder in data/ directory")
        print("   (or update the data_dir parameter)")
        return
    
    # Normalize
    points_normalized, center, scale = normalize_point_cloud(points)
    
    # Downsample if too many points
    if len(points_normalized) > 5000:
        points_normalized = downsample_point_cloud(points_normalized, 5000)
        if normals is not None:
            # Re-estimate normals after downsampling
            normals = None
    
    # Estimate normals if not available
    if normals is None:
        print("Estimating normals...")
        normals = estimate_normals(points_normalized, k=10)
    
    # Train
    print("\nTraining VisCo Grids...")
    model = train_visco_grids(
        points=points_normalized,
        normals=normals,
        initial_resolution=64,
        final_resolution=128,
        epochs_per_resolution=(2, 2),
        iterations_per_epoch=1000,
        verbose=True
    )
    
    # Extract mesh
    print("\nExtracting mesh...")
    try:
        vertices, faces = model.extract_mesh(level=0.0)
        print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Save mesh
        save_mesh_to_obj(vertices, faces, "bunny_reconstructed.obj")
        print("Saved mesh to 'bunny_reconstructed.obj'")
    except Exception as e:
        print(f"Error extracting mesh: {e}")


def example_custom_point_cloud(filepath: str):
    """Example with custom point cloud file."""
    print("=" * 60)
    print(f"Example: Custom Point Cloud - {filepath}")
    print("=" * 60)
    
    # Load based on file extension
    if filepath.endswith('.ply'):
        points, normals = load_point_cloud_from_ply(filepath)
    elif filepath.endswith('.obj'):
        points, normals = load_point_cloud_from_obj(filepath, sample_points=5000)
    else:
        print(f"Unsupported file format: {filepath}")
        return
    
    print(f"Loaded {len(points)} points")
    
    # Normalize
    points_normalized, center, scale = normalize_point_cloud(points)
    
    # Estimate normals if not available
    if normals is None:
        print("Estimating normals...")
        normals = estimate_normals(points_normalized, k=10)
    
    # Train
    print("\nTraining VisCo Grids...")
    model = train_visco_grids(
        points=points_normalized,
        normals=normals,
        initial_resolution=64,
        final_resolution=128,
        epochs_per_resolution=(2, 2),
        iterations_per_epoch=1000,
        verbose=True
    )
    
    # Extract and save mesh
    print("\nExtracting mesh...")
    try:
        vertices, faces = model.extract_mesh(level=0.0)
        print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        output_name = Path(filepath).stem + "_reconstructed.obj"
        save_mesh_to_obj(vertices, faces, output_name)
        print(f"Saved mesh to '{output_name}'")
    except Exception as e:
        print(f"Error extracting mesh: {e}")


def main():
    """Main function with dataset selection."""
    print("VisCo Grids - Dataset Examples")
    print("=" * 60)
    print()
    
    # Print dataset information
    print_dataset_info()
    
    print("\n" + "=" * 60)
    print("Quick Start Examples:")
    print("=" * 60)
    print()
    print("1. Stanford Bunny (if downloaded):")
    print("   python example_with_dataset.py --bunny")
    print()
    print("2. Custom point cloud file:")
    print("   python example_with_dataset.py --file path/to/pointcloud.ply")
    print()
    print("3. List available datasets:")
    print("   python example_with_dataset.py --list")
    print()
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--bunny':
            example_stanford_bunny()
        elif sys.argv[1] == '--file' and len(sys.argv) > 2:
            example_custom_point_cloud(sys.argv[2])
        elif sys.argv[1] == '--list':
            print_dataset_info()
        else:
            print("Unknown option. Use --bunny, --file <path>, or --list")
    else:
        print("Run with --bunny, --file <path>, or --list")


if __name__ == "__main__":
    main()

