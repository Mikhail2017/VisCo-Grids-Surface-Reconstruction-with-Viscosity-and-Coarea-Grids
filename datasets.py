"""
Utilities for loading and preparing point cloud datasets for VisCo Grids.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import os
from pathlib import Path

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def find_ply_file(directory: str, preferred_names: Optional[List[str]] = None, recursive: bool = False) -> Optional[str]:
    """
    Find a PLY file in a directory, trying preferred names first.
    
    Args:
        directory: Directory to search
        preferred_names: List of preferred filenames to try first
        recursive: If True, search subdirectories as well
        
    Returns:
        Path to PLY file if found, None otherwise
    """
    if preferred_names is None:
        preferred_names = []
    
    # Try preferred names first in the directory
    for name in preferred_names:
        path = os.path.join(directory, name)
        if os.path.exists(path) and name.endswith('.ply'):
            return path
    
    # Search for any PLY file in the directory
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith('.ply'):
                return os.path.join(directory, file)
        
        # If recursive, search subdirectories
        if recursive:
            for root, dirs, files in os.walk(directory):
                # Try preferred names in subdirectories
                for name in preferred_names:
                    path = os.path.join(root, name)
                    if os.path.exists(path):
                        return path
                # Then any PLY file
                for file in files:
                    if file.endswith('.ply'):
                        return os.path.join(root, file)
    
    return None


def load_point_cloud_from_ply(filepath: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load point cloud from PLY file.
    
    Args:
        filepath: Path to PLY file
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals of shape (N, 3)
    """
    if HAS_TRIMESH:
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.PointCloud):
            points = torch.from_numpy(mesh.vertices).float()
            normals = None
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = torch.from_numpy(mesh.vertex_normals).float()
            return points, normals
        elif isinstance(mesh, trimesh.Trimesh):
            # Sample points from mesh
            points, face_indices = mesh.sample(10000, return_index=True)
            points = torch.from_numpy(points).float()
            # Compute normals from faces
            normals = None
            if mesh.vertex_normals is not None:
                normals = torch.from_numpy(mesh.vertex_normals).float()
            return points, normals
    else:
        # Simple PLY parser (basic implementation)
        points = []
        normals = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            in_vertex_section = False
            for line in lines:
                line = line.strip()
                if line == 'end_header':
                    in_vertex_section = True
                    continue
                if in_vertex_section and line:
                    parts = line.split()
                    if len(parts) >= 3:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                        if len(parts) >= 6:
                            nx, ny, nz = float(parts[3]), float(parts[4]), float(parts[5])
                            normals.append([nx, ny, nz])
        
        points = torch.tensor(points, dtype=torch.float32)
        normals = torch.tensor(normals, dtype=torch.float32) if normals else None
        return points, normals


def load_point_cloud_from_obj(filepath: str, sample_points: int = 10000) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load point cloud from OBJ file (mesh).
    
    Args:
        filepath: Path to OBJ file
        sample_points: Number of points to sample from mesh
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals of shape (N, 3)
    """
    if HAS_TRIMESH:
        mesh = trimesh.load(filepath)
        if isinstance(mesh, trimesh.Trimesh):
            points, _ = mesh.sample(sample_points, return_index=True)
            points = torch.from_numpy(points).float()
            normals = None
            if mesh.vertex_normals is not None:
                # Sample normals (simplified - use nearest vertex normal)
                normals = torch.from_numpy(mesh.vertex_normals).float()
            return points, normals
    else:
        raise ImportError("trimesh is required for OBJ loading. Install with: pip install trimesh")


def load_stanford_bunny(data_dir: str = "./data") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load Stanford Bunny point cloud.
    
    The Stanford Bunny is a classic 3D test model.
    Download from: http://graphics.stanford.edu/data/3Dscanrep/
    
    The archive structure:
    - bunny/data/ contains individual scan files (bun000.ply, bun045.ply, etc.)
    - bunny/reconstruction/ contains merged reconstructions (bun_zipper.ply, etc.)
    
    This function prefers reconstruction files as they are the final merged meshes.
    
    Args:
        data_dir: Directory containing the bunny data
    
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals
    """
    bunny_dir = os.path.join(data_dir, "bunny")
    
    # Preferred reconstruction files (final merged meshes)
    reconstruction_names = [
        "bun_zipper.ply",      # Main zippered reconstruction
        "bun_zipper_res4.ply", # Higher resolution
        "bun_zipper_res3.ply", # Medium resolution
        "bun_zipper_res2.ply", # Lower resolution
        "bun_vrip.ply",        # VRIP reconstruction (if available)
    ]
    
    # First, try reconstruction folder (preferred)
    reconstruction_dir = os.path.join(bunny_dir, "reconstruction")
    if os.path.exists(reconstruction_dir):
        bunny_path = find_ply_file(reconstruction_dir, preferred_names=reconstruction_names)
        if bunny_path:
            print(f"Loading Stanford Bunny from: {bunny_path}")
            return load_point_cloud_from_ply(bunny_path)
    
    # Try root bunny directory
    bunny_path = find_ply_file(bunny_dir, preferred_names=reconstruction_names)
    if bunny_path:
        print(f"Loading Stanford Bunny from: {bunny_path}")
        return load_point_cloud_from_ply(bunny_path)
    
    # Search recursively (will find files in data/ subdirectory too)
    bunny_path = find_ply_file(bunny_dir, preferred_names=reconstruction_names, recursive=True)
    if bunny_path:
        print(f"Loading Stanford Bunny from: {bunny_path}")
        return load_point_cloud_from_ply(bunny_path)
    
    # If still not found, provide helpful error message
    raise FileNotFoundError(
        f"Stanford Bunny PLY file not found in {bunny_dir}\n"
        f"Expected structure:\n"
        f"  {bunny_dir}/\n"
        f"    reconstruction/\n"
        f"      bun_zipper.ply  (preferred)\n"
        f"    data/\n"
        f"      bun000.ply, bun045.ply, ... (individual scans)\n"
        f"\nDownload from: http://graphics.stanford.edu/data/3Dscanrep/\n"
        f"Extract bunny.tar.gz to {data_dir}/"
    )


def load_armadillo(data_dir: str = "./data") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load Armadillo point cloud from Stanford 3D Scanning Repository.
    
    Download from: http://graphics.stanford.edu/data/3Dscanrep/
    
    Args:
        data_dir: Directory containing the armadillo data
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals
    """
    armadillo_path = os.path.join(data_dir, "armadillo", "Armadillo.ply")
    if not os.path.exists(armadillo_path):
        raise FileNotFoundError(
            f"Armadillo not found at {armadillo_path}\n"
            "Download from: http://graphics.stanford.edu/data/3Dscanrep/"
        )
    return load_point_cloud_from_ply(armadillo_path)


def load_dragon(data_dir: str = "./data") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load Dragon point cloud from Stanford 3D Scanning Repository.
    
    Download from: http://graphics.stanford.edu/data/3Dscanrep/
    
    Args:
        data_dir: Directory containing the dragon data
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals
    """
    dragon_path = os.path.join(data_dir, "dragon", "dragon.ply")
    if not os.path.exists(dragon_path):
        raise FileNotFoundError(
            f"Dragon not found at {dragon_path}\n"
            "Download from: http://graphics.stanford.edu/data/3Dscanrep/"
        )
    return load_point_cloud_from_ply(dragon_path)


def load_shapenet_model(
    model_path: str,
    sample_points: int = 10000
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a ShapeNet model (OBJ format).
    
    ShapeNet: https://www.shapenet.org/
    
    Args:
        model_path: Path to ShapeNet OBJ file
        sample_points: Number of points to sample
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals
    """
    return load_point_cloud_from_obj(model_path, sample_points=sample_points)


def load_modelnet_model(
    model_path: str,
    sample_points: int = 10000
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load a ModelNet model (OBJ or OFF format).
    
    ModelNet: https://modelnet.cs.princeton.edu/
    
    Args:
        model_path: Path to ModelNet model file
        sample_points: Number of points to sample
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals
    """
    if model_path.endswith('.obj'):
        return load_point_cloud_from_obj(model_path, sample_points=sample_points)
    elif model_path.endswith('.off') or model_path.endswith('.ply'):
        return load_point_cloud_from_ply(model_path)
    else:
        raise ValueError(f"Unsupported file format: {model_path}")


def normalize_point_cloud(
    points: torch.Tensor,
    center: bool = True,
    scale: bool = True,
    margin: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize point cloud to unit cube [0, 1]^3.
    
    Args:
        points: Point cloud of shape (N, 3)
        center: Whether to center the point cloud
        scale: Whether to scale to unit cube
        margin: Margin to leave on each side
        
    Returns:
        normalized_points: Points in [margin, 1-margin]^3
        center_offset: Center offset applied
        scale_factor: Scale factor applied
    """
    points = points.clone()
    center_offset = torch.zeros(3, device=points.device)
    scale_factor = torch.ones(1, device=points.device)
    
    if center:
        center_offset = points.mean(dim=0)
        points = points - center_offset
    
    if scale:
        min_vals = points.min(dim=0)[0]
        max_vals = points.max(dim=0)[0]
        scale_factor = (max_vals - min_vals).max()
        if scale_factor > 0:
            points = points / scale_factor
    
    # Scale to [margin, 1-margin]^3
    points = points * (1 - 2 * margin) + 0.5
    
    return points, center_offset, scale_factor


def downsample_point_cloud(
    points: torch.Tensor,
    num_points: int,
    method: str = 'random'
) -> torch.Tensor:
    """
    Downsample point cloud to specified number of points.
    
    Args:
        points: Point cloud of shape (N, 3)
        num_points: Target number of points
        method: Downsampling method ('random' or 'fps' for farthest point sampling)
        
    Returns:
        Downsampled points of shape (num_points, 3)
    """
    if len(points) <= num_points:
        return points
    
    if method == 'random':
        indices = torch.randperm(len(points))[:num_points]
        return points[indices]
    elif method == 'fps':
        # Farthest Point Sampling (simplified)
        if not HAS_SKLEARN:
            raise ImportError("sklearn required for FPS. Using random sampling instead.")
            return downsample_point_cloud(points, num_points, method='random')
        
        # Simple FPS implementation
        selected = [0]
        points_np = points.cpu().numpy()
        
        for _ in range(num_points - 1):
            distances = []
            for i in range(len(points_np)):
                if i not in selected:
                    min_dist = min([np.linalg.norm(points_np[i] - points_np[s]) for s in selected])
                    distances.append((min_dist, i))
            if distances:
                _, idx = max(distances)
                selected.append(idx)
        
        return points[selected]
    else:
        raise ValueError(f"Unknown method: {method}")


def add_noise_to_point_cloud(
    points: torch.Tensor,
    noise_level: float = 0.01
) -> torch.Tensor:
    """
    Add Gaussian noise to point cloud.
    
    Args:
        points: Point cloud of shape (N, 3)
        noise_level: Standard deviation of noise (relative to point cloud scale)
        
    Returns:
        Noisy point cloud
    """
    scale = (points.max(dim=0)[0] - points.min(dim=0)[0]).max()
    noise = torch.randn_like(points) * noise_level * scale
    return points + noise


# Dataset information
DATASET_INFO = {
    "stanford_3d": {
        "name": "Stanford 3D Scanning Repository",
        "url": "http://graphics.stanford.edu/data/3Dscanrep/",
        "description": "Classic 3D models: Bunny, Dragon, Armadillo, Buddha, etc.",
        "format": "PLY",
        "note": "Mentioned in the VisCo Grids paper"
    },
    "shapenet": {
        "name": "ShapeNet",
        "url": "https://www.shapenet.org/",
        "description": "Large-scale 3D shape dataset with 3M+ models",
        "format": "OBJ",
        "note": "Requires registration"
    },
    "modelnet": {
        "name": "ModelNet",
        "url": "https://modelnet.cs.princeton.edu/",
        "description": "3D object recognition dataset (ModelNet10, ModelNet40)",
        "format": "OFF/OBJ",
        "note": "Popular benchmark dataset"
    },
    "abc_dataset": {
        "name": "ABC Dataset",
        "url": "https://deep-geometry.github.io/abc-dataset/",
        "description": "CAD models for geometric deep learning",
        "format": "OBJ/STEP",
        "note": "High-quality CAD models"
    },
    "scannet": {
        "name": "ScanNet",
        "url": "http://www.scan-net.org/",
        "description": "Real-world indoor scene scans",
        "format": "PLY",
        "note": "Real-world scans, may need preprocessing"
    },
    "eth_3d": {
        "name": "ETH 3D",
        "url": "https://www.eth3d.net/",
        "description": "Multi-view stereo dataset",
        "format": "PLY",
        "note": "Real-world outdoor scenes"
    }
}


def print_dataset_info():
    """Print information about available datasets."""
    print("Available Datasets for VisCo Grids Experiments:\n")
    for key, info in DATASET_INFO.items():
        print(f"{key.upper()}:")
        print(f"  Name: {info['name']}")
        print(f"  URL: {info['url']}")
        print(f"  Description: {info['description']}")
        print(f"  Format: {info['format']}")
        print(f"  Note: {info['note']}")
        print()


if __name__ == "__main__":
    print_dataset_info()

