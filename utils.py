"""
Utility functions for VisCo Grids.
"""

import torch
import numpy as np
from typing import Optional, Tuple

# Import mesh utilities if available
try:
    from .mesh_utils import save_mesh_to_obj, load_mesh_from_obj, compute_vertex_normals
except ImportError:
    try:
        from mesh_utils import save_mesh_to_obj, load_mesh_from_obj, compute_vertex_normals
    except ImportError:
        save_mesh_to_obj = None
        load_mesh_from_obj = None
        compute_vertex_normals = None


def load_point_cloud_from_numpy(
    filepath: str,
    has_normals: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Load point cloud from numpy file.
    
    Args:
        filepath: Path to .npy file
        has_normals: Whether file contains normals (shape should be (N, 6) if True)
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Optional normals of shape (N, 3)
    """
    data = np.load(filepath)
    
    if has_normals:
        if data.shape[1] != 6:
            raise ValueError("Expected shape (N, 6) when has_normals=True")
        points = torch.from_numpy(data[:, :3]).float()
        normals = torch.from_numpy(data[:, 3:]).float()
        return points, normals
    else:
        if data.shape[1] != 3:
            raise ValueError("Expected shape (N, 3) when has_normals=False")
        points = torch.from_numpy(data).float()
        return points, None


def save_point_cloud_to_numpy(
    points: torch.Tensor,
    filepath: str,
    normals: Optional[torch.Tensor] = None
):
    """
    Save point cloud to numpy file.
    
    Args:
        points: Point cloud of shape (N, 3)
        filepath: Path to save .npy file
        normals: Optional normals of shape (N, 3)
    """
    if normals is not None:
        data = torch.cat([points, normals], dim=1).cpu().numpy()
    else:
        data = points.cpu().numpy()
    np.save(filepath, data)


def normalize_point_cloud(
    points: torch.Tensor,
    margin: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize point cloud to unit cube [0, 1]^3 with margin.
    
    Args:
        points: Point cloud of shape (N, 3)
        margin: Margin to leave on each side (default: 0.05)
        
    Returns:
        normalized_points: Points in [margin, 1-margin]^3
        center: Original center
        scale: Original scale
    """
    min_vals = points.min(dim=0)[0]
    max_vals = points.max(dim=0)[0]
    center = (min_vals + max_vals) / 2
    scale = (max_vals - min_vals).max()
    
    # Normalize to [0, 1] with margin
    normalized = (points - center) / scale
    normalized = normalized * (1 - 2 * margin) + 0.5
    
    return normalized, center, scale


def denormalize_point_cloud(
    normalized_points: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    margin: float = 0.05
) -> torch.Tensor:
    """
    Denormalize point cloud from unit cube back to original coordinates.
    
    Args:
        normalized_points: Points in [margin, 1-margin]^3
        center: Original center
        scale: Original scale
        margin: Margin used in normalization
        
    Returns:
        Original coordinates
    """
    # Reverse normalization
    points = (normalized_points - 0.5) / (1 - 2 * margin)
    points = points * scale + center
    return points

