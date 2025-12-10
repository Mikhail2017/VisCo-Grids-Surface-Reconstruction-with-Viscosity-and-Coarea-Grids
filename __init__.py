"""
VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids
"""

from .visco_grids import VisCoGrids
from .train import train_visco_grids, estimate_normals
from .utils import (
    load_point_cloud_from_numpy,
    save_point_cloud_to_numpy,
    normalize_point_cloud,
    denormalize_point_cloud
)

try:
    from .mesh_utils import save_mesh_to_obj, load_mesh_from_obj, compute_vertex_normals
except ImportError:
    try:
        from mesh_utils import save_mesh_to_obj, load_mesh_from_obj, compute_vertex_normals
    except ImportError:
        save_mesh_to_obj = None
        load_mesh_from_obj = None
        compute_vertex_normals = None

__version__ = "1.0.0"
__all__ = [
    'VisCoGrids',
    'train_visco_grids',
    'estimate_normals',
    'load_point_cloud_from_numpy',
    'save_point_cloud_to_numpy',
    'normalize_point_cloud',
    'denormalize_point_cloud',
    'save_mesh_to_obj',
    'load_mesh_from_obj',
    'compute_vertex_normals'
]

