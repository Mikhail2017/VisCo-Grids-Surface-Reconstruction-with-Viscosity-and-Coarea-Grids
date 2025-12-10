"""
VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids
Implementation based on the paper by Pumarola et al., NeurIPS 2022
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Install with 'pip install scikit-image' for marching cubes mesh extraction.")
try:
    from sklearn.neighbors import KDTree
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not found. Install with 'pip install scikit-learn' for SDF initialization.")


class VisCoGrids(nn.Module):
    """
    VisCo Grids: Grid-based SDF estimation with Viscosity and Coarea priors.
    
    The method uses a 3D voxel grid to represent a Signed Distance Function (SDF)
    and optimizes it using:
    - Data loss: point and normal constraints
    - Viscosity loss: encourages SDF-like behavior
    - Coarea loss: minimizes surface area
    """
    
    def __init__(
        self,
        grid_resolution: int = 64,
        lambda_p: float = 0.1,
        lambda_n: float = 1e-5,
        lambda_v: float = 1e-4,
        lambda_c: float = 1e-6,
        epsilon: float = 1e-2,
        beta: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize VisCo Grids.
        
        Args:
            grid_resolution: Resolution of the 3D grid (n x n x n)
            lambda_p: Weight for point loss
            lambda_n: Weight for normal loss
            lambda_v: Weight for viscosity loss
            lambda_c: Weight for coarea loss
            epsilon: Viscosity parameter
            beta: Coarea parameter (Laplace distribution scale)
            device: Device to run on
        """
        super().__init__()
        self.grid_resolution = grid_resolution
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        self.lambda_v = lambda_v
        self.lambda_c = lambda_c
        self.epsilon = epsilon
        self.beta = beta
        self.device = device
        self.h = 1.0 / grid_resolution  # Grid spacing
        
        # Initialize SDF grid values (learnable parameters)
        self.sdf_grid = nn.Parameter(
            torch.zeros(grid_resolution, grid_resolution, grid_resolution, device=device)
        )
        
        # Active voxel mask (for pruning)
        self.register_buffer('active_mask', torch.ones(
            grid_resolution, grid_resolution, grid_resolution, dtype=torch.bool, device=device
        ))
    
    def normalize_point_cloud(self, points: torch.Tensor) -> torch.Tensor:
        """
        Normalize point cloud to unit cube [0, 1]^3.
        
        Args:
            points: Point cloud of shape (N, 3)
            
        Returns:
            Normalized points in [0, 1]^3
        """
        points = points.clone()
        # Center and scale to fit in [0, 1]^3
        min_vals = points.min(dim=0)[0]
        max_vals = points.max(dim=0)[0]
        center = (min_vals + max_vals) / 2
        scale = (max_vals - min_vals).max()
        
        points = (points - center) / scale
        points = points * 0.45 + 0.5  # Scale to [0.05, 0.95] with some margin
        
        return points
    
    def initialize_from_pointcloud(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        k_neighbors: int = 5
    ):
        """
        Initialize SDF grid values based on approximate distances to point cloud.
        Uses KD tree for efficient nearest neighbor search and normal orientation for sign.
        
        Args:
            points: Point cloud of shape (N, 3) in [0, 1]^3 (normalized)
            normals: Optional normals of shape (N, 3)
            k_neighbors: Number of nearest neighbors to consider for sign determination
        """
        if not HAS_SKLEARN:
            print("Warning: scikit-learn not available. Using zero initialization.")
            return
        
        # Convert to numpy for KDTree
        points_np = points.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy() if normals is not None else None
        
        # Build KDTree
        tree = KDTree(points_np)
        
        # Create grid of voxel centers
        n = self.grid_resolution
        coords = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
        x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
        grid_centers = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Find nearest neighbors for each grid center
        k_actual = min(k_neighbors, len(points_np))
        distances, indices = tree.query(grid_centers, k=k_actual)
        
        # Handle case where k=1 returns 1D array
        if k_actual == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)
        
        # Compute SDF values
        sdf_values = np.zeros(len(grid_centers))
        
        for i in range(len(grid_centers)):
            center = grid_centers[i]
            nearest_idx = indices[i, 0]  # Closest point
            dist = distances[i, 0]  # Distance to closest point
            
            if normals_np is not None:
                # Use normal orientation to determine sign
                # Vector from grid center to nearest point
                to_point = points_np[nearest_idx] - center
                
                # Get average normal from k nearest neighbors
                k_nearest_indices = indices[i, :k_actual]
                avg_normal = normals_np[k_nearest_indices].mean(axis=0)
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
                
                # Determine sign:
                # - Normal points outward (away from inside)
                # - If grid center is outside: to_point goes toward surface (opposite to normal) → dot < 0 → SDF positive
                # - If grid center is inside: to_point goes toward surface (same as normal) → dot > 0 → SDF negative
                # So we invert the sign of the dot product
                sign = -np.sign(np.dot(to_point, avg_normal))
                
                # If sign is zero or ambiguous, use majority vote from k neighbors
                if abs(sign) < 0.1:
                    to_points = points_np[k_nearest_indices] - center
                    dots = np.dot(to_points, avg_normal)
                    sign = -np.sign(dots.mean())
                    if abs(sign) < 0.1:
                        sign = 1.0  # Default to positive (outside)
            else:
                # No normals: use a simple heuristic
                # For points far from the surface, use positive sign
                # This is a simple approximation
                sign = 1.0
            
            sdf_values[i] = sign * dist
        
        # Reshape to grid
        sdf_grid_np = sdf_values.reshape(n, n, n)
        
        # Convert to torch and update parameter
        with torch.no_grad():
            self.sdf_grid.data = torch.from_numpy(sdf_grid_np).float().to(self.device)
    
    def trilinear_interpolate(
        self, 
        points: torch.Tensor, 
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Trilinear interpolation of grid values at given points.
        
        Args:
            points: Points of shape (N, 3) in [0, 1]^3
            grid: Optional grid to interpolate (defaults to self.sdf_grid)
            
        Returns:
            Interpolated values of shape (N,)
        """
        if grid is None:
            grid = self.sdf_grid
        
        # Ensure points are on the same device as grid
        points = points.to(grid.device)
        
        # Clamp points to valid range
        points = torch.clamp(points, 0, 1 - 1e-6)
        
        # Convert to grid coordinates
        grid_coords = points / self.h
        grid_coords = torch.clamp(grid_coords, 0, self.grid_resolution - 1)
        
        # Get integer and fractional parts
        i0 = grid_coords.floor().long()
        i1 = (i0 + 1).clamp(max=self.grid_resolution - 1)
        
        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
        
        # Fractional parts
        xd = grid_coords[:, 0] - x0.float()
        yd = grid_coords[:, 1] - y0.float()
        zd = grid_coords[:, 2] - z0.float()
        
        # Get corner values
        c000 = grid[x0, y0, z0]
        c001 = grid[x0, y0, z1]
        c010 = grid[x0, y1, z0]
        c011 = grid[x0, y1, z1]
        c100 = grid[x1, y0, z0]
        c101 = grid[x1, y0, z1]
        c110 = grid[x1, y1, z0]
        c111 = grid[x1, y1, z1]
        
        # Interpolate along x
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        # Interpolate along y
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        # Interpolate along z
        values = c0 * (1 - zd) + c1 * zd
        
        return values
    
    def compute_gradient(
        self, 
        points: torch.Tensor, 
        grid: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute gradient of SDF at given points using trilinear interpolation.
        
        Args:
            points: Points of shape (N, 3) in [0, 1]^3
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            Gradients of shape (N, 3)
        """
        if grid is None:
            grid = self.sdf_grid
        
        # Ensure points are on the same device as grid
        points = points.to(grid.device)
        
        # Ensure points require gradients
        points = points.clone()
        if not points.requires_grad:
            points = points.requires_grad_(True)
        
        values = self.trilinear_interpolate(points, grid)
        
        # Compute gradients
        if values.requires_grad:
            gradients = torch.autograd.grad(
                outputs=values,
                inputs=points,
                grad_outputs=torch.ones_like(values),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
        else:
            # Fallback: use finite differences approximation
            eps = 1e-5
            gradients = torch.zeros_like(points)
            for i in range(3):
                points_plus = points.clone()
                points_plus[:, i] += eps
                values_plus = self.trilinear_interpolate(points_plus, grid)
                gradients[:, i] = (values_plus - values) / eps
        
        return gradients
    
    def compute_gradient_at_voxel_centers(self, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute gradient at voxel centers using trilinear basis functions.
        Based on Appendix A of the paper.
        
        Args:
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            Gradients of shape (N, 3) where N is number of active voxels
        """
        if grid is None:
            grid = self.sdf_grid
        
        # Get active voxel centers
        active_indices = torch.nonzero(self.active_mask, as_tuple=False)
        if len(active_indices) == 0:
            return torch.zeros(0, 3, device=self.device)
        
        # Voxel centers in [0, 1]^3
        centers = (active_indices.float() + 0.5) * self.h
        
        # Compute gradient using trilinear interpolation
        gradients = self.compute_gradient(centers, grid)
        
        return gradients
    
    def compute_finite_differences(self, grid: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute finite differences for gradient and Laplacian at grid nodes.
        Uses symmetric finite differences as in the paper.
        
        Args:
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            gradients: Gradient at grid nodes, shape (n, n, n, 3)
            laplacians: Laplacian at grid nodes, shape (n, n, n)
        """
        if grid is None:
            grid = self.sdf_grid
        
        n = self.grid_resolution
        h = self.h
        
        # Pad grid for boundary handling
        padded = torch.zeros(n + 2, n + 2, n + 2, device=grid.device)
        padded[1:-1, 1:-1, 1:-1] = grid
        
        # First-order derivatives (symmetric finite differences)
        dx = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2 * h)
        dy = (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2 * h)
        dz = (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2 * h)
        
        gradients = torch.stack([dx, dy, dz], dim=-1)
        
        # Second-order derivatives (for Laplacian)
        d2x = (padded[2:, 1:-1, 1:-1] - 2 * grid + padded[:-2, 1:-1, 1:-1]) / (h ** 2)
        d2y = (padded[1:-1, 2:, 1:-1] - 2 * grid + padded[1:-1, :-2, 1:-1]) / (h ** 2)
        d2z = (padded[1:-1, 1:-1, 2:] - 2 * grid + padded[1:-1, 1:-1, :-2]) / (h ** 2)
        
        laplacians = d2x + d2y + d2z
        
        return gradients, laplacians
    
    def data_loss(
        self, 
        points: torch.Tensor, 
        normals: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute data loss: point loss + normal loss.
        
        Args:
            points: Point cloud of shape (N, 3)
            normals: Optional normals of shape (N, 3)
            
        Returns:
            point_loss: Point loss (L_p)
            normal_loss: Normal loss (L_n)
        """
        # Point loss: f(q_k) should be close to 0
        sdf_values = self.trilinear_interpolate(points)
        point_loss = (sdf_values ** 2).mean()
        
        # Normal loss: gradient should match normals
        normal_loss = torch.tensor(0.0, device=self.device)
        if normals is not None:
            # Ensure normals are on the same device as the model
            normals = normals.to(self.device)
            gradients = self.compute_gradient(points)
            normal_loss = ((gradients - normals) ** 2).sum(dim=1).mean()
        
        return point_loss, normal_loss
    
    def viscosity_loss(self, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute viscosity loss to encourage SDF-like behavior.
        
        L_viscosity = (1/N) * sum_I [ (||∇f(p_I)|| - 1) * sign(f(p_I)) - ε * Δf(p_I) ]^2
        
        Args:
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            Viscosity loss
        """
        if grid is None:
            grid = self.sdf_grid
        
        gradients, laplacians = self.compute_finite_differences(grid)
        
        # Compute gradient norms
        grad_norms = torch.norm(gradients, dim=-1)
        
        # Viscosity loss term
        sign_f = torch.sign(grid)
        viscosity_term = (grad_norms - 1.0) * sign_f - self.epsilon * laplacians
        
        # Average over all grid nodes
        loss = (viscosity_term ** 2).mean()
        
        return loss
    
    def laplace_cdf(self, s: torch.Tensor) -> torch.Tensor:
        """
        Centered Laplace CDF: Ψ_β(s)
        
        Ψ_β(s) = { 1 - (1/2) * exp(-s/β)  if s ≤ 0
                 { (1/2) * exp(s/β)       if s ≥ 0
        
        Args:
            s: Input values
            
        Returns:
            CDF values
        """
        mask_neg = s <= 0
        result = torch.zeros_like(s)
        result[mask_neg] = 1.0 - 0.5 * torch.exp(s[mask_neg] / self.beta)
        result[~mask_neg] = 0.5 * torch.exp(-s[~mask_neg] / self.beta)
        return result
    
    def laplace_pdf(self, s: torch.Tensor) -> torch.Tensor:
        """
        Laplace PDF: Φ_β(s) = (1/(2β)) * exp(-|s|/β)
        
        Args:
            s: Input values
            
        Returns:
            PDF values
        """
        return (1.0 / (2.0 * self.beta)) * torch.exp(-torch.abs(s) / self.beta)
    
    def coarea_loss(self, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute coarea loss to minimize surface area.
        
        L_coarea = (1/N) * sum_I [ Φ_β(-f(w_I)) * ||∇f(w_I)|| ]
        
        where w_I are voxel centers.
        
        Args:
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            Coarea loss
        """
        if grid is None:
            grid = self.sdf_grid
        
        # Get active voxel centers
        active_indices = torch.nonzero(self.active_mask, as_tuple=False)
        if len(active_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Voxel centers in [0, 1]^3
        centers = (active_indices.float() + 0.5) * self.h
        
        # Compute SDF values and gradients at centers
        sdf_values = self.trilinear_interpolate(centers, grid)
        gradients = self.compute_gradient(centers, grid)
        grad_norms = torch.norm(gradients, dim=1)
        
        # Coarea loss: Φ_β(-f) * ||∇f||
        phi_beta = self.laplace_pdf(-sdf_values)
        loss = (phi_beta * grad_norms).mean()
        
        return loss
    
    def total_loss(
        self, 
        points: torch.Tensor, 
        normals: Optional[torch.Tensor] = None,
        grid: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss: L = L_data + L_prior
        
        Args:
            points: Point cloud of shape (N, 3)
            normals: Optional normals of shape (N, 3)
            grid: Optional grid to use (defaults to self.sdf_grid)
            
        Returns:
            total_loss: Total loss
            loss_dict: Dictionary with individual loss components
        """
        # Data loss
        point_loss, normal_loss = self.data_loss(points, normals)
        data_loss = self.lambda_p * point_loss + self.lambda_n * normal_loss
        
        # Prior loss
        viscosity_loss = self.viscosity_loss(grid)
        coarea_loss = self.coarea_loss(grid)
        prior_loss = self.lambda_v * viscosity_loss + self.lambda_c * coarea_loss
        
        # Total loss
        total_loss = data_loss + prior_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'point': point_loss.item(),
            'normal': normal_loss.item(),
            'viscosity': viscosity_loss.item(),
            'coarea': coarea_loss.item(),
            'data': data_loss.item(),
            'prior': prior_loss.item()
        }
        
        return total_loss, loss_dict
    
    def prune_voxels(self, threshold: float = 0.9):
        """
        Prune voxels with SDF values outside threshold.
        
        Args:
            threshold: Pruning threshold (voxels with |f| > threshold are pruned)
        """
        with torch.no_grad():
            self.active_mask = torch.abs(self.sdf_grid) < threshold
    
    def upsample_grid(self, new_resolution: int):
        """
        Upsample grid to higher resolution using trilinear interpolation.
        
        Args:
            new_resolution: New grid resolution
        """
        old_resolution = self.grid_resolution
        old_grid = self.sdf_grid.data.clone()
        
        # Create new grid
        self.grid_resolution = new_resolution
        self.h = 1.0 / new_resolution
        
        # Create coordinate grid for new resolution
        coords = torch.linspace(0, 1 - 1e-6, new_resolution, device=self.device)
        x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
        new_points = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1)
        
        # Interpolate from old grid
        old_visco = VisCoGrids(
            grid_resolution=old_resolution,
            lambda_p=self.lambda_p,
            lambda_n=self.lambda_n,
            lambda_v=self.lambda_v,
            lambda_c=self.lambda_c,
            epsilon=self.epsilon,
            beta=self.beta,
            device=self.device
        )
        old_visco.sdf_grid.data = old_grid
        
        with torch.no_grad():
            new_values = old_visco.trilinear_interpolate(new_points, old_grid)
            new_grid = new_values.reshape(new_resolution, new_resolution, new_resolution)
        
        # Update grid
        self.sdf_grid = nn.Parameter(new_grid)
        self.active_mask = torch.ones(
            new_resolution, new_resolution, new_resolution, 
            dtype=torch.bool, device=self.device
        )
    
    def get_surface_points(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Extract surface points (zero level set) using marching cubes approximation.
        Simple implementation: returns points where SDF is close to zero.
        
        Args:
            threshold: Threshold for zero level set
            
        Returns:
            Surface points of shape (M, 3)
        """
        # For simplicity, return points where |SDF| < threshold
        # A full implementation would use marching cubes
        active_indices = torch.nonzero(
            (torch.abs(self.sdf_grid) < threshold) & self.active_mask, 
            as_tuple=False
        )
        
        if len(active_indices) == 0:
            return torch.zeros(0, 3, device=self.device)
        
        surface_points = (active_indices.float() + 0.5) * self.h
        return surface_points
    
    def extract_mesh(
        self, 
        level: float = 0.0,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh from SDF using marching cubes algorithm.
        
        Args:
            level: Iso-value for the surface (default: 0.0 for zero level set)
            spacing: Voxel spacing in each dimension (default: (h, h, h))
            
        Returns:
            vertices: Mesh vertices of shape (N, 3) in [0, 1]^3
            faces: Mesh faces (triangles) of shape (M, 3) with vertex indices
            
        Raises:
            ImportError: If scikit-image is not installed
        """
        if not HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for mesh extraction. "
                "Install with: pip install scikit-image"
            )
        
        # Get SDF grid as numpy array
        with torch.no_grad():
            sdf_np = self.sdf_grid.cpu().numpy()
        
        # Set spacing (voxel size)
        if spacing is None:
            spacing = (self.h, self.h, self.h)
        
        # Apply marching cubes
        # Note: skimage marching_cubes expects array in (z, y, x) order
        # Our grid is (x, y, z), so we transpose to (z, y, x)
        sdf_for_mc = np.transpose(sdf_np, (2, 1, 0))
        
        # Run marching cubes
        # spacing should match the transposed order: (z_spacing, y_spacing, x_spacing)
        spacing_mc = (spacing[2], spacing[1], spacing[0])
        
        verts, faces, normals, values = measure.marching_cubes(
            sdf_for_mc,
            level=level,
            spacing=spacing_mc,
            allow_degenerate=False
        )
        
        # Convert vertices back to (x, y, z) convention
        # skimage returns vertices in (z, y, x) coordinates
        # We need to swap back: (z, y, x) -> (x, y, z)
        vertices = np.zeros_like(verts)
        vertices[:, 0] = verts[:, 2]  # x = z
        vertices[:, 1] = verts[:, 1]  # y = y
        vertices[:, 2] = verts[:, 0]  # z = x
        
        return vertices, faces

