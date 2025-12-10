"""
Training script for VisCo Grids SDF estimation.
Implements coarse-to-fine optimization as described in the paper.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple
from visco_grids import VisCoGrids
import time
import os
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not found. Install with 'pip install tensorboard' for visualization.")


def train_visco_grids(
    points: torch.Tensor,
    normals: Optional[torch.Tensor] = None,
    initial_resolution: int = 64,
    final_resolution: int = 256,
    epochs_per_resolution: Tuple[int, int, int] = (5, 5, 3),
    iterations_per_epoch: int = 12800,
    batch_size_ratio: float = 0.1,
    prune_threshold: float = 0.9,
    learning_rate: float = 0.001,
    lambda_p: float = 0.1,
    lambda_n: float = 1e-5,
    lambda_v: float = 1e-4,
    lambda_c: float = 1e-6,
    epsilon: float = 1e-2,
    beta: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    save_intermediate_meshes: bool = False,
    log_dir: Optional[str] = None,
    use_tensorboard: bool = True
) -> VisCoGrids:
    """
    Train VisCo Grids using coarse-to-fine optimization.
    
    Args:
        points: Point cloud of shape (N, 3)
        normals: Optional normals of shape (N, 3)
        initial_resolution: Starting grid resolution
        final_resolution: Final grid resolution
        epochs_per_resolution: Number of epochs for each resolution level
        iterations_per_epoch: Number of iterations per epoch
        batch_size_ratio: Ratio of points to sample per iteration
        prune_threshold: Threshold for voxel pruning
        learning_rate: Learning rate for Adam optimizer
        lambda_p: Weight for point loss
        lambda_n: Weight for normal loss
        lambda_v: Weight for viscosity loss
        lambda_c: Weight for coarea loss
        epsilon: Viscosity parameter
        beta: Coarea parameter
        device: Device to run on
        verbose: Whether to print progress
        save_intermediate_meshes: Whether to save meshes at each resolution
        log_dir: Directory for TensorBoard logs (default: 'runs/visco_grids')
        use_tensorboard: Whether to use TensorBoard for logging
        
    Returns:
        Trained VisCoGrids model
    """
    # Normalize point cloud
    points = points.to(device)
    if normals is not None:
        normals = normals.to(device)
    
    # Normalize to [0, 1]^3
    # Store normalization parameters for later use
    min_vals = points.min(dim=0)[0]
    max_vals = points.max(dim=0)[0]
    center = (min_vals + max_vals) / 2
    scale = (max_vals - min_vals).max()
    points_normalized = (points - center) / scale
    points_normalized = points_normalized * 0.45 + 0.5
    
    # Initialize model
    model = VisCoGrids(
        grid_resolution=initial_resolution,
        lambda_p=lambda_p,
        lambda_n=lambda_n,
        lambda_v=lambda_v,
        lambda_c=lambda_c,
        epsilon=epsilon,
        beta=beta,
        device=device
    )
    model = model.to(device)
    
    # Coarse-to-fine resolutions
    resolutions = [initial_resolution]
    if initial_resolution < final_resolution:
        if final_resolution == 256:
            resolutions = [64, 128, 256]
        elif final_resolution == 128:
            resolutions = [64, 128]
        else:
            # Interpolate resolutions
            current = initial_resolution
            while current < final_resolution:
                current = min(current * 2, final_resolution)
                resolutions.append(current)
    
    total_iterations = 0
    
    # Initialize TensorBoard writer
    writer = None
    if use_tensorboard and HAS_TENSORBOARD:
        if log_dir is None:
            log_dir = 'runs/visco_grids'
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        if verbose:
            print(f"TensorBoard logging to: {log_dir}")
            print(f"View with: tensorboard --logdir {log_dir}")
    
    for res_idx, resolution in enumerate(resolutions):
        if resolution != model.grid_resolution:
            if verbose:
                print(f"Upsampling from {model.grid_resolution} to {resolution}")
            model.upsample_grid(resolution)
        
        # Initialize SDF grid from point cloud (only for coarsest level)
        if res_idx == 0:
            if verbose:
                print(f"Initializing SDF grid from point cloud at resolution {resolution}...")
            model.initialize_from_pointcloud(points_normalized, normals, k_neighbors=5)
        
        epochs = epochs_per_resolution[min(res_idx, len(epochs_per_resolution) - 1)]
        
        # Optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999)
        )
        
        # Prune voxels
        model.prune_voxels(prune_threshold)
        
        if verbose:
            num_active = model.active_mask.sum().item()
            total_voxels = model.grid_resolution ** 3
            print(f"\nResolution {resolution}x{resolution}x{resolution}")
            print(f"Active voxels: {num_active}/{total_voxels} ({100*num_active/total_voxels:.1f}%)")
            print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_losses = {
                'total': 0.0,
                'point': 0.0,
                'normal': 0.0,
                'viscosity': 0.0,
                'coarea': 0.0
            }
            
            for iteration in range(iterations_per_epoch):
                # Sample batch of points
                num_points = points_normalized.shape[0]
                batch_size = max(1, int(batch_size_ratio * num_points))
                indices = torch.randint(0, num_points, (batch_size,), device=device)
                batch_points = points_normalized[indices]
                batch_normals = normals[indices] if normals is not None else None
                
                # Forward pass
                optimizer.zero_grad()
                loss, loss_dict = model.total_loss(batch_points, batch_normals)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key]
                
                # Log to TensorBoard
                if writer is not None:
                    writer.add_scalar(f'Loss/Total_res{resolution}', loss_dict['total'], total_iterations)
                    writer.add_scalar(f'Loss/Point_res{resolution}', loss_dict['point'], total_iterations)
                    writer.add_scalar(f'Loss/Normal_res{resolution}', loss_dict['normal'], total_iterations)
                    writer.add_scalar(f'Loss/Viscosity_res{resolution}', loss_dict['viscosity'], total_iterations)
                    writer.add_scalar(f'Loss/Coarea_res{resolution}', loss_dict['coarea'], total_iterations)
                    writer.add_scalar(f'Loss/Data_res{resolution}', loss_dict['data'], total_iterations)
                    writer.add_scalar(f'Loss/Prior_res{resolution}', loss_dict['prior'], total_iterations)
                
                total_iterations += 1
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= iterations_per_epoch
            
            # Log epoch averages to TensorBoard
            if writer is not None:
                writer.add_scalar(f'Epoch/Loss_res{resolution}', epoch_losses['total'], epoch)
                writer.add_scalar(f'Epoch/Point_res{resolution}', epoch_losses['point'], epoch)
                writer.add_scalar(f'Epoch/Normal_res{resolution}', epoch_losses['normal'], epoch)
                writer.add_scalar(f'Epoch/Viscosity_res{resolution}', epoch_losses['viscosity'], epoch)
                writer.add_scalar(f'Epoch/Coarea_res{resolution}', epoch_losses['coarea'], epoch)
            
            if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Loss={epoch_losses['total']:.6f}, "
                      f"Point={epoch_losses['point']:.6f}, "
                      f"Visc={epoch_losses['viscosity']:.6f}, "
                      f"Coarea={epoch_losses['coarea']:.6f}")
        
        # Prune after each resolution
        model.prune_voxels(prune_threshold)
        
        # Save intermediate mesh if requested
        if save_intermediate_meshes:
            try:
                vertices, faces = model.extract_mesh(level=0.0)
                from mesh_utils import save_mesh_to_ply, compute_vertex_normals
                vertex_normals = compute_vertex_normals(vertices, faces)
                filename = f"mesh_resolution_{resolution}.ply"
                save_mesh_to_ply(vertices, faces, filename, normals=vertex_normals)
                if verbose:
                    print(f"   Saved intermediate mesh to '{filename}'")
            except Exception as e:
                if verbose:
                    print(f"   Warning: Could not save intermediate mesh: {e}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        if verbose:
            print(f"\nTensorBoard logs saved to: {log_dir}")
    
    if verbose:
        print("\nTraining complete!")
        print(f"Total iterations: {total_iterations}")
    
    return model


def estimate_normals(
    points: torch.Tensor,
    k: int = 10
) -> torch.Tensor:
    """
    Estimate normals from point cloud using PCA on k-nearest neighbors.
    
    Args:
        points: Point cloud of shape (N, 3)
        k: Number of nearest neighbors
        
    Returns:
        Estimated normals of shape (N, 3)
    """
    from sklearn.neighbors import NearestNeighbors
    
    points_np = points.cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points_np)
    distances, indices = nbrs.kneighbors(points_np)
    
    normals = np.zeros_like(points_np)
    
    for i in range(len(points_np)):
        # Get k nearest neighbors (excluding self)
        neighbors = points_np[indices[i, 1:]]
        # Center neighbors
        neighbors_centered = neighbors - neighbors.mean(axis=0)
        # PCA
        cov = neighbors_centered.T @ neighbors_centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Normal is eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]
        # Orient consistently (point towards origin or average)
        if normal @ (points_np[i] - neighbors.mean(axis=0)) < 0:
            normal = -normal
        normals[i] = normal
    
    return torch.from_numpy(normals).float().to(points.device)

