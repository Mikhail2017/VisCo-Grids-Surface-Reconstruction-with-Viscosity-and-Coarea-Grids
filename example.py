"""
Example script demonstrating VisCo Grids SDF estimation from point clouds.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple
from train import train_visco_grids, estimate_normals
from visco_grids import VisCoGrids


def generate_sphere_point_cloud(
    num_points: int = 1000, 
    radius: float = 0.3,
    sigma: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a point cloud around a sphere.
    
    Args:
        num_points: Number of points
        radius: Sphere radius
        sigma: Standard deviation for radial distribution. 
               sigma=0 means points are exactly on the sphere surface.
               sigma>0 adds Gaussian noise in radial direction.
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Normals of shape (N, 3)
    """
    # Generate points on sphere
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.arccos(np.random.uniform(-1, 1, num_points))
    
    # Unit sphere directions
    x_dir = np.sin(phi) * np.cos(theta)
    y_dir = np.sin(phi) * np.sin(theta)
    z_dir = np.cos(phi)
    directions = np.stack([x_dir, y_dir, z_dir], axis=1)
    
    # Generate radial distances
    if sigma > 0:
        # Add Gaussian noise in radial direction
        radial_distances = radius + np.random.normal(0, sigma, num_points)
        radial_distances = np.maximum(radial_distances, 0.01)  # Ensure positive
    else:
        # Points exactly on sphere
        radial_distances = np.full(num_points, radius)
    
    # Scale directions by radial distances
    points = directions * radial_distances[:, np.newaxis]
    
    # Center at origin
    points = points - points.mean(axis=0)
    
    # Normals point outward from center (use original directions, not noisy points)
    normals = directions
    
    return torch.from_numpy(points).float(), torch.from_numpy(normals).float()


def generate_torus_point_cloud(
    num_points: int = 2000,
    major_radius: float = 0.3,
    minor_radius: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a point cloud on a torus.
    
    Args:
        num_points: Number of points
        major_radius: Major radius of torus
        minor_radius: Minor radius of torus
        
    Returns:
        points: Point cloud of shape (N, 3)
        normals: Normals of shape (N, 3)
    """
    # Generate points on torus
    u = np.random.uniform(0, 2 * np.pi, num_points)
    v = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = (major_radius + minor_radius * np.cos(v)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v)) * np.sin(u)
    z = minor_radius * np.sin(v)
    
    points = np.stack([x, y, z], axis=1)
    
    # Add some noise
    points += np.random.normal(0, 0.01, points.shape)
    
    # Center at origin
    points = points - points.mean(axis=0)
    
    # Compute normals (simplified)
    normals = np.zeros_like(points)
    for i in range(len(points)):
        # Normal points from center of torus tube to point
        center = np.array([
            major_radius * np.cos(u[i]),
            major_radius * np.sin(u[i]),
            0
        ])
        normal = points[i] - center
        normal = normal / np.linalg.norm(normal)
        normals[i] = normal
    
    return torch.from_numpy(points).float(), torch.from_numpy(normals).float()


def visualize_results(
    points: torch.Tensor,
    model: VisCoGrids,
    title: str = "VisCo Grids Reconstruction",
    show_mesh: bool = True
):
    """
    Visualize point cloud and reconstructed surface.
    
    Args:
        points: Input point cloud
        model: Trained VisCo Grids model
        title: Plot title
        show_mesh: Whether to extract and show mesh using marching cubes
    """
    fig = plt.figure(figsize=(20, 5))
    
    # Original point cloud
    ax1 = fig.add_subplot(141, projection='3d')
    points_np = points.cpu().numpy()
    ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1, alpha=0.5)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])
    
    # Extract mesh using marching cubes
    if show_mesh:
        try:
            vertices, faces = model.extract_mesh(level=0.0)
            ax2 = fig.add_subplot(142, projection='3d')
            
            # Plot mesh (sample faces for visualization if too many)
            if len(faces) > 10000:
                # Sample faces for faster rendering
                sample_idx = np.random.choice(len(faces), 10000, replace=False)
                faces_sample = faces[sample_idx]
            else:
                faces_sample = faces
            
            # Create mesh collection with both front and back faces visible
            # Add original faces
            mesh = Poly3DCollection(vertices[faces_sample], alpha=0.7, edgecolor='k', linewidths=0.1)
            mesh.set_facecolor([0.7, 0.7, 0.9])
            # Add reversed faces to make backfaces visible
            faces_reversed = faces_sample[:, [0, 2, 1]]  # Reverse face orientation
            mesh_back = Poly3DCollection(vertices[faces_reversed], alpha=0.7, edgecolor='k', linewidths=0.1)
            mesh_back.set_facecolor([0.7, 0.7, 0.9])
            ax2.add_collection3d(mesh)
            ax2.add_collection3d(mesh_back)
            
            # Set limits
            ax2.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
            ax2.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
            ax2.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
            ax2.set_title(f'Mesh (Marching Cubes)\n{len(vertices)} vertices, {len(faces)} faces')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_box_aspect([1, 1, 1])
            
            print(f"   Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        except ImportError:
            print("   Warning: scikit-image not available, skipping mesh extraction")
            ax2 = fig.add_subplot(142, projection='3d')
            ax2.text(0.5, 0.5, 0.5, 'Mesh extraction\nrequires scikit-image', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mesh (Marching Cubes)')
        except Exception as e:
            print(f"   Error extracting mesh: {e}")
            ax2 = fig.add_subplot(142, projection='3d')
            ax2.text(0.5, 0.5, 0.5, f'Mesh extraction failed:\n{str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Mesh (Marching Cubes)')
    else:
        # Extract surface points (fallback)
        surface_points = model.get_surface_points(threshold=0.05)
        if len(surface_points) > 0:
            surface_np = surface_points.cpu().numpy()
            ax2 = fig.add_subplot(142, projection='3d')
            ax2.scatter(surface_np[:, 0], surface_np[:, 1], surface_np[:, 2], s=1, alpha=0.5)
            ax2.set_title('Reconstructed Surface')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
    
    # SDF visualization (slice)
    ax3 = fig.add_subplot(143)
    with torch.no_grad():
        # Create a slice through the middle
        z_slice = model.grid_resolution // 2
        sdf_slice = model.sdf_grid[:, :, z_slice].cpu().numpy()
        im = ax3.imshow(sdf_slice, cmap='RdYlBu', origin='lower')
        ax3.set_title('SDF Slice (middle z)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im, ax=ax3)
    
    # SDF histogram
    ax4 = fig.add_subplot(144)
    with torch.no_grad():
        sdf_values = model.sdf_grid.cpu().numpy().flatten()
        ax4.hist(sdf_values, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', label='Zero level')
        ax4.set_title('SDF Value Distribution')
        ax4.set_xlabel('SDF Value')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main example: train VisCo Grids on a synthetic point cloud.
    """
    print("VisCo Grids SDF Estimation Example")
    print("=" * 50)
    
    # Generate synthetic point cloud
    print("\n1. Generating synthetic point cloud...")
    sigma = 0.01  # Control point distribution: 0 = on sphere, >0 = distributed around sphere
    points, normals = generate_sphere_point_cloud(num_points=1000, sigma=sigma)
    print(f"   Generated {len(points)} points (sigma={sigma})")
    
    # Save input point cloud to XYZ format
    try:
        from mesh_utils import save_pointcloud_to_xyz
        points_np = points.cpu().numpy()
        normals_np = normals.cpu().numpy() if normals is not None else None
        save_pointcloud_to_xyz(points_np, "input_pointcloud.xyz", normals=normals_np)
        print("   Saved input point cloud to 'input_pointcloud.xyz'")
    except Exception as e:
        print(f"   Warning: Could not save XYZ file: {e}")
    
    # Train model
    print("\n2. Training VisCo Grids...")
    print("   (This may take a few minutes...)")
    
    model = train_visco_grids(
        points=points,
        normals=normals,
        initial_resolution=64,
        final_resolution=128,  # Use 128 for faster demo
        epochs_per_resolution=(2, 2),  # Fewer epochs for demo
        iterations_per_epoch=1000,  # Fewer iterations for demo
        batch_size_ratio=0.1,
        prune_threshold=0.9,
        learning_rate=0.001,
        verbose=True,
        save_intermediate_meshes=True,  # Save meshes at each resolution
        use_tensorboard=True,  # Enable TensorBoard logging
        log_dir='runs/visco_grids_example'  # TensorBoard log directory
    )
    
    # Evaluate
    print("\n3. Evaluating reconstruction...")
    with torch.no_grad():
        # Normalize points
        min_vals = points.min(dim=0)[0]
        max_vals = points.max(dim=0)[0]
        center = (min_vals + max_vals) / 2
        scale = (max_vals - min_vals).max()
        points_normalized = (points - center) / scale
        points_normalized = points_normalized * 0.45 + 0.5
        
        # Compute final loss
        loss, loss_dict = model.total_loss(points_normalized, normals)
        print(f"   Final loss: {loss_dict['total']:.6f}")
        print(f"   Point loss: {loss_dict['point']:.6f}")
        print(f"   Normal loss: {loss_dict['normal']:.6f}")
        print(f"   Viscosity loss: {loss_dict['viscosity']:.6f}")
        print(f"   Coarea loss: {loss_dict['coarea']:.6f}")
    
    # Extract and save mesh
    print("\n4. Extracting mesh using marching cubes...")
    try:
        vertices, faces = model.extract_mesh(level=0.0)
        print(f"   Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Save mesh to PLY file
        try:
            from mesh_utils import save_mesh_to_ply, compute_vertex_normals
            # Compute normals for PLY file
            vertex_normals = compute_vertex_normals(vertices, faces)
            save_mesh_to_ply(vertices, faces, "reconstructed_mesh.ply", normals=vertex_normals)
            print("   Saved final mesh to 'reconstructed_mesh.ply'")
        except Exception as e:
            print(f"   Warning: Could not save PLY file: {e}")
    except ImportError:
        print("   Warning: scikit-image not available. Install with: pip install scikit-image")
    except Exception as e:
        print(f"   Error extracting mesh: {e}")
    
    # Visualize
    print("\n5. Visualizing results...")
    visualize_results(points, model, "VisCo Grids: Sphere Reconstruction", show_mesh=True)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

