# VisCo Grids Project Summary

## Overview

VisCo Grids is a PyTorch-based implementation of the surface reconstruction method described in the paper:

**"VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids"**  
*Pumarola et al., NeurIPS 2022*

The method reconstructs 3D surfaces from unorganized point clouds by optimizing a grid-based Signed Distance Function (SDF). Unlike neural network approaches, VisCo Grids uses simple voxel grids with trilinear interpolation, making it faster to train and providing instant inference.

### Key Innovations

1. **Viscosity Loss**: Replaces the standard Eikonal loss to avoid bad local minima and encourage smooth SDF solutions
2. **Coarea Loss**: Minimizes surface area to prevent excessive or "ghost" surface artifacts
3. **Grid-based Representation**: Faster than Implicit Neural Representations (INRs) with direct grid lookup

---

## Architecture

```
Point Cloud Input
       │
       ▼
┌─────────────────┐
│  Normalization  │  (Scale to [0,1]³)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SDF Grid Init  │  (KD-tree based initialization)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Coarse-to-Fine Optimization     │
│  ┌─────────────────────────────┐    │
│  │  Resolution 64³ → 128³ → 256³│   │
│  │  • Data Loss (point+normal)  │   │
│  │  • Viscosity Loss            │   │
│  │  • Coarea Loss               │   │
│  │  • Voxel Pruning             │   │
│  └─────────────────────────────┘    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Marching Cubes  │  (Mesh extraction)
└────────┬────────┘
         │
         ▼
   Output Mesh (OBJ/PLY)
```

---

## Module Summaries

### Core Algorithm

#### `visco_grids.py` - Main VisCo Grids Implementation

The heart of the project containing the `VisCoGrids` class (PyTorch `nn.Module`).

**Key Components:**

| Component | Description |
|-----------|-------------|
| `__init__()` | Initialize grid with configurable resolution and loss weights |
| `sdf_grid` | Learnable 3D tensor storing SDF values at grid nodes |
| `active_mask` | Boolean mask for voxel pruning |

**Core Methods:**

| Method | Purpose |
|--------|---------|
| `normalize_point_cloud()` | Scale points to unit cube [0,1]³ |
| `initialize_from_pointcloud()` | KD-tree based SDF initialization with sign from normals |
| `trilinear_interpolate()` | Query SDF at arbitrary points via trilinear interpolation |
| `compute_gradient()` | Compute SDF gradient using autograd |
| `compute_finite_differences()` | Compute gradient and Laplacian at grid nodes |

**Loss Functions:**

| Loss | Formula | Purpose |
|------|---------|---------|
| `data_loss()` | `λ_p·Σ|f(q)|² + λ_n·Σ‖∇f(q)-n‖²` | Point and normal constraints |
| `viscosity_loss()` | `Σ[(‖∇f‖-1)·sign(f) - ε·Δf]²` | Encourage SDF behavior |
| `coarea_loss()` | `Σ[Φ_β(-f)·‖∇f‖]` | Minimize surface area |
| `total_loss()` | `L_data + L_prior` | Combined optimization objective |

**Grid Operations:**

| Method | Purpose |
|--------|---------|
| `prune_voxels()` | Remove voxels with |SDF| > threshold |
| `upsample_grid()` | Trilinear upsampling to higher resolution |
| `extract_mesh()` | Marching cubes mesh extraction |
| `get_surface_points()` | Extract zero level set points |

**Default Hyperparameters (from paper):**
- `λ_p = 0.1` (point loss weight)
- `λ_n = 1e-5` (normal loss weight)
- `λ_v = 1e-4` (viscosity loss weight)
- `λ_c = 1e-6` (coarea loss weight)
- `ε = 1e-2` (viscosity parameter)
- `β = 0.01` (coarea Laplace scale)

---

### Training Pipeline

#### `train.py` - Training Script

Implements coarse-to-fine optimization strategy.

**Main Function:**

```python
train_visco_grids(
    points,                    # Input point cloud (N, 3)
    normals=None,              # Optional normals (N, 3)
    initial_resolution=64,     # Starting grid resolution
    final_resolution=256,      # Target grid resolution
    epochs_per_resolution=(5, 5, 3),
    iterations_per_epoch=12800,
    batch_size_ratio=0.1,      # Sample 10% of points per iteration
    prune_threshold=0.9,       # Voxel pruning threshold
    learning_rate=0.001,
    use_tensorboard=True       # Enable TensorBoard logging
) -> VisCoGrids
```

**Training Strategy:**
1. Initialize at coarse resolution (64³)
2. Train with Adam optimizer (β₁=0.9, β₂=0.999)
3. Prune inactive voxels
4. Upsample to next resolution
5. Repeat until final resolution

**Helper Function:**

```python
estimate_normals(points, k=10) -> torch.Tensor
```
PCA-based normal estimation using k-nearest neighbors.

**TensorBoard Logging:**
- Per-iteration losses at each resolution
- Epoch-averaged losses
- Separate tracking for data vs prior losses

---

### Data Loading

#### `datasets.py` - Dataset Utilities

Comprehensive point cloud loading and preprocessing.

**File Loaders:**

| Function | Format | Description |
|----------|--------|-------------|
| `load_point_cloud_from_ply()` | PLY | Load point cloud/mesh from PLY |
| `load_point_cloud_from_obj()` | OBJ | Sample points from OBJ mesh |
| `find_ply_file()` | PLY | Search directories for PLY files |

**Standard Dataset Loaders:**

| Function | Dataset | Source |
|----------|---------|--------|
| `load_stanford_bunny()` | Stanford Bunny | Stanford 3D Repository |
| `load_armadillo()` | Armadillo | Stanford 3D Repository |
| `load_dragon()` | Dragon | Stanford 3D Repository |
| `load_shapenet_model()` | ShapeNet | shapenet.org |
| `load_modelnet_model()` | ModelNet | Princeton |

**Point Cloud Processing:**

| Function | Purpose |
|----------|---------|
| `normalize_point_cloud()` | Center and scale to [0,1]³ with margin |
| `downsample_point_cloud()` | Random or FPS downsampling |
| `add_noise_to_point_cloud()` | Add Gaussian noise for testing |

**Dataset Information:**

```python
DATASET_INFO = {
    "stanford_3d": {...},  # Bunny, Dragon, Armadillo
    "shapenet": {...},     # 3M+ models
    "modelnet": {...},     # ModelNet10/40
    "abc_dataset": {...},  # CAD models
    "scannet": {...},      # Indoor scans
    "eth_3d": {...}        # Outdoor scenes
}
```

---

### Mesh Utilities

#### `mesh_utils.py` - Mesh I/O and Processing

**Export Functions:**

| Function | Format | Features |
|----------|--------|----------|
| `save_mesh_to_obj()` | OBJ | Vertices, faces, optional normals |
| `save_mesh_to_ply()` | PLY | ASCII format with normals |
| `save_pointcloud_to_xyz()` | XYZ | Simple point + normal format |

**Import Functions:**

| Function | Format | Features |
|----------|--------|----------|
| `load_mesh_from_obj()` | OBJ | Handles v, vn, f with triangulation |

**Mesh Processing:**

| Function | Purpose |
|----------|---------|
| `invert_face_normals()` | Flip triangle orientations |
| `fix_triangle_orientations()` | BFS-based consistent orientation |
| `compute_vertex_normals()` | Average face normals per vertex |

---

### General Utilities

#### `utils.py` - Point Cloud Utilities

**NumPy I/O:**

| Function | Purpose |
|----------|---------|
| `load_point_cloud_from_numpy()` | Load from .npy (with optional normals) |
| `save_point_cloud_to_numpy()` | Save to .npy format |

**Normalization:**

| Function | Purpose |
|----------|---------|
| `normalize_point_cloud()` | Scale to [margin, 1-margin]³ |
| `denormalize_point_cloud()` | Restore original coordinates |

---

### Examples

#### `example.py` - Synthetic Data Examples

**Synthetic Point Cloud Generators:**

| Function | Shape | Parameters |
|----------|-------|------------|
| `generate_sphere_point_cloud()` | Sphere | radius, noise sigma |
| `generate_torus_point_cloud()` | Torus | major/minor radius |

**Visualization:**

```python
visualize_results(points, model, title, show_mesh=True)
```
Creates 4-panel matplotlib figure:
1. Input point cloud
2. Reconstructed mesh (marching cubes)
3. SDF slice visualization
4. SDF value histogram

**Main Workflow:**
1. Generate synthetic sphere
2. Save input point cloud
3. Train VisCo Grids
4. Extract and save mesh
5. Visualize results

#### `example_with_dataset.py` - Real Dataset Examples

**Example Functions:**

| Function | Dataset | Description |
|----------|---------|-------------|
| `example_stanford_bunny()` | Stanford Bunny | Classic test model |
| `example_custom_point_cloud()` | Any PLY/OBJ | User-provided files |

**Command Line Interface:**
```bash
python example_with_dataset.py --bunny        # Stanford Bunny
python example_with_dataset.py --file <path>  # Custom file
python example_with_dataset.py --list         # List datasets
```

---

### Configuration Files

#### `requirements.txt` - Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥1.9.0 | Deep learning framework |
| numpy | ≥1.19.0 | Numerical computing |
| matplotlib | ≥3.3.0 | Visualization |
| scikit-learn | ≥0.24.0 | KD-tree, normal estimation |
| scikit-image | ≥0.18.0 | Marching cubes |
| trimesh | ≥3.9.0 | Mesh loading |
| tensorboard | ≥2.0.0 | Training visualization |

#### `.gitignore` - Ignored Files

- Python bytecode (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `.venv`)
- IDE files (`.vscode/`, `.idea/`)
- Output files (`*.obj`, `*.ply`, `*.xyz`)
- TensorBoard logs (`runs/`)
- Model checkpoints (`*.pth`, `*.pt`)

---

### Documentation

#### `README.md` - Project Overview

- Installation instructions
- Basic usage examples
- Method overview
- Hyperparameter documentation
- Dataset recommendations

#### `DATASETS.md` - Dataset Guide

- Detailed dataset descriptions
- Download instructions
- File format specifications
- Usage examples for each dataset

#### `IMPLEMENTATION.md` - Technical Details

- Algorithm implementation notes
- Equation implementations
- Differences from paper
- Performance considerations

---

## Typical Usage Workflow

```python
import torch
from train import train_visco_grids, estimate_normals
from datasets import load_point_cloud_from_ply, normalize_point_cloud
from mesh_utils import save_mesh_to_obj

# 1. Load point cloud
points, normals = load_point_cloud_from_ply("input.ply")

# 2. Normalize to unit cube
points_norm, center, scale = normalize_point_cloud(points)

# 3. Estimate normals if needed
if normals is None:
    normals = estimate_normals(points_norm, k=10)

# 4. Train model
model = train_visco_grids(
    points=points_norm,
    normals=normals,
    initial_resolution=64,
    final_resolution=128,
    verbose=True
)

# 5. Extract mesh
vertices, faces = model.extract_mesh(level=0.0)

# 6. Save result
save_mesh_to_obj(vertices, faces, "output.obj")
```

---

## Key Algorithms

### Trilinear Interpolation

For a point `p` in the grid, find the 8 corner values and interpolate:

```
f(p) = Σ w_i · f_i
```

where `w_i` are trilinear weights based on fractional position.

### Viscosity Loss

Regularizes the Eikonal equation to avoid bad minima:

```
L_v = (1/N) · Σ [(‖∇f‖ - 1) · sign(f) - ε · Δf]²
```

### Coarea Loss

Uses Laplace distribution to approximate surface area:

```
L_c = (1/N) · Σ [Φ_β(-f) · ‖∇f‖]

where Φ_β(s) = (1/2β) · exp(-|s|/β)
```

### Coarse-to-Fine Strategy

1. **64³**: Capture global structure
2. **128³**: Add medium details
3. **256³**: Refine fine details

Each level inherits from previous via trilinear upsampling.

---

## Performance Notes

- **Memory**: Voxel pruning significantly reduces memory at high resolutions
- **Speed**: Grid-based approach is faster than neural networks
- **GPU**: Recommended for resolutions ≥ 128³
- **Inference**: Instant (direct grid lookup, no network evaluation)

---

## File Structure

```
VisCo-Grids/
├── visco_grids.py          # Core algorithm
├── train.py                # Training pipeline
├── datasets.py             # Data loading utilities
├── mesh_utils.py           # Mesh I/O
├── utils.py                # General utilities
├── example.py              # Synthetic examples
├── example_with_dataset.py # Real dataset examples
├── requirements.txt        # Dependencies
├── README.md               # Project overview
├── DATASETS.md             # Dataset documentation
├── IMPLEMENTATION.md       # Technical details
├── summary.md              # This file
└── .gitignore              # Git ignore rules
```

---

## References

- **Paper**: [VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids](https://arxiv.org/abs/2303.14569)
- **Stanford 3D Repository**: http://graphics.stanford.edu/data/3Dscanrep/
- **ShapeNet**: https://www.shapenet.org/
- **ModelNet**: https://modelnet.cs.princeton.edu/
