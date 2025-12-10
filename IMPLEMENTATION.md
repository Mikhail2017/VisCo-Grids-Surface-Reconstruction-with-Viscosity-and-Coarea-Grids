# VisCo Grids Implementation Details

This document describes the implementation of VisCo Grids for SDF estimation from normalized point clouds.

## Implementation Overview

The implementation follows the paper "VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids" (Pumarola et al., NeurIPS 2022).

## Core Components

### 1. Grid-Based SDF Representation (`visco_grids.py`)

- **VisCoGrids class**: Main model implementing the grid-based SDF
- **Trilinear interpolation**: Continuous evaluation of SDF at arbitrary points
- **Finite differences**: Gradient and Laplacian computation at grid nodes
- **Loss functions**: Data loss, viscosity loss, and coarea loss

### 2. Training Pipeline (`train.py`)

- **Coarse-to-fine optimization**: Multi-resolution training (64³ → 128³ → 256³)
- **Voxel pruning**: Memory-efficient training by removing inactive voxels
- **Normal estimation**: Optional PCA-based normal estimation
- **Adam optimizer**: With paper-specified hyperparameters

### 3. Utilities (`utils.py`)

- Point cloud loading/saving
- Normalization/denormalization helpers
- Coordinate transformations

## Key Equations Implemented

### Data Loss
```
L_data = λ_p * Σ|f(q_k)|² + λ_n * Σ||∇f(q_k) - n_k||²
```

### Viscosity Loss
```
L_viscosity = (1/N) * Σ_I [(||∇f(p_I)|| - 1) * sign(f(p_I)) - ε * Δf(p_I)]²
```

Where:
- `∇f(p_I)` is computed using symmetric finite differences
- `Δf(p_I) = D²_x f + D²_y f + D²_z f` is the Laplacian

### Coarea Loss
```
L_coarea = (1/N) * Σ_I [Φ_β(-f(w_I)) * ||∇f(w_I)||]
```

Where:
- `Φ_β(s) = (1/(2β)) * exp(-|s|/β)` is the Laplace PDF
- `w_I` are voxel centers

## Implementation Details

### Trilinear Interpolation

The SDF is evaluated at arbitrary points using trilinear interpolation:
- Points are converted to grid coordinates
- Corner values are retrieved from the 8 nearest grid nodes
- Interpolation weights are computed based on fractional parts
- Final value is a weighted sum of corner values

### Gradient Computation

Two methods are implemented:

1. **Automatic differentiation**: For arbitrary points (used in data loss)
   - Uses PyTorch's autograd with trilinear interpolation
   - Computes exact gradients

2. **Finite differences**: For grid nodes (used in viscosity loss)
   - Symmetric finite differences: `D_x f = (f_{i+1} - f_{i-1}) / (2h)`
   - Second-order differences for Laplacian: `D²_x f = (f_{i+1} - 2f_i + f_{i-1}) / h²`

### Coarse-to-Fine Training

1. Start at 64³ resolution
2. Train for specified epochs
3. Upsample to 128³ using trilinear interpolation
4. Train at 128³
5. Upsample to 256³ (if final_resolution >= 256)
6. Train at final resolution

### Voxel Pruning

After each resolution level:
- Remove voxels where `|SDF| > threshold`
- Only active voxels are used in subsequent training
- Reduces memory usage and speeds up training

## Default Hyperparameters

From the paper:
- `λ_p = 0.1`: Point loss weight
- `λ_n = 1e-5`: Normal loss weight  
- `λ_v = 1e-4`: Viscosity loss weight
- `λ_c = 1e-6`: Coarea loss weight
- `ε = 1e-2`: Viscosity parameter
- `β = 0.01`: Coarea parameter (Laplace scale)

Training:
- Learning rate: `0.001`
- Adam: `β₁ = 0.9`, `β₂ = 0.999`
- Batch size: 10% of active voxels
- Epochs: `(5, 5, 3)` for resolutions `(64, 128, 256)`
- Iterations per epoch: `12800`

## Usage Example

```python
from VisCoGrids import train_visco_grids, VisCoGrids
import torch

# Load point cloud
points = torch.randn(1000, 3)
normals = None  # Optional

# Train
model = train_visco_grids(
    points=points,
    normals=normals,
    initial_resolution=64,
    final_resolution=128,
    epochs_per_resolution=(2, 2),
    verbose=True
)

# Extract surface
surface_points = model.get_surface_points(threshold=0.05)
```

## Differences from Paper

1. **Simplified surface extraction**: Uses simple thresholding instead of marching cubes
2. **Normal estimation**: Optional PCA-based estimation if normals not provided
3. **Flexible resolutions**: Supports any resolution, not just powers of 2

## Performance Considerations

- **Memory**: Pruning reduces memory usage significantly at high resolutions
- **Speed**: Grid-based approach is faster than neural networks
- **GPU**: Recommended for resolutions >= 128³

## Mesh Extraction

The SDF grid can be converted to a mesh using marching cubes:

```python
vertices, faces = model.extract_mesh(level=0.0)
```

Implementation details:
- Uses scikit-image's `marching_cubes` function
- Handles coordinate system conversion (our grid is x,y,z, skimage expects z,y,x)
- Returns vertices in [0, 1]³ coordinate space
- Can save to OBJ format using `mesh_utils.save_mesh_to_obj()`

## Future Improvements

- [ ] Multi-GPU support
- [ ] Sparse grid representation for better memory efficiency
- [ ] Adaptive resolution based on point density
- [ ] GPU-accelerated marching cubes

