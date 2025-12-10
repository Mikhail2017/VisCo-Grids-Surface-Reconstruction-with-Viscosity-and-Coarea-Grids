# VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids

Implementation of the VisCo Grids method for Signed Distance Function (SDF) estimation from point clouds, based on the paper:

**VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids**  
Albert Pumarola, Artsiom Sanakoyeu, Lior Yariv, Ali Thabet, Yaron Lipman  
NeurIPS 2022

## Overview

VisCo Grids is a grid-based surface reconstruction method that replaces neural networks with simple grid functions, incorporating two novel geometric priors:

1. **Viscosity Loss**: Replaces the Eikonal loss to avoid bad minima and encourage smooth SDF solutions
2. **Coarea Loss**: Minimizes surface area to prevent excessive or "ghost" surface parts

## Key Features

- **Grid-based representation**: Uses a 3D voxel grid with trilinear interpolation
- **Coarse-to-fine optimization**: Starts at 64³ resolution and scales up to 256³
- **Efficient training**: Faster than Implicit Neural Representations (INRs)
- **Instant inference**: Direct grid lookup, no network evaluation needed
- **Mesh extraction**: Marching cubes algorithm for converting SDF to mesh

## Installation

```bash
# Required packages
pip install torch numpy matplotlib scikit-learn scikit-image tensorboard
```

**Note**: TensorBoard is optional but recommended for visualizing training progress. Install with `pip install tensorboard`.

## Usage

### Basic Example

```python
import torch
from train import train_visco_grids
from visco_grids import VisCoGrids

# Load or generate point cloud
points = torch.randn(1000, 3)  # Your point cloud
normals = None  # Optional: provide normals or estimate them

# Train model
model = train_visco_grids(
    points=points,
    normals=normals,
    initial_resolution=64,
    final_resolution=128,
    epochs_per_resolution=(2, 2),
    iterations_per_epoch=1000,
    verbose=True
)

# Extract mesh using marching cubes
vertices, faces = model.extract_mesh(level=0.0)
print(f"Extracted mesh: {len(vertices)} vertices, {len(faces)} faces")

# Save mesh to OBJ file
from mesh_utils import save_mesh_to_obj
save_mesh_to_obj(vertices, faces, "output_mesh.obj")
```

### With Normals

```python
from train import train_visco_grids, estimate_normals

# Estimate normals if not provided
normals = estimate_normals(points, k=10)

# Train with normals
model = train_visco_grids(
    points=points,
    normals=normals,
    initial_resolution=64,
    final_resolution=256,
    epochs_per_resolution=(5, 5, 3),
    iterations_per_epoch=12800
)
```

### Run Example

```bash
# Synthetic example
python example.py

# Real dataset example
python example_with_dataset.py --bunny
```

### TensorBoard Visualization

VisCo Grids supports TensorBoard logging to visualize training progress in real-time. All loss components are logged during training.

**Enable TensorBoard logging:**

```python
model = train_visco_grids(
    points=points,
    normals=normals,
    use_tensorboard=True,  # Enable TensorBoard
    log_dir='runs/visco_grids_experiment'  # Optional: specify log directory
)
```

**View training progress:**

1. Start TensorBoard server:
   ```bash
   tensorboard --logdir runs/
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:6006
   ```

**What's logged:**

- **Loss/Total_res{resolution}**: Total loss at each iteration
- **Loss/Point_res{resolution}**: Point loss (SDF should be zero at points)
- **Loss/Normal_res{resolution}**: Normal loss (gradient should match normals)
- **Loss/Viscosity_res{resolution}**: Viscosity loss (encourages SDF-like behavior)
- **Loss/Coarea_res{resolution}**: Coarea loss (minimizes surface area)
- **Loss/Data_res{resolution}**: Combined data loss (point + normal)
- **Loss/Prior_res{resolution}**: Combined prior loss (viscosity + coarea)
- **Epoch/*_res{resolution}**: Epoch-averaged losses for each resolution level

Losses are logged separately for each resolution level (64, 128, 256) to track coarse-to-fine optimization progress.

## Datasets

Several datasets work well with VisCo Grids:

### Recommended Datasets

1. **Stanford 3D Scanning Repository** (Mentioned in paper)
   - URL: http://graphics.stanford.edu/data/3Dscanrep/
   - Models: Bunny, Dragon, Armadillo, Buddha
   - Format: PLY
   - Usage: `load_point_cloud_from_ply("path/to/bunny.ply")`

2. **ShapeNet**
   - URL: https://www.shapenet.org/
   - Large collection of 3D models
   - Format: OBJ
   - Usage: `load_point_cloud_from_obj("path/to/model.obj")`

3. **ModelNet**
   - URL: https://modelnet.cs.princeton.edu/
   - Popular benchmark (ModelNet10, ModelNet40)
   - Format: OFF/OBJ
   - Usage: `load_modelnet_model("path/to/model.off")`

4. **ABC Dataset**
   - URL: https://deep-geometry.github.io/abc-dataset/
   - High-quality CAD models
   - Format: OBJ/STEP

See `datasets.py` for loading utilities and `example_with_dataset.py` for examples.

## Method Details

### Loss Function

The total loss combines data and prior terms:

```
L = L_data + L_prior
```

**Data Loss:**
- Point loss: `L_p = λ_p * Σ|f(q_k)|²` (encourages zero level set to pass through points)
- Normal loss: `L_n = λ_n * Σ||∇f(q_k) - n_k||²` (matches gradients to normals)

**Prior Loss:**
- Viscosity loss: `L_v = λ_v * (1/N) * Σ[(||∇f|| - 1)sign(f) - εΔf]²`
- Coarea loss: `L_c = λ_c * (1/N) * Σ[Φ_β(-f) * ||∇f||]`

### Hyperparameters

Default values (from paper):
- `λ_p = 0.1`: Point loss weight
- `λ_n = 1e-5`: Normal loss weight
- `λ_v = 1e-4`: Viscosity loss weight
- `λ_c = 1e-6`: Coarea loss weight
- `ε = 1e-2`: Viscosity parameter
- `β = 0.01`: Coarea parameter (Laplace distribution scale)

### Training Strategy

1. **Coarse-to-fine**: Start at 64³, upsample to 128³, then 256³
2. **Voxel pruning**: Remove voxels with |SDF| > threshold to save memory
3. **Batch sampling**: Sample 10% of active voxels per iteration
4. **Adam optimizer**: Learning rate 0.001, β₁=0.9, β₂=0.999

## Implementation Details

### Grid Representation

- Unit cube `[0, 1]³` discretized with `n × n × n` grid
- SDF values stored at grid nodes: `f_I ∈ ℝ`
- Trilinear interpolation for continuous evaluation

### Finite Differences

- **Gradient**: Symmetric finite differences
  ```
  D_x f = (f_{i+1,j,k} - f_{i-1,j,k}) / (2h)
  ```
- **Laplacian**: Second-order finite differences
  ```
  D²_x f = (f_{i+1,j,k} - 2f_{i,j,k} + f_{i-1,j,k}) / h²
  ```

### Viscosity Loss

The viscosity loss regularizes the Eikonal equation:
```
(||∇f|| - 1)sign(f) - εΔf = 0
```

This avoids bad minima that occur with the standard Eikonal loss.

### Coarea Loss

Uses the coarea formula to approximate surface area:
- Transform SDF to indicator function using Laplace CDF: `χ_β(p) = Ψ_β(-f(p))`
- Integrate gradient norm: `L_coarea = ∫ ||∇χ_β|| dp`
- As `β → 0`, this converges to the area of the zero level set

## Mesh Extraction

After training, extract a mesh from the SDF using marching cubes:

```python
# Extract mesh (zero level set)
vertices, faces = model.extract_mesh(level=0.0)

# Save to OBJ file
from mesh_utils import save_mesh_to_obj
save_mesh_to_obj(vertices, faces, "output_mesh.obj")
```

The `extract_mesh` method uses scikit-image's marching cubes algorithm to convert the SDF grid to a triangular mesh.

## File Structure

```
VisCoGrids/
├── visco_grids.py    # Main VisCo Grids implementation
├── train.py          # Training script with coarse-to-fine optimization
├── example.py        # Example usage with synthetic data
├── mesh_utils.py    # Mesh I/O utilities (OBJ format)
├── utils.py          # Point cloud utilities
└── README.md         # This file
```

## References

- Paper: [VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids](https://arxiv.org/abs/2210.15645)
- Original implementation: (to be published)

## License

This implementation is provided for research purposes. Please cite the original paper if you use this code.

