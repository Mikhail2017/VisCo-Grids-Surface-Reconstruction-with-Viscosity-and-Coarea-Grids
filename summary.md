# VisCo Grids — Project Summary

## Overview

**VisCo Grids** is a PyTorch-based implementation of grid-based **Signed Distance Function (SDF)** estimation for 3D surface reconstruction from point clouds. The core idea is to represent a surface implicitly as the zero level-set of an SDF stored on a regular 3D voxel grid, and to optimize that grid using a combination of data-fitting terms and geometric regularizers inspired by **viscosity solutions** and the **coarea formula** from geometric measure theory.

Given an input point cloud (with optional normals), the pipeline:

1. Normalizes the points into a unit cube.
2. Initializes a coarse voxel grid.
3. Progressively refines the grid through a coarse-to-fine training schedule.
4. Extracts a triangle mesh from the optimized SDF via Marching Cubes.

---

## Module Breakdown

### `visco_grids.py` — Core Model

The central `VisCoGrids` class (`nn.Module`) holds a learnable 3D grid of SDF values and implements all loss functions and geometric operations.

#### Key Algorithms

| Component | Description |
|---|---|
| **Trilinear Interpolation** | Queries continuous SDF values at arbitrary 3D points by interpolating the 8 surrounding voxel corners. This makes the discrete grid differentiable with respect to query position. |
| **Gradient Computation** | Two modes: (a) **autograd-based** — differentiates through trilinear interpolation for exact gradients at arbitrary query points; (b) **finite differences** — central differences on the grid for bulk computation at all voxel centers. |
| **SDF Initialization** | Uses a KD-tree (`sklearn.neighbors.KDTree`) to find the nearest surface point for each voxel center. The unsigned distance is the Euclidean distance to the nearest neighbor. The sign is determined by the dot product between the center-to-point vector and the average normal of `k` nearest neighbors: negative if the voxel is inside the surface, positive if outside. |
| **Data Loss** | Two terms: **(a) Point loss** — `λ_p · mean(SDF(pᵢ)²)` penalizes non-zero SDF at known surface points; **(b) Normal loss** — `λ_n · mean(|∇SDF(pᵢ) − nᵢ|²)` aligns the SDF gradient with provided normals. |
| **Viscosity Loss** | Enforces the Eikonal equation `|∇SDF| = 1` with a viscosity correction: `λ_v · mean[((|∇f| − 1)·sign(f) − ε·Δf)²]`, where `Δf` is the Laplacian computed via second-order central finite differences. The `sign(f)` term and Laplacian `ε·Δf` come from the viscosity solution framework, ensuring the optimizer converges to the correct weak solution of the Eikonal equation. |
| **Coarea Loss** | Minimizes total surface area via the coarea formula: `λ_c · mean(Φ_β(−f) · |∇f|)`, where `Φ_β` is the **Laplace distribution PDF** `(1/2β)·exp(−|s|/β)`. This acts as a smoothed Dirac delta that concentrates the penalty near the zero level-set (`f ≈ 0`), weighted by gradient magnitude. The scale parameter `β` controls the narrow-band width. |
| **Voxel Pruning** | Marks voxels with `|SDF| > threshold` as inactive. Subsequent loss computations skip pruned voxels, focusing compute on the narrow band around the surface. |
| **Grid Upsampling** | Creates a new higher-resolution grid and populates it by trilinear interpolation from the old grid. The active mask is reset to all-active after upsampling. |
| **Mesh Extraction** | Applies **Marching Cubes** (`skimage.measure.marching_cubes`) to the SDF grid at the zero iso-level. Handles coordinate transposition (the grid is stored as `(x,y,z)` but skimage expects `(z,y,x)`). Post-processes with BFS-based orientation fixing and normal inversion. |

#### Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `lambda_p` | 0.1 | Point constraint weight |
| `lambda_n` | 1e-5 | Normal constraint weight |
| `lambda_v` | 1e-4 | Viscosity (Eikonal) regularizer weight |
| `lambda_c` | 1e-6 | Coarea (surface area) regularizer weight |
| `epsilon` | 1e-2 | Viscosity Laplacian damping coefficient |
| `beta` | 0.01 | Laplace distribution scale (narrow-band width for coarea) |

---

### `train.py` — Training Pipeline

Orchestrates the multi-resolution optimization loop.

#### Algorithm: Coarse-to-Fine Training

1. **Normalization** — Input points are centered and uniformly scaled into `[0.05, 0.95]³` within the unit cube, preserving aspect ratio.

2. **SDF Initialization** — At the coarsest resolution, the grid is initialized using KD-tree nearest-neighbor distances with normal-based sign determination (see `initialize_from_pointcloud` above). This provides a much better starting point than zero initialization.

3. **Resolution Schedule** — Training proceeds through multiple resolution stages (e.g., 64 → 128 → 256). At each stage:
   - The grid is upsampled from the previous resolution via trilinear interpolation.
   - A fresh Adam optimizer is created (learning rate default 0.001, betas (0.9, 0.999)).
   - Voxels far from the surface are pruned.
   - Multiple epochs are run, each consisting of `iterations_per_epoch` mini-batch steps.
   - At each iteration, a random subset of surface points (size = `batch_size_ratio × N`) is sampled and the total loss is backpropagated through the grid parameters.

4. **Pruning** — After training at each resolution, `prune_voxels(threshold=0.9)` deactivates voxels far from the surface, reducing computation at the next (finer) resolution.

5. **Logging** — Optional TensorBoard integration logs per-iteration and per-epoch losses for each resolution stage.

#### Normal Estimation (`estimate_normals`)

When normals are not provided, they are estimated via **PCA on local neighborhoods**:
- A ball-tree is built over the point cloud (`sklearn.neighbors.NearestNeighbors`).
- For each point, the `k+1` nearest neighbors are found (excluding self).
- The 3×3 covariance matrix of the centered neighbors is eigen-decomposed.
- The eigenvector with the **smallest eigenvalue** is the estimated normal (direction of least variance).
- Orientation is made consistent by ensuring the normal points away from the local centroid.

---

### `datasets.py` — Dataset Loading & Preprocessing

Provides loaders for standard 3D datasets and point cloud preprocessing utilities.

#### Loaders

| Function | Description |
|---|---|
| `load_stanford_bunny` | Searches `data/bunny/` for PLY files, preferring reconstruction files (`bun_zipper.ply`) over raw scans. Uses `find_ply_file` with recursive search and preferred-name ordering. |
| `load_armadillo` / `load_dragon` | Load Stanford Armadillo and Dragon models from expected paths. |
| `load_point_cloud_from_ply` | Generic PLY loader. Uses `trimesh` if available; falls back to a simple ASCII PLY parser that reads vertex positions and optional normals. |
| `load_point_cloud_from_obj` | Loads an OBJ mesh via `trimesh` and samples `N` points on its surface using `trimesh.sample`. |
| `load_shapenet_model` / `load_modelnet_model` | Thin wrappers around the OBJ/PLY loaders for ShapeNet and ModelNet datasets. |

#### Preprocessing

| Function | Algorithm |
|---|---|
| `normalize_point_cloud` | Centers at mean, scales by max extent, maps to `[margin, 1−margin]³`. Returns offset and scale for later denormalization. |
| `downsample_point_cloud` | **Random**: uniform random index selection. **FPS (Farthest Point Sampling)**: iteratively selects the point maximizing minimum distance to the already-selected set, producing a more spatially uniform subset. Current FPS implementation is O(N·M) where M is the target count. |
| `add_noise_to_point_cloud` | Adds isotropic Gaussian noise scaled relative to the point cloud's bounding box extent. |

---

### `mesh_utils.py` — Mesh I/O and Processing

Post-processing utilities for extracted meshes.

#### I/O

| Function | Format | Notes |
|---|---|---|
| `save_mesh_to_obj` / `load_mesh_from_obj` | Wavefront OBJ | Supports vertices, normals, and triangulated faces. Handles `v/vt`, `v//vn`, `v/vt/vn` face formats. Automatically triangulates polygonal faces via fan triangulation. |
| `save_mesh_to_ply` | Stanford PLY | ASCII format with optional per-vertex normals. |
| `save_pointcloud_to_xyz` | XYZ text | Simple space-delimited `x y z [nx ny nz]` format. |

#### Mesh Processing

| Function | Algorithm |
|---|---|
| `fix_triangle_orientations` | **BFS-based consistent orientation propagation**: (1) Builds an edge→face adjacency map. (2) Computes per-face normals via cross products. (3) Starting from an arbitrary seed face, performs BFS over the edge-adjacency graph. (4) For each unvisited neighbor sharing an edge, checks if its normal aligns with the current face's normal (dot product test with threshold −0.1). If misaligned, the neighbor's winding order is flipped. (5) Repeats for all connected components. |
| `invert_face_normals` | Swaps vertex indices 1 and 2 in every triangle, reversing the winding order and thus flipping all face normals. |
| `compute_vertex_normals` | For each face, computes the cross-product normal. Accumulates (area-weighted, since the cross product magnitude is proportional to triangle area) face normals onto each vertex. Normalizes the result per vertex. |

---

### `utils.py` — General Utilities

| Function | Description |
|---|---|
| `load_point_cloud_from_numpy` / `save_point_cloud_to_numpy` | NumPy `.npy` serialization. Supports `(N, 3)` points-only or `(N, 6)` points+normals layout. |
| `normalize_point_cloud` | Centers and uniformly scales points into the unit cube with configurable margin. Returns center and scale for reversibility. |
| `denormalize_point_cloud` | Inverts normalization given stored center and scale, mapping reconstructed geometry back to original world coordinates. |

---

### `example.py` — Synthetic Demos

Demonstrates the full pipeline on procedurally generated point clouds:

- **Sphere** — Uniform random points on a sphere via spherical coordinates (`θ ~ U[0,2π]`, `φ = arccos(U[−1,1])`). Optional Gaussian radial noise controlled by `sigma`.
- **Torus** — Parametric sampling: `(R + r·cos(v))·cos(u), (R + r·cos(v))·sin(u), r·sin(v)` with small additive noise.
- **Visualization** — Four-panel matplotlib figure: (1) input point cloud, (2) extracted mesh via `Poly3DCollection` (renders both front and back faces), (3) SDF cross-section heatmap at the middle z-slice, (4) histogram of all SDF values with zero level-set marked.

### `example_with_dataset.py` — Real Dataset Demos

CLI-driven script with `--bunny`, `--file <path>`, and `--list` options. Loads real data, optionally estimates normals, trains the model, extracts a mesh, and saves to OBJ.

---

## Algorithmic Summary

The VisCo Grids method solves the following optimization problem over a discrete voxel grid φ:

```
min_φ   λ_p · mean(φ(pᵢ)²)                           [Point loss: surface points have SDF ≈ 0]
      + λ_n · mean(|∇φ(pᵢ) − nᵢ|²)                   [Normal loss: gradient matches normals]
      + λ_v · mean[((|∇φ| − 1)·sign(φ) − ε·Δφ)²]     [Viscosity: Eikonal + viscosity correction]
      + λ_c · mean(Φ_β(−φ) · |∇φ|)                     [Coarea: surface area minimization]
```

Key properties:
- **Coarse-to-fine**: Start at low resolution (64³), progressively upsample to high resolution (256³).
- **Voxel pruning**: Only optimize near the surface, making fine resolutions tractable.
- **No neural network**: The SDF is stored directly as grid values — the only learnable parameters are the voxel SDF entries, optimized with Adam.
- **Viscosity regularization**: Goes beyond the standard Eikonal constraint by incorporating the Laplacian term `ε·Δφ`, which selects the viscosity solution and avoids spurious local minima.
- **Coarea regularization**: The Laplace PDF acts as a differentiable approximation to the Dirac delta, concentrating the surface-area penalty in a narrow band around the zero level-set.

The final surface is extracted as `{x : φ(x) = 0}` using the **Marching Cubes** algorithm.
