# VisCo Grids — Project & Algorithm Summary

## 1. Project Overview

This repository is a PyTorch implementation of:

> **"VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids"**
> *Pumarola et al., NeurIPS 2022* — [arXiv 2303.14569](https://arxiv.org/abs/2303.14569)

**Problem.** Given an unorganized 3D point cloud **P = {q₁, …, qₙ}** with optional
per-point normals **{n₁, …, nₙ}**, recover a watertight triangle mesh that
faithfully represents the underlying surface.

**Approach.** The surface is implicitly defined as the zero level-set of a Signed
Distance Function (SDF) **f : ℝ³ → ℝ** stored on a regular voxel grid.  The SDF
is optimized via gradient descent with three complementary loss terms — a data
fidelity term, a *viscosity* regularizer (replacing the classical Eikonal
penalty), and a *coarea* surface-area regularizer.  After optimization the mesh
is extracted with Marching Cubes.

**Why grids instead of neural networks?** Implicit Neural Representations (INRs)
such as DeepSDF parameterize **f** with an MLP.  VisCo Grids replaces the MLP
with a plain 3D tensor + trilinear interpolation:

| Property | INR-based | VisCo Grids |
|---|---|---|
| Training speed | Minutes–hours | Seconds–minutes |
| Inference | Forward pass per query | Single trilinear lookup |
| Memory | Network weights | Sparse voxel grid |
| Expressiveness | Spectral bias issues | Resolution-limited but direct |

---

## 2. End-to-End Pipeline

```
                         ┌──────────────────────────────────────────────┐
  Point Cloud            │         Coarse-to-Fine Optimization         │
  (N×3) + normals        │                                              │
       │                 │   ┌────────┐    ┌────────┐    ┌────────┐    │
       ▼                 │   │  64³   │──▶│  128³  │──▶│  256³  │    │
 ┌───────────┐           │   │ 5 ep.  │    │ 5 ep.  │    │ 3 ep.  │    │
 │ Normalize │──────────▶│   └───┬────┘    └───┬────┘    └───┬────┘    │
 │ to [0,1]³ │           │       │prune+       │prune+       │         │
 └───────────┘           │       │upsample     │upsample     │         │
       │                 └───────┼─────────────┼─────────────┼─────────┘
       │                         │             │             │
       │                         ▼             ▼             ▼
       │                   L = L_data + L_viscosity + L_coarea
       │                         │
       │                         ▼
       │                 ┌───────────────┐
       │                 │ Marching Cubes│
       │                 └──────┬────────┘
       │                        │
       ▼                        ▼
  Input (.ply/.obj)       Output mesh (.obj/.ply)
```

**Steps in detail:**

1. **Load & preprocess** — read PLY/OBJ/NPY; optionally estimate normals via PCA.
2. **Normalize** — translate centroid to origin, uniform-scale into `[0.05, 0.95]³`.
3. **Initialize SDF grid** — KD-tree nearest-neighbour distance with sign from normals.
4. **Optimize** — at each resolution level, run Adam for several epochs of stochastic mini-batches.
5. **Prune** — deactivate voxels whose absolute SDF exceeds a threshold.
6. **Upsample** — trilinear-interpolate the grid to 2× resolution.
7. **Extract mesh** — `skimage.measure.marching_cubes` on the final SDF grid at level 0.

---

## 3. Mathematical Foundations

### 3.1 Signed Distance Functions

A true SDF satisfies the *Eikonal equation*:

```
‖∇f(x)‖ = 1    ∀ x ∈ Ω
```

with `f(x) = 0` on the surface **S**, `f(x) > 0` outside, and `f(x) < 0` inside.

### 3.2 Data Loss

The data term anchors the SDF to the observed point cloud:

```
L_data = λ_p · (1/K) Σₖ |f(qₖ)|²  +  λ_n · (1/K) Σₖ ‖∇f(qₖ) − nₖ‖²
```

- **Point term** (`λ_p = 0.1`): forces the zero level-set to pass through observed
  surface points.  Implemented as `(sdf_values ** 2).mean()` where `sdf_values`
  are obtained by trilinear interpolation at the input points.
- **Normal term** (`λ_n = 1e-5`): aligns the SDF gradient with observed normals.
  Gradients `∇f(qₖ)` are computed via `torch.autograd.grad` through the
  differentiable trilinear interpolation, giving exact analytical derivatives.
  Only active when normals are provided; otherwise contributes zero.

### 3.3 Viscosity Loss (key contribution)

The naïve Eikonal penalty `(‖∇f‖ − 1)²` has many spurious local minima
(e.g., the trivial solution `f ≡ 0`).  The *viscosity solution* from PDE theory
is the unique physically meaningful solution.  VisCo Grids approximates it by
adding a vanishing diffusion term:

```
L_visc = (1/M) Σⱼ [ (‖∇f(xⱼ)‖ − 1) · sign(f(xⱼ)) − ε · Δf(xⱼ) ]²
```

where:

- **xⱼ** are all grid nodes (the sum runs over the full R×R×R grid).
- **∇f** is the spatial gradient via central finite differences:
  ```
  (∂f/∂x)ᵢⱼₖ = (f[i+1,j,k] − f[i−1,j,k]) / (2h)
  ```
- **Δf** is the discrete Laplacian:
  ```
  Δf[i,j,k] = (f[i+1,j,k] + f[i−1,j,k]
              + f[i,j+1,k] + f[i,j−1,k]
              + f[i,j,k+1] + f[i,j,k−1]
              − 6·f[i,j,k]) / h²
  ```
- **ε = 0.01** controls diffusion strength.
- **sign(f)** couples the Eikonal deviation to inside/outside classification.

**Why this works:** The term `sign(f)·(‖∇f‖−1)` penalizes gradient magnitudes
≠ 1 with opposite sign inside vs. outside.  The Laplacian `ε·Δf` acts as a
smoothing bias that breaks the symmetry of bad local minima.  Together they
select the unique viscosity solution of the Eikonal equation.

**Implementation detail:** The grid is zero-padded by one cell on each side
before computing differences.  This means boundary voxels see zero-valued
neighbours, which biases boundary SDF values toward positive (outside).  Both
gradient and Laplacian are computed in a single `compute_finite_differences()`
call that returns tensors of shape `(R, R, R, 3)` and `(R, R, R)` respectively.

### 3.4 Coarea Loss (key contribution)

Without area regularization the optimizer can produce "ghost" surfaces — extra
zero level-sets far from the data.  The coarea formula from geometric measure
theory states:

```
Area(S) = ∫_Ω δ(f(x)) · ‖∇f(x)‖ dx
```

Since the Dirac delta is intractable, it is approximated by a Laplace PDF:

```
Φ_β(s) = (1 / 2β) · exp(−|s| / β)
```

giving the differentiable coarea loss:

```
L_coarea = (1/M) Σⱼ Φ_β(−f(xⱼ)) · ‖∇f(xⱼ)‖
```

- **β = 0.01** controls the width of the approximation.  Smaller β → sharper
  peak → closer to true surface area, but harder to optimize.
- Evaluated at all active voxels using the same finite-difference gradients as
  the viscosity loss.
- Gradient norms are clamped to `[1e-6, 10.0]` and `|s|` is clamped to
  `10·β` to prevent numerical overflow in the exponential.

**Behaviour during training:** As the SDF becomes more accurate, values near the
surface approach 0 and `Φ_β(−f)` increases (peaks at `f = 0`).  Simultaneously
gradient norms approach 1.  This can cause the coarea loss to temporarily
increase before the surface smooths out and the loss decreases.

### 3.5 Total Loss

```
L_total = λ_p · L_point + λ_n · L_normal + λ_v · L_visc + λ_c · L_coarea
```

### 3.6 Hyperparameter Summary

| Symbol | Parameter | Default | Role |
|--------|-----------|---------|------|
| `λ_p` | Point weight | 0.1 | Strength of zero level-set constraint |
| `λ_n` | Normal weight | 1e-5 | Strength of gradient-normal alignment |
| `λ_v` | Viscosity weight | 1e-4 | Eikonal + diffusion regularization |
| `λ_c` | Coarea weight | 1e-6 | Surface area minimization |
| `ε` | Viscosity diffusion | 1e-2 | Laplacian smoothing scale |
| `β` | Laplace scale | 0.01 | Coarea delta approximation width |
| `lr` | Learning rate | 0.001 | Adam step size |
| `β₁, β₂` | Adam momenta | 0.9, 0.999 | Optimizer parameters |

---

## 4. Core Algorithms in Detail

### 4.1 Trilinear Interpolation (`trilinear_interpolate`)

Given a query point **p** ∈ [0,1]³ and a grid of resolution R with spacing
`h = 1/R`, the continuous SDF value is computed as:

```
p_grid = p / h                          # map to grid coordinates
p_grid = clamp(p_grid, 0, R−1)
(i₀, j₀, k₀) = floor(p_grid)          # lower corner indices
(i₁, j₁, k₁) = min(i₀+1, R−1)        # upper corner indices (clamped)
(dx, dy, dz) = p_grid − (i₀, j₀, k₀)  # fractional offsets ∈ [0,1)

f(p) = f[i₀,j₀,k₀]·(1−dx)(1−dy)(1−dz)
     + f[i₁,j₀,k₀]·  dx ·(1−dy)(1−dz)
     + f[i₀,j₁,k₀]·(1−dx)· dy ·(1−dz)
     + f[i₀,j₀,k₁]·(1−dx)(1−dy)· dz
     + f[i₁,j₁,k₀]·  dx · dy ·(1−dz)
     + f[i₁,j₀,k₁]·  dx ·(1−dy)· dz
     + f[i₀,j₁,k₁]·(1−dx)· dy · dz
     + f[i₁,j₁,k₁]·  dx · dy · dz
```

The implementation indexes the grid tensor directly with integer index tensors,
making it fully vectorized over N query points.  Because all operations are
standard PyTorch tensor ops, `torch.autograd` can differentiate through the
interpolation to compute `∇f(p)` exactly.

**Fallback gradient:** If autograd is unavailable (e.g., `requires_grad=False`),
`compute_gradient()` falls back to forward finite differences with step `1e-5`.

### 4.2 Finite-Difference Gradient and Laplacian (`compute_finite_differences`)

For the viscosity and coarea losses, derivatives are needed at *all grid nodes*
simultaneously.  The implementation:

1. Creates a zero-padded grid of shape `(R+2, R+2, R+2)`.
2. Copies `sdf_grid` into the interior `[1:R+1, 1:R+1, 1:R+1]`.
3. Computes first-order central differences via tensor slicing:
   ```python
   dx = (padded[2:, 1:-1, 1:-1] - padded[:-2, 1:-1, 1:-1]) / (2*h)
   dy = (padded[1:-1, 2:, 1:-1] - padded[1:-1, :-2, 1:-1]) / (2*h)
   dz = (padded[1:-1, 1:-1, 2:] - padded[1:-1, 1:-1, :-2]) / (2*h)
   ```
4. Computes second-order differences for the Laplacian:
   ```python
   d2x = (padded[2:, 1:-1, 1:-1] - 2*grid + padded[:-2, 1:-1, 1:-1]) / h²
   ```
5. Returns `gradients` of shape `(R, R, R, 3)` and `laplacians` of shape `(R, R, R)`.

All operations are pure tensor slicing — no loops — so they run efficiently on
GPU.  The zero-padding means boundary voxels effectively see `f = 0` outside the
domain, which is consistent with the SDF being positive (outside) far from the
surface.

### 4.3 KD-Tree SDF Initialization (`initialize_from_pointcloud`)

Rather than starting from a zero grid, the SDF is initialized to a reasonable
approximation using `sklearn.neighbors.KDTree`:

1. Build a KDTree over the N input points.
2. Create a regular grid of R³ voxel centers at positions `((i+0.5)/R)` for
   `i = 0, …, R−1` in each dimension.
3. For each voxel center **x**, query the `k` nearest points (default k=5).
4. **Unsigned distance:** `d = distance to nearest point`.
5. **Sign determination (when normals available):**
   - Compute vector `v = q_nn − x` (from voxel center toward nearest point).
   - Compute average normal `n̄` of the k nearest neighbours.
   - `sign = −sign(⟨v, n̄⟩)`:
     - If **x** is outside, **v** points toward the surface (opposite to **n̄**),
       so `⟨v, n̄⟩ < 0` and `sign = +1`.
     - If **x** is inside, **v** points toward the surface (same direction as
       **n̄**), so `⟨v, n̄⟩ > 0` and `sign = −1`.
   - If the dot product is ambiguous (near zero), a majority vote over all k
     neighbours is used.
6. **Sign determination (no normals):** All signs default to +1 (positive/outside).
7. Set `sdf_grid[i,j,k] = sign · d`.

This initialization dramatically accelerates convergence compared to starting
from zeros, because the optimizer only needs to refine the SDF rather than
discover the entire distance field from scratch.

### 4.4 Voxel Pruning (`prune_voxels`)

After each resolution stage, voxels far from the surface are deactivated:

```python
active_mask = (|sdf_grid| < threshold)   # default threshold = 0.9
```

The `active_mask` is a boolean buffer (not a parameter) of shape `(R, R, R)`.
It is used in the coarea loss to restrict the sum to active voxels only.  The
viscosity loss currently sums over *all* grid nodes (including inactive ones)
because it operates on the full finite-difference tensors.

**Note:** Pruning does not remove voxels from memory — the full `(R, R, R)`
tensor is always allocated.  The mask only controls which voxels contribute to
certain loss terms.  True sparse storage is listed as a future improvement.

### 4.5 Grid Upsampling (`upsample_grid`)

Between resolution stages the grid is upsampled:

1. Save the old grid data and resolution.
2. Create a temporary `VisCoGrids` instance at the old resolution.
3. Generate a regular grid of `R_new³` query points in [0, 1)³.
4. Evaluate the old grid at each new point via `trilinear_interpolate()`.
5. Reshape the result to `(R_new, R_new, R_new)` and assign as the new
   `nn.Parameter`.
6. Reset `active_mask` to all-True at the new resolution.

The upsampled grid provides a warm start.  The active mask is re-pruned after
the next training stage.

### 4.6 Marching Cubes Mesh Extraction (`extract_mesh`)

Uses `skimage.measure.marching_cubes` with coordinate handling:

1. The internal SDF grid is stored in `(x, y, z)` order.
2. scikit-image expects `(z, y, x)` order, so the grid is transposed:
   `sdf_for_mc = np.transpose(sdf_np, (2, 1, 0))`.
3. Spacing is similarly reordered: `(h_z, h_y, h_x)`.
4. After marching cubes, vertices are swapped back: `(z,y,x) → (x,y,z)`.
5. Triangle orientations are fixed via BFS propagation
   (`fix_triangle_orientations`), then all faces are flipped
   (`invert_face_normals`) to ensure outward-pointing normals.

### 4.7 Normal Estimation via PCA (`estimate_normals`)

When input normals are unavailable:

1. Build a `sklearn.neighbors.NearestNeighbors` index (ball tree, k+1 neighbours
   including self).
2. For each point, extract the k nearest neighbours (excluding self).
3. Center the neighbourhood: `N_centered = N − mean(N)`.
4. Compute covariance: `C = N_centeredᵀ · N_centered`.
5. Eigendecompose: `eigenvalues, eigenvectors = eigh(C)`.
6. The normal is the eigenvector with the *smallest* eigenvalue (direction of
   least variance = surface normal direction).
7. **Orientation heuristic:** flip the normal so it points away from the local
   centroid: `if n · (pᵢ − mean(neighbours)) < 0 then n = −n`.

**Limitation:** This heuristic does not guarantee globally consistent
orientation.  For complex shapes with concavities, some normals may be flipped.
Providing pre-oriented normals yields better results.

---

## 5. Module-by-Module Reference

### 5.1 `visco_grids.py` — Core Model

**Class `VisCoGrids(nn.Module)`**

The single learnable parameter is `sdf_grid`, an `nn.Parameter` of shape
`(R, R, R)` initialized to zeros.  The `active_mask` is a registered buffer
(not optimized) of the same shape.

| Method | Algorithm | Complexity |
|--------|-----------|------------|
| `trilinear_interpolate(points)` | 8-corner weighted sum (§4.1) | O(N) |
| `compute_gradient(points)` | `torch.autograd.grad` through interp | O(N) |
| `compute_finite_differences()` | Padded central differences (§4.2) | O(R³) |
| `compute_gradient_at_voxel_centers()` | Autograd at voxel centers | O(R³) |
| `initialize_from_pointcloud(pts, norms, k)` | KD-tree + signed dist (§4.3) | O(R³ log N) |
| `data_loss(points, normals)` | Point + normal MSE (§3.2) | O(B) per batch |
| `viscosity_loss()` | Eikonal + Laplacian penalty (§3.3) | O(R³) |
| `coarea_loss()` | Laplace-weighted grad magnitude (§3.4) | O(R³) |
| `total_loss(points, normals)` | Weighted sum of all losses | O(R³ + B) |
| `prune_voxels(threshold)` | Threshold mask update (§4.4) | O(R³) |
| `upsample_grid(new_resolution)` | Trilinear resampling (§4.5) | O(R_new³) |
| `extract_mesh(level)` | Marching cubes + orientation fix (§4.6) | O(R³) |
| `get_surface_points(threshold)` | Threshold `|f| < threshold` | O(R³) |

**Gradient computation — two code paths:**

1. **For data loss** (`compute_gradient`): Uses `torch.autograd.grad` through
   `trilinear_interpolate`.  This gives exact analytical gradients at arbitrary
   query points and supports `create_graph=True` for second-order optimization.
2. **For viscosity/coarea losses** (`compute_finite_differences`): Uses tensor-
   slicing central differences on the full grid.  Much faster than calling
   autograd at every voxel center.

### 5.2 `train.py` — Training Pipeline

**`train_visco_grids()`** orchestrates the full coarse-to-fine loop:

```
Determine resolution schedule:
  final=256 → [64, 128, 256]
  final=128 → [64, 128]
  otherwise  → double until final

for res_idx, resolution in enumerate(resolutions):
    if res_idx > 0:
        upsample grid from previous resolution
    else:
        create model, initialize SDF from point cloud

    optimizer = Adam(model.parameters(), lr=0.001)
    prune voxels (threshold=0.9)

    for epoch in range(epochs_per_resolution[res_idx]):
        for iteration in range(iterations_per_epoch):  # default 12,800
            batch = random sample of (batch_size_ratio × N) points
            loss = model.total_loss(batch_points, batch_normals)
            loss.backward()
            optimizer.step()
            log to TensorBoard (per-iteration)
        log epoch averages to TensorBoard

    prune voxels again
    optionally save intermediate mesh
```

**Key design decisions:**
- A fresh Adam optimizer is created at each resolution level (momentum buffers
  are not carried over).
- Pruning happens both *before* training (to set the initial mask) and *after*
  training (to prepare for the next level).
- The batch samples points from the *input point cloud*, not from voxel centers.
  Voxel-center losses (viscosity, coarea) always use the full grid.

**TensorBoard logging:** When enabled, logs 7 scalar values per iteration
(`Total`, `Point`, `Normal`, `Viscosity`, `Coarea`, `Data`, `Prior`) tagged by
resolution, plus 5 epoch-averaged values.

### 5.3 `datasets.py` — Data Loading & Preprocessing

**Loading pipeline per format:**

- **PLY** (`load_point_cloud_from_ply`):
  - With `trimesh`: loads as `PointCloud` or `Trimesh`.  For meshes, returns
    vertex positions directly (up to 10K sampled points).
  - Without `trimesh`: falls back to a simple ASCII PLY parser that reads
    lines after `end_header`, splitting on whitespace for x,y,z (and
    optionally nx,ny,nz).
- **OBJ** (`load_point_cloud_from_obj`): Requires `trimesh`.  Calls
  `mesh.sample(sample_points)` for uniform surface sampling.
- **Stanford models** (`load_stanford_bunny`, `load_dragon`, `load_armadillo`):
  Search for PLY files in expected directory structures.  The bunny loader
  prefers `reconstruction/bun_zipper.ply` (the merged mesh) over individual
  scan files.

**`normalize_point_cloud(points, margin=0.05)`:**

```python
center = points.mean(dim=0)
points = points - center
scale = (max - min).max()
points = points / scale
points = points * (1 - 2*margin) + 0.5   # → [margin, 1-margin]³
```

Returns `(normalized, center, scale)` for later denormalization.

**`downsample_point_cloud(points, num_points, method)`:**
- `'random'`: `torch.randperm(N)[:num_points]` — O(N).
- `'fps'`: Greedy farthest-point sampling — O(N · num_points) with a naive
  inner loop.  Produces more spatially uniform subsets but is slow for large N.

**`add_noise_to_point_cloud(points, noise_level)`:**
Adds isotropic Gaussian noise scaled to `noise_level × extent` of the point
cloud.

### 5.4 `mesh_utils.py` — Mesh I/O & Processing

**I/O functions** handle OBJ and PLY with manual line-by-line parsing (no heavy
dependencies beyond numpy):

- **OBJ writer** (`save_mesh_to_obj`): Writes `v`, optionally `vn`, then `f`
  lines with 1-based indexing.  When normals are present, uses `f v//vn` format.
- **OBJ reader** (`load_mesh_from_obj`): Parses `v`, `vn`, `f` lines.  Handles
  `v/vt`, `v//vn`, `v/vt/vn` face formats.  Automatically triangulates
  polygonal faces via fan triangulation (first vertex + consecutive pairs).
- **PLY writer** (`save_mesh_to_ply`): ASCII PLY with vertex properties
  `(x, y, z, [nx, ny, nz])` and face property list `vertex_indices`.

**`fix_triangle_orientations(vertices, faces)`** — BFS-based consistent winding:

1. Build edge→face adjacency: for each triangle edge `(v₁, v₂)` (canonicalized
   as `(min, max)`), record which faces share it.
2. Compute initial face normals via cross product.
3. BFS from face 0.  For each unvisited neighbour sharing an edge:
   - Compute dot product of current face normal with neighbour normal.
   - If `dot < −0.1` (normals point in opposite directions), flip the
     neighbour by swapping vertices 1 and 2: `face = face[[0, 2, 1]]`.
   - Negate the stored normal for the flipped face.
4. Process all connected components (restart BFS from first unvisited face).

**`compute_vertex_normals(vertices, faces)`** — Area-weighted averaging:

```
for each face (v₀, v₁, v₂):
    face_normal = (v₁−v₀) × (v₂−v₀)     # magnitude ∝ 2×area
    normals[v₀] += face_normal
    normals[v₁] += face_normal
    normals[v₂] += face_normal
normalize each vertex normal to unit length
```

The cross product is *not* normalized before accumulation, so larger triangles
contribute proportionally more — this is the standard area-weighted scheme.

### 5.5 `utils.py` — General Utilities

Provides NumPy-based I/O and normalization that mirrors `datasets.py` but with
a simpler interface:

- **`load_point_cloud_from_numpy(filepath, has_normals)`**: Loads `.npy` files.
  If `has_normals=True`, expects shape `(N, 6)` and splits into points + normals.
- **`save_point_cloud_to_numpy(points, filepath, normals)`**: Concatenates
  points and normals (if provided) and saves as `.npy`.
- **`normalize_point_cloud(points, margin)`**: Same algorithm as `datasets.py`
  version.  Returns `(normalized, center, scale)`.
- **`denormalize_point_cloud(normalized, center, scale, margin)`**: Inverse:
  ```
  points = (normalized − 0.5) / (1 − 2·margin) · scale + center
  ```

Also attempts to import `save_mesh_to_obj`, `load_mesh_from_obj`, and
`compute_vertex_normals` from `mesh_utils` for convenience re-export.

### 5.6 `example.py` — Synthetic Demonstrations

**Point cloud generators:**

- **`generate_sphere_point_cloud(num_points, radius, sigma)`**:
  - Uniform directions via: `θ ~ U(0, 2π)`, `φ = arccos(U(−1, 1))`.
  - Radial distance: `r = radius + N(0, σ)` (clamped to ≥ 0.01).
  - When `σ = 0`, points lie exactly on the sphere.
  - Normals are the unit direction vectors (not the noisy point directions).

- **`generate_torus_point_cloud(num_points, R, r)`**:
  - Parametric: `x = (R + r·cos v)·cos u`, etc., with `u, v ~ U(0, 2π)`.
  - Gaussian noise `N(0, 0.01)` added to all coordinates.
  - Normals computed as `(point − tube_center) / ‖…‖` where `tube_center`
    is the closest point on the major circle.

**`visualize_results(points, model, title, show_mesh)`** — 4-panel figure:

1. **3D scatter** of input points.
2. **3D mesh** from marching cubes (subsampled to 10K faces for rendering).
   Both front and back faces are rendered by adding reversed-winding copies.
3. **2D heatmap** of the SDF at the grid's middle z-slice (`z = R/2`), using
   `RdYlBu` colormap.
4. **Histogram** of all R³ SDF values with a red dashed line at zero.

### 5.7 `example_with_dataset.py` — Real Data Demonstrations

Wraps dataset loaders and training into CLI-driven examples:

- **`--bunny`**: Loads Stanford Bunny, downsamples to 5000 points if needed,
  estimates normals, trains at 64→128, saves OBJ.
- **`--file <path>`**: Loads any PLY or OBJ file, same pipeline.
- **`--list`**: Prints the `DATASET_INFO` dictionary.

---

## 6. Coarse-to-Fine Strategy — Detailed Analysis

| Stage | Resolution | Total Voxels | Default Epochs | Iters/Epoch | Purpose |
|-------|-----------|-------------|----------------|-------------|---------|
| 1 | 64³ | 262,144 | 5 | 12,800 | Global topology |
| 2 | 128³ | 2,097,152 | 5 | 12,800 | Medium features |
| 3 | 256³ | 16,777,216 | 3 | 12,800 | Fine detail |

**Why it works:**

- At 64³ each voxel spans `1/64 ≈ 0.016` of the domain.  The loss landscape is
  smooth and the optimizer quickly finds the coarse shape.
- Upsampling to 128³ preserves the coarse solution while doubling spatial
  resolution.  The optimizer only needs to refine details.
- Pruning between stages deactivates voxels with `|f| > 0.9`, typically
  eliminating 80–95% of voxels from the coarea loss computation.
- Fewer epochs at 256³ suffice because the warm start is already close.

**Optimizer reset:** A fresh Adam optimizer is created at each resolution.  This
avoids stale momentum estimates from the previous (lower-resolution) grid, which
would have different parameter shapes.

---

## 7. Typical Usage

```python
import torch
from train import train_visco_grids, estimate_normals
from datasets import load_point_cloud_from_ply, normalize_point_cloud
from mesh_utils import save_mesh_to_obj

# 1. Load point cloud
points, normals = load_point_cloud_from_ply("input.ply")

# 2. Normalize to unit cube
points_norm, center, scale = normalize_point_cloud(points)

# 3. Estimate normals if not available
if normals is None:
    normals = estimate_normals(points_norm, k=10)

# 4. Train model (coarse-to-fine)
model = train_visco_grids(
    points=points_norm,
    normals=normals,
    initial_resolution=64,
    final_resolution=256,
    verbose=True
)

# 5. Extract mesh at zero level-set
vertices, faces = model.extract_mesh(level=0.0)

# 6. Save output
save_mesh_to_obj(vertices, faces, "output.obj")
```

---

## 8. File Structure

```
VisCo-Grids/
├── visco_grids.py            # Core VisCoGrids model (SDF grid, losses, mesh extraction)
├── train.py                  # Coarse-to-fine training loop, normal estimation
├── datasets.py               # Point cloud loading (PLY/OBJ), dataset helpers
├── mesh_utils.py             # Mesh I/O (OBJ/PLY/XYZ), orientation fixing
├── utils.py                  # NumPy I/O, normalization / denormalization
├── example.py                # Synthetic sphere/torus demos with visualization
├── example_with_dataset.py   # Real dataset demos (Stanford Bunny, custom files)
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview & quick start
├── DATASETS.md               # Dataset download & usage guide
├── IMPLEMENTATION.md         # Technical implementation notes
├── summary.md                # This file
└── .gitignore                # Git ignore rules
```

---

## 9. Dependencies

| Package | Min Version | Role in Project |
|---------|-------------|-----------------|
| `torch` | ≥ 1.9.0 | Learnable SDF grid, autograd gradients, Adam optimizer |
| `numpy` | ≥ 1.19.0 | Array operations, marching cubes interface |
| `scikit-learn` | ≥ 0.24.0 | KDTree for SDF init, NearestNeighbors for normals |
| `scikit-image` | ≥ 0.18.0 | `marching_cubes` for mesh extraction |
| `trimesh` | ≥ 3.9.0 | PLY/OBJ loading, mesh surface sampling |
| `matplotlib` | ≥ 3.3.0 | Visualization (point clouds, SDF slices, meshes) |
| `tensorboard` | ≥ 2.0.0 | Training loss logging and monitoring |

All except `torch` and `numpy` degrade gracefully — the code checks for their
presence at import time and prints warnings or falls back to simpler
implementations.

---

## 10. References

- **Paper**: Pumarola et al., *VisCo Grids: Surface Reconstruction with
  Viscosity and Coarea Grids*, NeurIPS 2022.
  [arXiv:2303.14569](https://arxiv.org/abs/2303.14569)
- **Viscosity solutions**: Crandall & Lions, *Viscosity solutions of
  Hamilton-Jacobi equations*, Trans. AMS, 1983.
- **Coarea formula**: Federer, *Geometric Measure Theory*, 1969.
- **Marching Cubes**: Lorensen & Cline, *Marching Cubes: A High Resolution 3D
  Surface Construction Algorithm*, SIGGRAPH 1987.
- **Stanford 3D Repository**: http://graphics.stanford.edu/data/3Dscanrep/
- **ShapeNet**: https://www.shapenet.org/
- **ModelNet**: https://modelnet.cs.princeton.edu/
