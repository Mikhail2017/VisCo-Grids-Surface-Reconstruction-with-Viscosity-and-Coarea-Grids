# Datasets for VisCo Grids Experiments

This document lists recommended datasets for experimenting with VisCo Grids surface reconstruction.

## Recommended Datasets

### 1. Stanford 3D Scanning Repository ⭐ (Mentioned in Paper)

**URL**: http://graphics.stanford.edu/data/3Dscanrep/

**Description**: Classic 3D models used in many surface reconstruction papers, including the VisCo Grids paper.

**Models Available**:
- **Bunny**: ~35K points, iconic test model
- **Dragon**: ~437K points, complex geometry
- **Armadillo**: ~173K points, detailed surface
- **Buddha**: ~543K points, intricate details
- **Lucy**: ~14M points, high resolution

**Format**: PLY files

**Usage**:
```python
from datasets import load_point_cloud_from_ply, normalize_point_cloud

# Load point cloud
points, normals = load_point_cloud_from_ply("data/bunny/bunny.ply")

# Normalize to [0, 1]^3
points_normalized, center, scale = normalize_point_cloud(points)
```

**Download**: 
- Direct download links available on the website
- No registration required
- Free for research use

**Best For**: Testing basic reconstruction, comparing with paper results

---

### 2. ShapeNet

**URL**: https://www.shapenet.org/

**Description**: Large-scale 3D shape dataset with 3+ million models across thousands of categories.

**Statistics**:
- 3,000,000+ models
- 55 common object categories
- Various resolutions

**Format**: OBJ files

**Usage**:
```python
from datasets import load_point_cloud_from_obj, normalize_point_cloud

# Load and sample points from mesh
points, normals = load_point_cloud_from_obj(
    "path/to/shapenet/model.obj",
    sample_points=10000
)

# Normalize
points_normalized, center, scale = normalize_point_cloud(points)
```

**Download**: 
- Requires registration
- Free for academic use
- Large download size (~1TB for full dataset)

**Best For**: Large-scale experiments, category-specific reconstruction

---

### 3. ModelNet

**URL**: https://modelnet.cs.princeton.edu/

**Description**: Popular benchmark dataset for 3D object recognition and reconstruction.

**Variants**:
- **ModelNet10**: 10 categories, 4,899 models
- **ModelNet40**: 40 categories, 12,311 models

**Format**: OFF/OBJ files

**Usage**:
```python
from datasets import load_modelnet_model, normalize_point_cloud

# Load model
points, normals = load_modelnet_model(
    "path/to/modelnet/model.off",
    sample_points=10000
)

# Normalize
points_normalized, center, scale = normalize_point_cloud(points)
```

**Download**: 
- Direct download available
- No registration required
- Free for research

**Best For**: Benchmarking, category-based experiments

---

### 4. ABC Dataset

**URL**: https://deep-geometry.github.io/abc-dataset/

**Description**: Large dataset of CAD models for geometric deep learning.

**Statistics**:
- 1,000,000+ CAD models
- High-quality geometric data
- Various complexity levels

**Format**: OBJ, STEP

**Usage**:
```python
from datasets import load_point_cloud_from_obj, normalize_point_cloud

points, normals = load_point_cloud_from_obj("path/to/abc/model.obj")
points_normalized, center, scale = normalize_point_cloud(points)
```

**Download**: 
- Requires registration
- Free for research
- Large dataset

**Best For**: High-quality CAD model reconstruction

---

### 5. ScanNet

**URL**: http://www.scan-net.org/

**Description**: Real-world indoor scene scans with RGB-D data.

**Statistics**:
- 1,513 scans
- 2.5M views
- 21 semantic classes

**Format**: PLY, RGB-D sequences

**Note**: Real-world scans may need preprocessing (noise removal, downsampling)

**Best For**: Real-world scene reconstruction (advanced)

---

### 6. ETH 3D

**URL**: https://www.eth3d.net/

**Description**: Multi-view stereo dataset with real-world outdoor scenes.

**Format**: PLY point clouds

**Note**: Outdoor scenes, may need preprocessing

**Best For**: Outdoor scene reconstruction (advanced)

---

## Quick Start

### Using the Dataset Utilities

```python
from datasets import (
    load_point_cloud_from_ply,
    normalize_point_cloud,
    downsample_point_cloud,
    print_dataset_info
)

# Print available datasets
print_dataset_info()

# Load a point cloud
points, normals = load_point_cloud_from_ply("path/to/pointcloud.ply")

# Normalize to [0, 1]^3 (required for VisCo Grids)
points_normalized, center, scale = normalize_point_cloud(points)

# Downsample if needed
if len(points_normalized) > 5000:
    points_normalized = downsample_point_cloud(points_normalized, 5000)

# Train VisCo Grids
from train import train_visco_grids
model = train_visco_grids(
    points=points_normalized,
    normals=normals,
    initial_resolution=64,
    final_resolution=128
)
```

### Example Script

```bash
# Run with Stanford Bunny
python example_with_dataset.py --bunny

# Run with custom file
python example_with_dataset.py --file path/to/pointcloud.ply

# List available datasets
python example_with_dataset.py --list
```

## Dataset Preparation Tips

1. **Normalization**: Always normalize point clouds to [0, 1]³ using `normalize_point_cloud()`

2. **Downsampling**: For large point clouds (>10K points), consider downsampling:
   ```python
   points = downsample_point_cloud(points, num_points=5000)
   ```

3. **Normal Estimation**: If normals are not available:
   ```python
   from train import estimate_normals
   normals = estimate_normals(points, k=10)
   ```

4. **Noise**: Real-world scans may have noise. Consider preprocessing:
   ```python
   # Add slight noise for robustness testing
   from datasets import add_noise_to_point_cloud
   points_noisy = add_noise_to_point_cloud(points, noise_level=0.01)
   ```

5. **File Formats**: Supported formats:
   - PLY (point clouds and meshes)
   - OBJ (meshes, will be sampled to points)
   - OFF (meshes)
   - NumPy arrays (`.npy`)

## Dataset Comparison

| Dataset | Size | Format | Registration | Best For |
|---------|------|--------|--------------|----------|
| Stanford 3D | Small | PLY | No | Quick tests, paper comparison |
| ShapeNet | Large | OBJ | Yes | Large-scale, diverse shapes |
| ModelNet | Medium | OFF/OBJ | No | Benchmarking |
| ABC | Large | OBJ/STEP | Yes | High-quality CAD |
| ScanNet | Large | PLY | Yes | Real-world scenes |
| ETH 3D | Medium | PLY | No | Outdoor scenes |

## References

- Stanford 3D: Turk, G. and Levoy, M. "Zippered polygon meshes from range images"
- ShapeNet: Chang et al. "ShapeNet: An Information-Rich 3D Model Repository"
- ModelNet: Wu et al. "3D ShapeNets: A Deep Representation for Volumetric Shapes"
- ABC: Koch et al. "ABC: A Big CAD Model Dataset For Geometric Deep Learning"
- ScanNet: Dai et al. "ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes"
- ETH 3D: Schöps et al. "A Multi-View Stereo Benchmark with High-Resolution Images"

