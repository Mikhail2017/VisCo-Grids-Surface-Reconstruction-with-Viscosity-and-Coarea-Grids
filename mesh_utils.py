"""
Utilities for mesh I/O and processing.
"""

import numpy as np
from typing import Tuple, Optional


def save_mesh_to_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    normals: Optional[np.ndarray] = None
):
    """
    Save mesh to OBJ file format.
    
    Args:
        vertices: Vertex positions of shape (N, 3)
        faces: Face indices of shape (M, 3)
        filepath: Path to save OBJ file
        normals: Optional vertex normals of shape (N, 3)
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write normals if provided
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        
        # Write faces (OBJ uses 1-based indexing)
        if normals is not None:
            for face in faces:
                # Format: f v1//vn1 v2//vn2 v3//vn3
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")
        else:
            for face in faces:
                # Format: f v1 v2 v3
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def load_mesh_from_obj(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load mesh from OBJ file format.
    
    Args:
        filepath: Path to OBJ file
        
    Returns:
        vertices: Vertex positions of shape (N, 3)
        faces: Face indices of shape (M, 3)
        normals: Optional vertex normals of shape (N, 3)
    """
    vertices = []
    faces = []
    normals = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                # Vertex
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn':
                # Normal
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # Face (handle different formats)
                face_verts = []
                for part in parts[1:]:
                    # Handle formats like: v, v/vt, v//vn, v/vt/vn
                    vertex_idx = int(part.split('/')[0]) - 1  # OBJ uses 1-based indexing
                    face_verts.append(vertex_idx)
                if len(face_verts) >= 3:
                    # Triangulate if needed (simple: first vertex + pairs)
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i+1]])
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    normals = np.array(normals) if normals else None
    
    return vertices, faces, normals


def save_mesh_to_ply(
    vertices: np.ndarray,
    faces: np.ndarray,
    filepath: str,
    normals: Optional[np.ndarray] = None
):
    """
    Save mesh to PLY file format.
    
    Args:
        vertices: Vertex positions of shape (N, 3)
        faces: Face indices of shape (M, 3)
        filepath: Path to save PLY file
        normals: Optional vertex normals of shape (N, 3)
    """
    with open(filepath, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for i, v in enumerate(vertices):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            if normals is not None:
                n = normals[i]
                f.write(f" {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            f.write("\n")
        
        # Write faces (PLY uses 0-based indexing)
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def save_pointcloud_to_xyz(
    points: np.ndarray,
    filepath: str,
    normals: Optional[np.ndarray] = None
):
    """
    Save point cloud to XYZ file format.
    
    Args:
        points: Point positions of shape (N, 3)
        filepath: Path to save XYZ file
        normals: Optional normals of shape (N, 3)
    """
    with open(filepath, 'w') as f:
        for i, point in enumerate(points):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
            if normals is not None:
                normal = normals[i]
                f.write(f" {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")
            f.write("\n")


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute vertex normals from mesh faces.
    
    Args:
        vertices: Vertex positions of shape (N, 3)
        faces: Face indices of shape (M, 3)
        
    Returns:
        normals: Vertex normals of shape (N, 3)
    """
    normals = np.zeros_like(vertices)
    
    # Compute face normals
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-8)
        
        # Add to vertex normals
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / (norms + 1e-8)
    
    return normals

