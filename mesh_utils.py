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


def invert_face_normals(faces: np.ndarray) -> np.ndarray:
    """
    Invert all face normals by flipping triangle orientation.
    
    Args:
        faces: Face indices of shape (M, 3)
        
    Returns:
        faces: Faces with inverted normals (swapped vertex order)
    """
    # Flip each triangle by swapping two vertices
    return faces[:, [0, 2, 1]]


def fix_triangle_orientations(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Fix triangle orientations to be locally consistent.
    Ensures neighboring triangles have aligned normals.
    
    Uses BFS to propagate consistent orientations through the mesh.
    
    Args:
        vertices: Vertex positions of shape (N, 3)
        faces: Face indices of shape (M, 3)
        
    Returns:
        faces: Faces with consistent orientations
    """
    if len(faces) == 0:
        return faces
    
    faces = faces.copy()
    num_faces = len(faces)
    
    # Build edge-face adjacency: edge -> list of (face_idx, edge_position_in_face)
    edge_to_faces = {}
    for face_idx, face in enumerate(faces):
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            # Use canonical edge representation (smaller index first)
            edge = tuple(sorted([int(v1), int(v2)]))
            if edge not in edge_to_faces:
                edge_to_faces[edge] = []
            edge_to_faces[edge].append((face_idx, i))
    
    # Compute face normals
    def compute_face_normal(face):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return normal / norm
        return np.array([0.0, 0.0, 1.0])  # Default normal for degenerate faces
    
    face_normals = np.array([compute_face_normal(face) for face in faces])
    
    # Track which faces have been processed and their orientation
    processed = np.zeros(num_faces, dtype=bool)
    face_flipped = np.zeros(num_faces, dtype=bool)
    
    # BFS to propagate consistent orientations
    from collections import deque
    
    # Process all connected components
    while not processed.all():
        # Find first unprocessed face
        start_face = np.where(~processed)[0][0]
        queue = deque([start_face])
        processed[start_face] = True
        
        while queue:
            current_face_idx = queue.popleft()
            current_face = faces[current_face_idx]
            current_normal = face_normals[current_face_idx]
            if face_flipped[current_face_idx]:
                current_normal = -current_normal
            
            # Check all edges of current face
            for i in range(3):
                v1, v2 = int(current_face[i]), int(current_face[(i + 1) % 3])
                edge = tuple(sorted([v1, v2]))
                
                if edge in edge_to_faces:
                    # Find neighboring faces sharing this edge
                    for neighbor_face_idx, neighbor_edge_pos in edge_to_faces[edge]:
                        if neighbor_face_idx == current_face_idx:
                            continue
                        
                        if not processed[neighbor_face_idx]:
                            neighbor_face = faces[neighbor_face_idx]
                            
                            # Check edge direction in neighbor face
                            nv1 = int(neighbor_face[neighbor_edge_pos])
                            nv2 = int(neighbor_face[(neighbor_edge_pos + 1) % 3])
                            edge_reversed = (nv1 == v2 and nv2 == v1)
                            
                            # Get neighbor normal
                            neighbor_normal = face_normals[neighbor_face_idx]
                            
                            # For consistent orientation:
                            # - If edge is in same direction, normals should point in same direction
                            # - If edge is reversed, normals should point in opposite directions
                            # But we want normals to be consistent, so we check the dot product
                            # and flip if needed
                            
                            # Compute dot product of normals
                            dot_product = np.dot(current_normal, neighbor_normal)
                            
                            # If normals are pointing in opposite directions, flip the neighbor
                            if dot_product < -0.1:  # Use threshold to avoid numerical issues
                                # Flip the neighbor face (swap two vertices)
                                faces[neighbor_face_idx] = faces[neighbor_face_idx][[0, 2, 1]]
                                face_flipped[neighbor_face_idx] = True
                                face_normals[neighbor_face_idx] = -face_normals[neighbor_face_idx]
                            
                            processed[neighbor_face_idx] = True
                            queue.append(neighbor_face_idx)
    
    return faces


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
    normals = np.zeros((len(vertices), 3), dtype=np.float64)
    
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

