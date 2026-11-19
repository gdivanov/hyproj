"""
hyproj.percolation
------------------
Analyzes 3D connectivity of high-permeability sand facies using 3D site percolation
theory (6-connectivity). Identifies spanning preferential flow pathways, computes
percolation order parameters, and estimates shortest-path tortuosity to screen
realizations prior to full numerical PDE simulation in MODFLOW 6.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.ndimage import label


@dataclass
class PercolationMetrics:
    """Summary physics metrics for a single 3D geological realization."""
    sand_fraction: float                 # Bulk sand proportion (p)
    num_clusters: int                    # Total discrete sand geobodies
    largest_cluster_fraction: float      # Largest cluster volume / total domain volume
    percolates_x: bool                   # True if a continuous sand path connects x=0 to x=nx-1
    percolates_y: bool                   # True if a continuous sand path connects y=0 to y=ny-1
    percolates_z: bool                   # True if a continuous sand path connects z=0 to z=nz-1
    spanning_volume_fraction_x: float    # Order parameter P_inf along X (primary flow axis)
    tortuosity_x: Optional[float]        # Ratio of shortest connected path length to domain length Lx
    spanning_cluster_ids_x: List[int]    # Cluster labels bridging West (x=0) to East (x=nx-1)


def get_3d_6_connectivity_kernel() -> np.ndarray:
    """
    Returns a 3x3x3 structuring element representing 6-connectivity (shared voxel faces).
    In finite-difference groundwater models (like MODFLOW), flux only crosses shared
    cell faces—never diagonal edges or corners.
    """
    kernel = np.zeros((3, 3, 3), dtype=np.int8)
    kernel[1, 1, 1] = 1  # Center cell
    kernel[0, 1, 1] = 1  # Top (-z)
    kernel[2, 1, 1] = 1  # Bottom (+z)
    kernel[1, 0, 1] = 1  # North (-y)
    kernel[1, 2, 1] = 1  # South (+y)
    kernel[1, 1, 0] = 1  # West (-x)
    kernel[1, 1, 2] = 1  # East (+x)
    return kernel


def label_sand_clusters(facies_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Labels all connected 3D sand clusters using 6-connectivity.
    
    Parameters:
    -----------
    facies_mask : np.ndarray (shape: nz, ny, nx)
        Binary array where 1 = permeable sand, 0 = low-permeability till.
        
    Returns:
    --------
    labeled_grid : np.ndarray
        Integer array where 0 = till and 1..N = unique sand cluster IDs.
    num_clusters : int
        Total count of isolated sand geobodies.
    """
    structure = get_3d_6_connectivity_kernel()
    labeled_grid, num_clusters = label(facies_mask, structure=structure)
    return labeled_grid, int(num_clusters)


def find_spanning_clusters(
    labeled_grid: np.ndarray,
    axis: int
) -> List[int]:
    """
    Identifies cluster IDs that simultaneously touch both opposite faces along `axis`.
    
    Axis mapping for (nz, ny, nx) array:
      - axis = 0 -> Vertical (Top z=0 to Bottom z=nz-1)
      - axis = 1 -> Transverse (North y=0 to South y=ny-1)
      - axis = 2 -> Longitudinal / Regional Gradient (West x=0 to East x=nx-1)
    """
    face_start = np.take(labeled_grid, indices=0, axis=axis)
    face_end = np.take(labeled_grid, indices=-1, axis=axis)
    
    # Exclude label 0 (which represents non-conductive glacial till)
    ids_start = set(np.unique(face_start)) - {0}
    ids_end = set(np.unique(face_end)) - {0}
    
    spanning_ids = sorted(list(ids_start.intersection(ids_end)))
    return spanning_ids


def compute_shortest_path_tortuosity_x(
    labeled_grid: np.ndarray,
    spanning_ids: List[int],
    dx: float,
    dy: float,
    dz: float
) -> Optional[float]:
    """
    Uses multi-source Breadth-First Search (BFS) weighted by physical cell dimensions
    to approximate the geometric tortuosity (tau = L_geodesic / L_euclidean) across X.
    
    Returns None if no spanning cluster exists along X.
    """
    if not spanning_ids:
        return None
    
    nz, ny, nx = labeled_grid.shape
    straight_line_length = (nx - 1) * dx
    if straight_line_length <= 0:
        return 1.0

    # Restrict search strictly to voxels belonging to spanning clusters
    spanning_mask = np.isin(labeled_grid, spanning_ids)
    
    # Distance array initialized to infinity
    dist = np.full((nz, ny, nx), np.inf, dtype=np.float64)
    queue = deque()
    
    # Seed BFS from all spanning sand cells on the West inlet face (x = 0)
    for z in range(nz):
        for y in range(ny):
            if spanning_mask[z, y, 0]:
                dist[z, y, 0] = 0.0
                queue.append((z, y, 0))
                
    # 6-connected neighbor offsets paired with physical step distances
    neighbors = [
        (-1, 0, 0, dz), (1, 0, 0, dz),
        (0, -1, 0, dy), (0, 1, 0, dy),
        (0, 0, -1, dx), (0, 0, 1, dx),
    ]
    
    min_exit_dist = np.inf
    
    while queue:
        z, y, x = queue.popleft()
        curr_d = dist[z, y, x]
        
        # Early pruning if current path already exceeds best found exit distance
        if curr_d >= min_exit_dist:
            continue
            
        # Check if we reached the East compliance boundary (x = nx - 1)
        if x == nx - 1:
            if curr_d < min_exit_dist:
                min_exit_dist = curr_d
            continue
            
        for dz_i, dy_i, dx_i, step_cost in neighbors:
            nz_i, ny_i, nx_i = z + dz_i, y + dy_i, x + dx_i
            if 0 <= nz_i < nz and 0 <= ny_i < ny and 0 <= nx_i < nx:
                if spanning_mask[nz_i, ny_i, nx_i]:
                    new_d = curr_d + step_cost
                    if new_d < dist[nz_i, ny_i, nx_i]:
                        dist[nz_i, ny_i, nx_i] = new_d
                        queue.append((nz_i, ny_i, nx_i))
                        
    if np.isinf(min_exit_dist):
        return None
        
    return float(min_exit_dist / straight_line_length)


def analyze_percolation(
    facies_mask: np.ndarray,
    dx: float = 10.0,
    dy: float = 10.0,
    dz: float = 2.0
) -> Tuple[PercolationMetrics, np.ndarray]:
    """
    Runs complete 3D percolation diagnostics on a binary sand/till realization.
    
    Returns:
    --------
    metrics : PercolationMetrics
        Quantitative connectivity and tortuosity metrics.
    spanning_mask_x : np.ndarray (bool, shape: nz, ny, nx)
        3D boolean mask highlighting ONLY the connected sand backbone that bridges
        West-to-East (useful for 3D PyVista visualization and backbone extraction).
    """
    total_voxels = facies_mask.size
    sand_fraction = float(np.mean(facies_mask))
    
    labeled_grid, num_clusters = label_sand_clusters(facies_mask)
    
    # Compute cluster sizes (ignoring index 0 = glacial till)
    if num_clusters > 0:
        counts = np.bincount(labeled_grid.ravel())[1:]
        largest_cluster_fraction = float(np.max(counts) / total_voxels)
    else:
        largest_cluster_fraction = 0.0
        
    span_z = find_spanning_clusters(labeled_grid, axis=0)
    span_y = find_spanning_clusters(labeled_grid, axis=1)
    span_x = find_spanning_clusters(labeled_grid, axis=2)
    
    spanning_mask_x = np.isin(labeled_grid, span_x) if span_x else np.zeros_like(facies_mask, dtype=bool)
    spanning_vol_frac_x = float(np.mean(spanning_mask_x))
    
    tortuosity_x = compute_shortest_path_tortuosity_x(
        labeled_grid, span_x, dx=dx, dy=dy, dz=dz
    )
    
    metrics = PercolationMetrics(
        sand_fraction=sand_fraction,
        num_clusters=num_clusters,
        largest_cluster_fraction=largest_cluster_fraction,
        percolates_x=len(span_x) > 0,
        percolates_y=len(span_y) > 0,
        percolates_z=len(span_z) > 0,
        spanning_volume_fraction_x=spanning_vol_frac_x,
        tortuosity_x=tortuosity_x,
        spanning_cluster_ids_x=span_x,
    )
    
    return metrics, spanning_mask_x