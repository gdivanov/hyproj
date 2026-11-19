"""
hyproj.geology
--------------
Generates synthetic geophysical borehole logs (Gamma Ray) and 3D anisotropic 
hydraulic conductivity (K) fields using 3D Fast Fourier Transform (FFT) spectral synthesis.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GridSpec:
    """Defines the 3D spatial domain of the hydrogeological model."""
    nx: int = 50          # Number of cells in X (e.g., 500m total at 10m resolution)
    ny: int = 50          # Number of cells in Y
    nz: int = 25          # Number of vertical layers (e.g., 50m deep at 2m resolution)
    dx: float = 10.0      # Cell width X (meters)
    dy: float = 10.0      # Cell width Y (meters)
    dz: float = 2.0       # Layer thickness Z (meters)


def generate_3d_random_field(
    grid: GridSpec,
    corr_lengths: Tuple[float, float, float] = (80.0, 80.0, 6.0),
    seed: int = 42
) -> np.ndarray:
    """
    Generates a standard normal 3D Gaussian Random Field (mean=0, std=1) with 
    anisotropic spatial correlation using Fourier filtering.
    
    Parameters:
    -----------
    grid : GridSpec
        Spatial dimensions of the 3D model.
    corr_lengths : tuple of (lx, ly, lz)
        Correlation lengths in meters. Larger lx, ly vs lz creates horizontal 
        sedimentary layering (lenses/channels).
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    
    # 1. Start with uncorrelated 3D Gaussian white noise
    white_noise = rng.standard_normal((grid.nz, grid.ny, grid.nx))
    
    # 2. Compute wavenumbers (spatial frequencies) in each direction
    kz = 2 * np.pi * np.fft.fftfreq(grid.nz, d=grid.dz)
    ky = 2 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)
    kx = 2 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing='ij')
    
    # 3. Construct anisotropic Gaussian power spectrum filter in Fourier space
    lx, ly, lz = corr_lengths
    power_spectrum = np.exp(-0.25 * ((KX * lx)**2 + (KY * ly)**2 + (KZ * lz)**2))
    
    # 4. Convolve white noise with filter via FFT and normalize to N(0, 1)
    fft_field = np.fft.fftn(white_noise) * np.sqrt(power_spectrum)
    correlated_field = np.real(np.fft.ifftn(fft_field))
    
    # Standardize to exact zero mean and unit variance
    correlated_field = (correlated_field - np.mean(correlated_field)) / np.std(correlated_field)
    return correlated_field


def map_field_to_hydrogeology(
    grf: np.ndarray,
    sand_fraction: float = 0.30,
    k_sand: float = 1e-4,
    k_till: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms a normalized 3D Gaussian field into:
      1. Binary Facies Mask (1 = High-K Sand Channel, 0 = Low-K Glacial Till)
      2. Synthetic Gamma Ray (GR) Log values (API units)
      3. Hydraulic Conductivity K field (m/s)
    """
    # Determine threshold quantile corresponding to desired sand proportion
    # E.g., if sand_fraction = 0.30, top 30% of GRF values become sand
    threshold = np.quantile(grf, 1.0 - sand_fraction)
    facies_mask = (grf >= threshold).astype(np.int8)
    
    # Synthesize Gamma Ray (API): Clean sand ~ 30 API, Clay Till ~ 105 API + sensor noise
    gr_field = np.where(facies_mask == 1, 30.0, 105.0) - (grf * 8.0)
    gr_field = np.clip(gr_field, 15.0, 150.0)
    
    # Map Hydraulic Conductivity (K) with subtle intra-facies log-normal variability
    log_k_sand = np.log10(k_sand) + 0.25 * (grf - threshold)
    log_k_till = np.log10(k_till) + 0.15 * grf
    
    k_field = np.where(facies_mask == 1, 10**log_k_sand, 10**log_k_till)
    
    return facies_mask, gr_field, k_field


def extract_synthetic_borehole(
    grid: GridSpec,
    gr_field: np.ndarray,
    k_field: np.ndarray,
    facies_mask: np.ndarray,
    ix: int,
    iy: int,
    well_id: str = "BH-01"
) -> pd.DataFrame:
    """
    Extracts a vertical 1D borehole log at grid index (ix, iy), mimicking 
    field site investigation data (e.g., for BGC Cambio Earth ingestion).
    """
    depths = np.arange(grid.nz) * grid.dz + (grid.dz / 2.0)
    
    df = pd.DataFrame({
        "well_id": well_id,
        "x_m": ix * grid.dx,
        "y_m": iy * grid.dy,
        "depth_m": depths,
        "gamma_ray_api": gr_field[:, iy, ix],
        "facies_code": facies_mask[:, iy, ix],
        "k_m_per_s": k_field[:, iy, ix]
    })
    return df