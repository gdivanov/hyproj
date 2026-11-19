"""
hyproj.modflow_engine
---------------------
Wraps USGS MODFLOW 6 via FloPy to solve 3D steady-state groundwater flow across
heterogeneous hydraulic conductivity (K) fields and deterministic equivalent baselines.
Extracts 3D hydraulic heads, volumetric seepage rates, and fast-path advective
breakthrough times.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import flopy

from hyproj.geology import GridSpec


@dataclass
class FlowBoundaryConfig:
    """Defines hydraulic head boundary conditions (Dirichlet CHD) along X."""
    head_west_m: float = 55.0       # Upgradient source head (e.g., beneath Tailings Facility)
    head_east_m: float = 50.0       # Downgradient compliance boundary head (e.g., creek/aquifer)
    porosity_sand: float = 0.25     # Effective porosity of sand channels (-)
    porosity_till: float = 0.40     # Effective porosity of glacial till matrix (-)
    vertical_anisotropy: float = 0.1 # Ratio Kz / Kx (sedimentary bedding reduces vertical K)


@dataclass
class ModflowResults:
    """Quantitative flow and transport outputs from a single MODFLOW 6 run."""
    converged: bool
    head_3d: np.ndarray                 # 3D hydraulic head field (nz, ny, nx) in meters
    qx_3d: np.ndarray                   # 3D X-component Darcy flux (m/s)
    total_seepage_m3_per_day: float     # Total volumetric discharge exiting East face (m^3/day)
    effective_k_m_per_s: float          # Back-calculated bulk Darcy K_eff across the block (m/s)
    max_linear_velocity_m_per_day: float # Peak pore-water velocity in fastest channel (m/day)
    advective_breakthrough_years: float # Estimated first-arrival time across domain (years)


def run_modflow6_steady(
    grid: GridSpec,
    k_field: np.ndarray,
    facies_mask: Optional[np.ndarray] = None,
    bc: FlowBoundaryConfig = FlowBoundaryConfig(),
    workspace: Path = Path("outputs/mf6_work"),
    sim_name: str = "hyproj_mf6",
    mf6_exe: str = "mf6"
) -> ModflowResults:
    """
    Builds, writes, executes, and post-processes a 3D MODFLOW 6 steady-state model.

    Parameters:
    -----------
    grid : GridSpec
        Spatial grid definition (nx, ny, nz, dx, dy, dz).
    k_field : np.ndarray (shape: nz, ny, nx)
        3D hydraulic conductivity field in m/s.
    facies_mask : np.ndarray, optional
        Binary facies array (1=sand, 0=till) used to assign spatially variable porosity.
        If None, uses uniform porosity_sand.
    bc : FlowBoundaryConfig
        Constant head boundary conditions and porosity settings.
    workspace : Path
        Directory where MODFLOW 6 input/output files are written.
    sim_name : str
        Base filename for the MODFLOW 6 simulation.
    mf6_exe : str
        Path or command name for the MODFLOW 6 executable.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # 1. Initialize MODFLOW 6 Simulation & Iterative Matrix Solver (IMS)
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        sim_ws=str(workspace),
        exe_name=mf6_exe,
        verbosity_level=0
    )

    # Steady-state stress period (1.0 second unit consistency with K in m/s)
    tdis = flopy.mf6.ModflowTdis(
        sim, nper=1, perioddata=[(1.0, 1, 1.0)], time_units="SECONDS"
    )

    # Conjugate Gradient / BiCGSTAB solver tuned for high K-contrast (5 orders of magnitude)
    ims = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=1e-5,
        outer_maximum=500,
        under_relaxation="DBD",
        inner_maximum=1000,
        inner_dvclose=1e-6,
        rcloserecord=[1e-6, "RELATIVE_RCLOSE"],
        linear_acceleration="BICGSTAB",
        scaling_method="DIAGONAL",
        reordering_method="MD",
        preconditioner_levels=7,
        preconditioner_drop_tolerance=1e-4,
    )

    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=sim_name[:16], save_flows=True
    )

    # 2. Discretization Package (DIS): Regular 3D structured grid
    top_elev = grid.nz * grid.dz
    botm_elevs = [top_elev - (k + 1) * grid.dz for k in range(grid.nz)]

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        length_units="METERS",
        nlay=grid.nz,
        nrow=grid.ny,
        ncol=grid.nx,
        delr=grid.dx,
        delc=grid.dy,
        top=top_elev,
        botm=botm_elevs
    )

    # 3. Initial Conditions (IC): Linear hydraulic gradient guess between West and East
    h_init = np.linspace(bc.head_west_m, bc.head_east_m, grid.nx)
    strt = np.broadcast_to(h_init, (grid.nz, grid.ny, grid.nx)).copy()
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # 4. Node Property Flow (NPF): Assign 3D heterogeneous K and enable specific discharge
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,  # Confined saturated flow (linear conductance, robust convergence)
        k=k_field,
        k33=k_field * bc.vertical_anisotropy,
        save_specific_discharge=True
    )

    # 5. Constant Head Boundaries (CHD): West face (ix=0) and East face (ix=nx-1)
    chd_spd = []
    for k in range(grid.nz):
        for i in range(grid.ny):
            chd_spd.append([(k, i, 0), bc.head_west_m])
            chd_spd.append([(k, i, grid.nx - 1), bc.head_east_m])

    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

    # 6. Output Control (OC): Save binary head (.hds) and cell-by-cell budget (.cbc)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{sim_name[:16]}.hds",
        budget_filerecord=f"{sim_name[:16]}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")]
    )

    # 7. Write files and execute MODFLOW 6 binary
    sim.write_simulation()
    success, _ = sim.run_simulation(silent=True)

    if not success:
        raise RuntimeError(f"MODFLOW 6 failed to converge in {workspace}")

    # 8. Post-process Heads and Cell-by-Cell Specific Discharge (Darcy Velocity)
    head_3d = gwf.output.head().get_data()
    bud = gwf.output.budget()

    # Extract specific discharge vector components (qx, qy, qz) in m/s
    spdis = bud.get_data(text="DATA-SPDIS")[0]
    qx_3d = spdis["qx"].reshape((grid.nz, grid.ny, grid.nx))

    # Total seepage exiting the East compliance face (cross-sectional area = dy * dz per cell)
    cell_area_yz = grid.dy * grid.dz
    # Use interior interface near East face (ix = nx - 2) for clean flux accounting
    total_q_m3_s = float(np.sum(qx_3d[:, :, -2]) * cell_area_yz)
    total_seepage_m3_per_day = total_q_m3_s * 86400.0

    # Back-calculate bulk equivalent hydraulic conductivity K_eff via Darcy's Law: Q = K_eff * A * (dh/dL)
    total_area_yz = (grid.ny * grid.dy) * (grid.nz * grid.dz)
    domain_length_x = (grid.nx - 1) * grid.dx
    dh = bc.head_west_m - bc.head_east_m
    effective_k_m_per_s = total_q_m3_s / (total_area_yz * (dh / domain_length_x))

    # Compute pore-water linear velocity v_x = q_x / n_e (m/s)
    if facies_mask is not None:
        porosity_3d = np.where(facies_mask == 1, bc.porosity_sand, bc.porosity_till)
    else:
        porosity_3d = np.full_like(k_field, bc.porosity_sand)

    vx_3d = qx_3d / porosity_3d  # m/s

    # Estimate preferential breakthrough time by computing slice-by-slice harmonic transit time
    # across the fastest connected streamtube (95th percentile velocity in each X-slice)
    slice_p95_vx = np.percentile(np.maximum(vx_3d, 1e-15), 99.5, axis=(0, 1))
    transit_seconds = float(np.sum(grid.dx / slice_p95_vx))
    advective_breakthrough_years = transit_seconds / (86400.0 * 365.25)
    max_linear_velocity_m_per_day = float(np.max(vx_3d) * 86400.0)

    return ModflowResults(
        converged=True,
        head_3d=head_3d,
        qx_3d=qx_3d,
        total_seepage_m3_per_day=total_seepage_m3_per_day,
        effective_k_m_per_s=effective_k_m_per_s,
        max_linear_velocity_m_per_day=max_linear_velocity_m_per_day,
        advective_breakthrough_years=advective_breakthrough_years,
    )


def create_deterministic_baseline_fields(
    grid: GridSpec,
    sand_fraction: float,
    k_sand: float = 1e-4,
    k_till: float = 1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs homogeneous 'melting pot' K fields for the benchmark comparison:
      1. Geometric Mean K_G = K_sand^p * K_till^(1-p)
      2. Arithmetic Mean K_A = p * K_sand + (1-p) * K_till
    """
    shape = (grid.nz, grid.ny, grid.nx)
    k_geom = (k_sand ** sand_fraction) * (k_till ** (1.0 - sand_fraction))
    k_arith = sand_fraction * k_sand + (1.0 - sand_fraction) * k_till

    return np.full(shape, k_geom, dtype=np.float64), np.full(shape, k_arith, dtype=np.float64)