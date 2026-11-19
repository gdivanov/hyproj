"""
hyproj.pipeline
---------------
Scalable Monte Carlo data pipeline:
  1. Parallel 3D Percolation Screening across depositional scenarios & sand fractions.
  2. Intelligent Triage: Selects P10, P50, and P90 realizations from the target ensemble
     and executes full 3D PDE solves in MODFLOW 6 alongside deterministic baselines.
  3. Persists structured Parquet summary tables, synthetic well CSVs, and 3D NetCDF grids.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
import xarray as xr
import flopy

from hyproj.config import PipelineConfig
from hyproj.geology import (
    extract_synthetic_borehole,
    generate_3d_random_field,
    map_field_to_hydrogeology,
)
from hyproj.modflow_engine import (
    create_deterministic_baseline_fields,
    run_modflow6_steady,
)
from hyproj.percolation import analyze_percolation


def ensure_modflow6_binary(bin_dir: Path = Path("bin")) -> str:
    """Checks for `mf6` on PATH or automatically downloads official USGS binary."""
    if shutil.which("mf6"):
        return "mf6"
    bin_dir.mkdir(parents=True, exist_ok=True)
    local_exe = bin_dir / "mf6"
    if not local_exe.exists() and not (bin_dir / "mf6.exe").exists():
        print("[Pipeline] Downloading USGS MODFLOW 6 binary via FloPy...")
        flopy.utils.get_modflow(str(bin_dir), subset="mf6")
    return str(bin_dir / "mf6.exe") if (bin_dir / "mf6.exe").exists() else str(local_exe)


def _evaluate_single_realization(
    cfg: PipelineConfig,
    scenario_name: str,
    corr_lengths: Tuple[float, float, float],
    sand_fraction: float,
    realization_idx: int,
    seed: int
) -> Dict:
    """Worker function executed in parallel across CPU cores."""
    grf = generate_3d_random_field(cfg.grid, corr_lengths=corr_lengths, seed=seed)
    facies_mask, _, _ = map_field_to_hydrogeology(
        grf, sand_fraction=sand_fraction, k_sand=cfg.k_sand, k_till=cfg.k_till
    )
    metrics, _ = analyze_percolation(
        facies_mask, dx=cfg.grid.dx, dy=cfg.grid.dy, dz=cfg.grid.dz
    )
    return {
        "scenario": scenario_name,
        "target_sand_fraction": sand_fraction,
        "realization_id": realization_idx,
        "seed": seed,
        "actual_sand_fraction": metrics.sand_fraction,
        "num_clusters": metrics.num_clusters,
        "largest_cluster_fraction": metrics.largest_cluster_fraction,
        "percolates_x": metrics.percolates_x,
        "percolates_y": metrics.percolates_y,
        "percolates_z": metrics.percolates_z,
        "spanning_volume_fraction_x": metrics.spanning_volume_fraction_x,
        "tortuosity_x": metrics.tortuosity_x if metrics.tortuosity_x is not None else np.nan,
        # Composite fast-screening connectivity index (higher = faster expected breakthrough)
        "connectivity_index": (
            metrics.spanning_volume_fraction_x / metrics.tortuosity_x
            if metrics.percolates_x and metrics.tortuosity_x
            else 0.0
        ),
    }


def run_percolation_sweep(cfg: PipelineConfig) -> pd.DataFrame:
    """Executes parallel Monte Carlo percolation sweep and saves Parquet dataset."""
    tasks = []
    base_seed = cfg.random_seed
    counter = 0

    for scen_name, corr_lens in cfg.scenarios.items():
        for p in cfg.sand_fractions:
            for r in range(cfg.realizations_per_step):
                counter += 1
                tasks.append((scen_name, tuple(corr_lens), p, r, base_seed + counter))

    print(f"[Pipeline] Screening {len(tasks)} 3D realizations via parallel percolation...")
    records = joblib.Parallel(n_jobs=cfg.n_jobs)(
        joblib.delayed(_evaluate_single_realization)(cfg, s, c, p, r, sd)
        for (s, c, p, r, sd) in tasks
    )

    df_perc = pd.DataFrame(records)
    raw_dir = cfg.data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "percolation_ensemble_metrics.parquet"
    df_perc.to_parquet(parquet_path, index=False)
    print(f"[Pipeline] Saved percolation metrics -> {parquet_path}")
    return df_perc


def run_modflow_triage_benchmark(
    cfg: PipelineConfig,
    df_perc: pd.DataFrame
) -> Tuple[pd.DataFrame, xr.Dataset]:
    """
    Runs MODFLOW 6 on:
      1. Deterministic Geometric Mean (K_G) and Arithmetic Mean (K_A) baselines.
      2. Quantile realizations (P10, P50, P90 connectivity) selected by percolation screening.
    Exports benchmark summary (Parquet) and 3D fields of the P90 worst-case run (NetCDF).
    """
    mf6_exe = ensure_modflow6_binary()
    primary_scenario = list(cfg.scenarios.keys())[0]
    corr_lens = tuple(cfg.scenarios[primary_scenario])
    p_target = cfg.target_sand_fraction

    subset = df_perc[
        (df_perc["scenario"] == primary_scenario)
        & (np.isclose(df_perc["target_sand_fraction"], p_target))
    ].sort_values("connectivity_index")

    benchmark_rows = []
    p90_dataset = None

    # A. Deterministic MODFLOW Baselines ("Melting Pot" Averages)
    k_geom_3d, k_arith_3d = create_deterministic_baseline_fields(
        cfg.grid, sand_fraction=p_target, k_sand=cfg.k_sand, k_till=cfg.k_till
    )
    for label_name, k_arr in [("Deterministic_Geometric_KG", k_geom_3d),
                              ("Deterministic_Arithmetic_KA", k_arith_3d)]:
        res = run_modflow6_steady(
            cfg.grid, k_arr, facies_mask=None, bc=cfg.flow_bc,
            workspace=cfg.output_dir / "mf6_runs" / label_name,
            sim_name=label_name.lower(), mf6_exe=mf6_exe
        )
        benchmark_rows.append({
            "model_type": label_name,
            "seed": -1,
            "percolates_x": True,
            "connectivity_index": np.nan,
            "total_seepage_m3_day": res.total_seepage_m3_per_day,
            "effective_k_m_s": res.effective_k_m_per_s,
            "max_velocity_m_day": res.max_linear_velocity_m_per_day,
            "breakthrough_years": res.advective_breakthrough_years,
        })

    # B. Percolation-Screened Quantile Realizations (P10, P50, P90)
    for q in cfg.quantiles_to_solve:
        idx = min(int(q * len(subset)), len(subset) - 1)
        row = subset.iloc[idx]
        seed = int(row["seed"])
        q_label = f"Percolation_Screened_P{int(q*100)}"
        print(f"  -> Solving MODFLOW 6: {q_label}...", flush=True)

        grf = generate_3d_random_field(cfg.grid, corr_lengths=corr_lens, seed=seed)
        facies, gr_field, k_field = map_field_to_hydrogeology(
            grf, sand_fraction=p_target, k_sand=cfg.k_sand, k_till=cfg.k_till
        )
        _, spanning_mask = analyze_percolation(
            facies, dx=cfg.grid.dx, dy=cfg.grid.dy, dz=cfg.grid.dz
        )

        res = run_modflow6_steady(
            cfg.grid, k_field, facies_mask=facies, bc=cfg.flow_bc,
            workspace=cfg.output_dir / "mf6_runs" / q_label,
            sim_name=q_label.lower(), mf6_exe=mf6_exe
        )

        benchmark_rows.append({
            "model_type": q_label,
            "seed": seed,
            "percolates_x": bool(row["percolates_x"]),
            "connectivity_index": float(row["connectivity_index"]),
            "total_seepage_m3_day": res.total_seepage_m3_per_day,
            "effective_k_m_s": res.effective_k_m_per_s,
            "max_velocity_m_day": res.max_linear_velocity_m_per_day,
            "breakthrough_years": res.advective_breakthrough_years,
        })

        # Save the P90 high-risk realization + synthetic borehole logs for 3D visualization
        if np.isclose(q, max(cfg.quantiles_to_solve)):
            well_dfs = [
                extract_synthetic_borehole(
                    cfg.grid, gr_field, k_field, facies,
                    ix=w["ix"], iy=w["iy"], well_id=w["well_id"]
                )
                for w in cfg.synthetic_wells
            ]
            raw_well_dir = cfg.data_dir / "raw"
            raw_well_dir.mkdir(parents=True, exist_ok=True)
            pd.concat(well_dfs, ignore_index=True).to_csv(
                raw_well_dir / "synthetic_borehole_logs.csv", index=False
            )

            p90_dataset = xr.Dataset(
                data_vars={
                    "facies": (("z", "y", "x"), facies),
                    "gamma_ray_api": (("z", "y", "x"), gr_field),
                    "hydraulic_conductivity_m_s": (("z", "y", "x"), k_field),
                    "spanning_backbone_x": (("z", "y", "x"), spanning_mask.astype(np.int8)),
                    "hydraulic_head_m": (("z", "y", "x"), res.head_3d),
                    "darcy_flux_qx_m_s": (("z", "y", "x"), res.qx_3d),
                },
                coords={
                    "z": np.arange(cfg.grid.nz) * cfg.grid.dz,
                    "y": np.arange(cfg.grid.ny) * cfg.grid.dy,
                    "x": np.arange(cfg.grid.nx) * cfg.grid.dx,
                },
                attrs={"scenario": primary_scenario, "seed": seed, "sand_fraction": p_target}
            )
            nc_path = cfg.data_dir / "processed" / "p90_high_risk_3d_model.nc"
            p90_dataset.to_netcdf(nc_path)
            print(f"[Pipeline] Saved 3D NetCDF volume -> {nc_path}")

    df_bench = pd.DataFrame(benchmark_rows)
    bench_path = cfg.data_dir / "processed" / "modflow_benchmark_comparison.parquet"
    df_bench.to_parquet(bench_path, index=False)
    print(f"[Pipeline] Saved MODFLOW benchmark -> {bench_path}")
    return df_bench, p90_dataset