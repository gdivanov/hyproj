"""
hyproj.config
-------------
Loads and validates external YAML simulation configurations into typed dataclasses.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml

from hyproj.geology import GridSpec
from hyproj.modflow_engine import FlowBoundaryConfig


@dataclass
class PipelineConfig:
    project_name: str
    output_dir: Path
    data_dir: Path
    n_jobs: int
    random_seed: int
    grid: GridSpec
    scenarios: Dict[str, List[float]]
    k_sand: float
    k_till: float
    synthetic_wells: List[Dict[str, Any]]
    sand_fractions: List[float]
    realizations_per_step: int
    mf_enabled: bool
    target_sand_fraction: float
    quantiles_to_solve: List[float]
    flow_bc: FlowBoundaryConfig


def load_config(yaml_path: str | Path = "config/simulation_config.yaml") -> PipelineConfig:
    """Parses YAML configuration file into structured PipelineConfig object."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    g = raw["grid"]
    grid = GridSpec(
        nx=int(g["nx"]), ny=int(g["ny"]), nz=int(g["nz"]),
        dx=float(g["dx"]), dy=float(g["dy"]), dz=float(g["dz"]),
    )

    geo = raw["geology"]
    scenarios = {
        name: [float(v) for v in cfg["corr_lengths"]]
        for name, cfg in geo["scenarios"].items()
    }

    mf = raw["modflow_benchmark"]
    flow_bc = FlowBoundaryConfig(
        head_west_m=float(mf["head_west_m"]),
        head_east_m=float(mf["head_east_m"]),
        porosity_sand=float(mf["porosity_sand"]),
        porosity_till=float(mf["porosity_till"]),
        vertical_anisotropy=float(mf["vertical_anisotropy"]),
    )

    proj = raw["project"]
    sweep = raw["percolation_sweep"]

    return PipelineConfig(
        project_name=proj["name"],
        output_dir=Path(proj["output_dir"]),
        data_dir=Path(proj["data_dir"]),
        n_jobs=int(proj["n_jobs"]),
        random_seed=int(proj["random_seed"]),
        grid=grid,
        scenarios=scenarios,
        k_sand=float(geo["k_sand_m_s"]),
        k_till=float(geo["k_till_m_s"]),
        synthetic_wells=geo["synthetic_wells"],
        sand_fractions=[float(p) for p in sweep["sand_fractions"]],
        realizations_per_step=int(sweep["realizations_per_step"]),
        mf_enabled=bool(mf["enabled"]),
        target_sand_fraction=float(mf["target_sand_fraction"]),
        quantiles_to_solve=[float(q) for q in mf["quantiles_to_solve"]],
        flow_bc=flow_bc,
    )