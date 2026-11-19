"""
hyproj.cli
----------
Single-command entry point (`hyproj-run`) to execute the end-to-end pipeline.
"""

import argparse
from hyproj.config import load_config
from hyproj.pipeline import run_modflow_triage_benchmark, run_percolation_sweep
from hyproj.visualize import (
    plot_cross_section_and_head,
    plot_probabilistic_cutoff_curves,
    plot_synthetic_boreholes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyproj 3D Percolation + MODFLOW 6 Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/simulation_config.yaml",
        help="Path to YAML simulation configuration file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    fig_dir = cfg.output_dir / "figures"

    # Step 1: Parallel Percolation Screening
    df_perc = run_percolation_sweep(cfg)
    plot_probabilistic_cutoff_curves(df_perc, fig_dir)

    # Step 2: MODFLOW 6 Head-to-Head Benchmark & 3D NetCDF Export
    if cfg.mf_enabled:
        df_bench, p90_ds = run_modflow_triage_benchmark(cfg, df_perc)
        print("\n=== MODFLOW 6 HEAD-TO-HEAD BENCHMARK RESULTS ===")
        print(df_bench.to_string(index=False))

        if p90_ds is not None:
            plot_cross_section_and_head(p90_ds, fig_dir)
            plot_synthetic_boreholes(cfg.data_dir / "raw" / "synthetic_borehole_logs.csv", fig_dir)

    print(f"\n[Done] All outputs and figures saved to: {cfg.output_dir.resolve()}")


if __name__ == "__main__":
    main()