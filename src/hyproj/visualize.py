"""
hyproj.visualize
----------------
Generates clear, publication-ready 2D/3D engineering figures so you can visually
verify stability, borehole logs, connected sand pathways, hydraulic head contours,
and probabilistic cut-off curves directly on your desktop.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def plot_probabilistic_cutoff_curves(df_perc: pd.DataFrame, fig_dir: Path) -> None:
    """Plots Percolation Probability P_span vs. Sand Fraction p across media types."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = (
        df_perc.groupby(["scenario", "target_sand_fraction"], as_index=False)
        .agg(
            prob_percolation_x=("percolates_x", "mean"),
            mean_tortuosity=("tortuosity_x", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for scen, grp in summary.groupby("scenario"):
        ax.plot(
            grp["target_sand_fraction"] * 100.0,
            grp["prob_percolation_x"] * 100.0,
            marker="o", linewidth=2.2, label=scen.replace("_", " ").title()
        )

    ax.axhline(50.0, color="gray", linestyle="--", linewidth=1, label="50% Critical Threshold (pc)")
    ax.set_title("Probabilistic Seepage Cut-Off Curves Across Depositional Media", fontsize=13, fontweight="bold")
    ax.set_xlabel("Bulk Sand Facies Proportion (%)", fontsize=11)
    ax.set_ylabel("West-to-East Percolation Probability (%)", fontsize=11)
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "01_probabilistic_cutoff_curves.png")
    plt.close(fig)


def plot_cross_section_and_head(ds: xr.Dataset, fig_dir: Path) -> None:
    """Plots a 2D vertical cross-section (X-Z slice at mid-Y) showing sand channels,
    spanning percolation backbone, hydraulic head contours, and Darcy velocity."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    mid_y = ds.sizes["y"] // 2
    sub = ds.isel(y=mid_y)

    X, Z = np.meshgrid(sub["x"].values, sub["z"].values)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, dpi=150)

    # Panel 1: Facies & Percolation Backbone + Head Contours
    im1 = axes[0].pcolormesh(X, Z, sub["facies"].values, cmap="YlOrBr", shading="auto", alpha=0.85)
    cs = axes[0].contour(X, Z, sub["hydraulic_head_m"].values, levels=12, colors="navy", linewidths=1.2)
    axes[0].clabel(cs, inline=True, fontsize=8, fmt="%.1f m")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)", fontsize=11)
    axes[0].set_title("P90 High-Risk Cross-Section: Sand Channels (Gold) & Hydraulic Head Contours", fontweight="bold")

    # Panel 2: Log10 Darcy Flux (qx) showing preferential flow focusing
    qx_abs = np.maximum(np.abs(sub["darcy_flux_qx_m_s"].values), 1e-12)
    im2 = axes[1].pcolormesh(X, Z, np.log10(qx_abs), cmap="viridis", shading="auto")
    cbar = fig.colorbar(im2, ax=axes[1], orientation="vertical", pad=0.02)
    cbar.set_label("log10(|qx|) [m/s]")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Distance Downgradient X (m)", fontsize=11)
    axes[1].set_ylabel("Depth (m)", fontsize=11)
    axes[1].set_title("MODFLOW 6 Specific Discharge: Preferential Channeling vs. Stagnant Till", fontweight="bold")

    plt.tight_layout()
    plt.savefig(fig_dir / "02_p90_cross_section_flow.png")
    plt.close(fig)


def plot_synthetic_boreholes(well_csv: Path, fig_dir: Path) -> None:
    """Plots synthetic Gamma Ray (API) and Log10(K) wireline logs for visual QC."""
    if not well_csv.exists():
        return
    df = pd.read_csv(well_csv)
    wells = df["well_id"].unique()
    fig, axes = plt.subplots(1, len(wells), figsize=(4 * len(wells), 6), sharey=True, dpi=150)
    if len(wells) == 1:
        axes = [axes]

    for ax, wid in zip(axes, wells):
        wdf = df[df["well_id"] == wid]
        ax.plot(wdf["gamma_ray_api"], wdf["depth_m"], color="darkgreen", linewidth=1.8)
        ax.fill_betweenx(wdf["depth_m"], 15, wdf["gamma_ray_api"], color="khaki", alpha=0.5)
        ax.axvline(65, color="red", linestyle="--", linewidth=0.9, label="Sand/Till Cut-off")
        ax.set_title(f"{wid}\n(X={wdf['x_m'].iloc[0]:.0f}m)", fontweight="bold")
        ax.set_xlabel("Gamma Ray (API)")
        ax.set_xlim(15, 130)
        ax.grid(True, alpha=0.3)

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    axes[0].legend(loc="lower right", fontsize=8)
    plt.suptitle("Synthetic Geophysical Borehole Logs (Extracted from 3D Model)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "03_synthetic_borehole_logs.png")
    plt.close(fig)