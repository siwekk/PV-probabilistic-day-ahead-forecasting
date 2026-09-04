#!/usr/bin/env python3
"""Create figures for the final reviewer-requested analyses."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"


def calibration_figure() -> None:
    data = json.loads((ROOT / "results/nrel_state_holdout.json").read_text())
    states = ["AZ", "CO", "WA"]
    methods = ["global", "season", "solar_elevation", "season_solar", "rolling_28d"]
    labels = ["Global", "Season", "Solar elevation", "Season + elevation", "Rolling 28 d"]
    metrics = [("coverage_90", "Coverage"), ("width_90", "Width (p.u.)"), ("interval_score_90", "Interval score")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    x = np.arange(len(states))
    width = 0.15
    for ax, (metric, title) in zip(axes, metrics):
        for index, (method, label) in enumerate(zip(methods, labels)):
            values = [data["rotations"][state]["models"]["physics_residual_trajectory"]["conditional_calibration"][method]["overall"]["all"][metric] for state in states]
            ax.bar(x + (index - 2) * width, values, width=width, label=label)
        ax.set_xticks(x, states)
        ax.set_title(title)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))
    fig.savefig(FIGURES / "conditional_calibration_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "conditional_calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def pvod_figure() -> None:
    data = json.loads((ROOT / "results/final_review_analysis.json").read_text())["pvod_station_meta_analysis"]
    stations = data["stations"]
    names = [row["station"] for row in stations]
    effects = np.array([row["physical_minus_nonphysical_mae"] for row in stations])
    errors = 1.96 * np.array([row["effect_block_se"] for row in stations])
    order = np.argsort(effects)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    y = np.arange(len(names))
    ax.errorbar(effects[order], y, xerr=errors[order], fmt="o", color="#1f5a94", capsize=2.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, np.array(names)[order])
    ax.set_xlabel("Physical minus non-physical MAE (p.u.)")
    ax.set_title("PVOD station effects")
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    fig.savefig(FIGURES / "pvod_station_meta_analysis.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "pvod_station_meta_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def timeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.0), constrained_layout=True)
    ax.set_xlim(-1, 37)
    ax.set_ylim(-0.9, 1.1)
    ax.arrow(0, 0, 35, 0, width=0.02, head_width=0.12, head_length=0.8, color="#333333", length_includes_head=True)
    ax.scatter([0, 12, 35], [0, 0, 0], color=["#b44b3a", "#1f5a94", "#1f5a94"], zorder=3)
    ax.text(0, 0.28, "Assumed issue\n12 UTC, day -1", ha="center")
    ax.text(12, 0.28, "Delivery starts\n00 UTC, lead 12 h", ha="center")
    ax.text(35, 0.28, "Delivery ends\n23 UTC, lead 35 h", ha="center")
    ax.plot([12, 35], [-0.32, -0.32], color="#2f7d4a", linewidth=5, solid_capstyle="butt")
    ax.text(23.5, -0.62, "Reconstructed UTC delivery window", ha="center", color="#2f7d4a")
    ax.text(18.5, 0.82, "Lead = 12 + UTC delivery hour", ha="center", fontweight="bold")
    ax.axis("off")
    fig.savefig(FIGURES / "ecmwf_schedule_timeline.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "ecmwf_schedule_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
    calibration_figure()
    pvod_figure()
    timeline_figure()


if __name__ == "__main__":
    main()
