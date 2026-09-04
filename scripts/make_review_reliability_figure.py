#!/usr/bin/env python3
"""Create the conditional reliability figure used in the revised manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"


def method_comparison() -> None:
    methods = [
        "Physical LightGBM",
        "Physical XGBoost",
        "Chronos-2 fully tuned",
        "Chronos-2 LoRA, 7 days",
        "Chronos-2 zero-shot",
        "Non-physical LightGBM",
    ]
    mae = np.array([0.087875, 0.088054, 0.089242, 0.091112, 0.091402, 0.093141])
    crps = np.array([0.063014, 0.062805, 0.062576, 0.064069, 0.064216, 0.067220])
    order = np.argsort(mae)
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, gridspec_kw={"wspace": 0.10})
    axes[0].scatter(mae[order], positions, color="#1f5a94", s=30)
    axes[0].set_yticks(positions, np.array(methods)[order])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Daylight MAE (p.u.)")
    axes[0].set_title("Point accuracy")
    axes[1].scatter(crps[order], positions, color="#b44b3a", s=30)
    axes[1].set_xlabel(r"Daylight $\mathrm{CRPS}_{9q}$ (p.u.)")
    axes[1].set_title("Distributional accuracy")
    axes[1].tick_params(axis="y", labelleft=False)
    for ax in axes:
        ax.grid(axis="x", color="0.88", linewidth=0.6)
    fig.savefig(FIGURES / "nrel_complete_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    posthoc = json.loads((RESULTS / "review_posthoc_analysis.json").read_text())
    vintage = json.loads((RESULTS / "ecmwf_vintage_review.json").read_text())
    seasonal = posthoc["nrel"]["state_transfer_seasonal_calibration"]
    conditional = vintage["models"]["nonphysical_direct"]["conditional"]

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "figure.dpi": 160})
    method_comparison()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)

    states = ["AZ", "CO", "WA"]
    seasons = ["winter", "spring", "summer", "autumn"]
    for state in states:
        values = [seasonal[f"{state}_{season}"]["coverage_90"] for season in seasons]
        axes[0, 0].plot(seasons, values, marker="o", label=state)
    axes[0, 0].set_title("NREL state transfer by season")
    axes[0, 0].set_ylabel("Empirical 90% coverage")
    axes[0, 0].legend(frameon=False, ncol=3)

    panels = [
        ("lead_band", "ECMWF by lead time"),
        ("solar_elevation_regime", "ECMWF by solar elevation"),
        ("season", "ECMWF by season"),
        ("clear_sky_variability", "ECMWF by clear-sky variability"),
        ("production_regime", "ECMWF by production"),
    ]
    for ax, (key, title) in zip(axes.flat[1:], panels):
        groups = conditional[key]
        labels = list(groups)
        values = [groups[label]["coverage_90"] for label in labels]
        x = np.arange(len(labels))
        ax.bar(x, values, color="#3f78a8", width=0.68)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)

    for ax in axes.flat:
        ax.axhline(0.90, color="#b33b32", linestyle="--", linewidth=0.9)
        ax.set_ylim(0.70, 1.02)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical 90% coverage")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "conditional_reliability.pdf")
    fig.savefig(FIGURES / "conditional_reliability.png", dpi=300)
    print(FIGURES / "conditional_reliability.pdf")


if __name__ == "__main__":
    main()
