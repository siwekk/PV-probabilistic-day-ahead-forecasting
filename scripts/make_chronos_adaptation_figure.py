#!/usr/bin/env python3
"""Plot the Chronos-2 data-budget experiment from frozen numerical results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper" / "figures"


def main() -> None:
    analysis = json.loads((ROOT / "results" / "extended_uncertainty_analysis.json").read_text())["chronos2"]
    budgets = np.array([1, 3, 7, 14, 30])
    mae = np.array([analysis["budget_ensembles"][str(day)]["metrics"]["mae"] for day in budgets])
    crps = np.array([analysis["budget_ensembles"][str(day)]["metrics"]["crps_9q"] for day in budgets])
    mae_sd = np.array([np.std([item["mae"] for item in analysis["seed_metrics"][str(day)]], ddof=1) for day in budgets])
    crps_sd = np.array([np.std([item["crps_9q"] for item in analysis["seed_metrics"][str(day)]], ddof=1) for day in budgets])
    full = analysis["full_finetuning"]
    full_mae = full["metrics"]["mae"]
    full_crps = full["metrics"]["crps_9q"]
    full_mae_sd = np.std([item["mae"] for item in full["seed_metrics"]], ddof=1) if len(full["seed_metrics"]) > 1 else 0.0
    full_crps_sd = np.std([item["crps_9q"] for item in full["seed_metrics"]], ddof=1) if len(full["seed_metrics"]) > 1 else 0.0

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    panels = [
        (axes[0], mae, mae_sd, analysis["zero_shot_metrics"]["mae"], full_mae, full_mae_sd, analysis["physical_lightgbm_metrics"]["mae"], "Daylight MAE (p.u.)"),
        (axes[1], crps, crps_sd, analysis["zero_shot_metrics"]["crps_9q"], full_crps, full_crps_sd, analysis["physical_lightgbm_metrics"]["crps_9q"], r"Daylight $\mathrm{CRPS}_{9q}$ (p.u.)"),
    ]
    for ax, values, spread, zero, full_value, full_spread, lgb, label in panels:
        ax.errorbar(budgets, values, yerr=spread, marker="o", color="#1f5a94",
                    capsize=2.5, linewidth=1.2, label="Low-rank, 3 seeds")
        ax.axhline(zero, color="#6b6b6b", linestyle="--", linewidth=1.0, label="Zero-shot")
        ax.errorbar([30], [full_value], yerr=[full_spread], marker="D", markersize=5.5,
                    color="#b44b3a", capsize=2.5, linewidth=1.2, label="Full fine-tuning, 3 seeds")
        ax.axhline(lgb, color="#2f7d4a", linestyle=":", linewidth=1.2, label="Physical LightGBM")
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(v) for v in budgets])
        ax.set_xlabel("Adaptation days per site")
        ax.set_ylabel(label)
        ax.grid(color="0.90", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(top=0.80)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "chronos_adaptation_budget.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "chronos_adaptation_budget.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
