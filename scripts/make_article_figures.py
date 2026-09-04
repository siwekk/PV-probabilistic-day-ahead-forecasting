#!/usr/bin/env python3
"""Create publication figures from frozen prediction files."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def method_comparison(metrics: pd.DataFrame) -> None:
    data = metrics.loc[metrics["dataset"] == "NREL known-site daylight"].copy()
    short_names = {
        "Physical residual and trajectory LightGBM": "Physical residual + trajectory",
        "Solar-geometry LightGBM": "Solar geometry",
        "Matched XGBoost quantile": "XGBoost quantile",
        "Clear-sky LightGBM": "Clear sky",
        "Physical residual LightGBM": "Physical residual",
        "Chronos-2 zero-shot, projected": "Chronos-2 zero-shot",
        "Non-physical direct LightGBM": "Non-physical LightGBM",
        "Issued DA forecast": "Issued DA forecast",
    }
    data["short_method"] = data["method"].map(short_names)
    data = data.sort_values("mae", ascending=True)
    positions = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharey=True, gridspec_kw={"wspace": 0.12})
    axes[0].scatter(data["mae"], positions, color="#1f5a94", s=30)
    axes[0].set_yticks(positions, data["short_method"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Daylight MAE (p.u.)")
    axes[0].grid(axis="x", color="0.88", linewidth=0.6)
    probabilistic = data.dropna(subset=["crps_3q"])
    ppos = probabilistic.index.map({index: position for position, index in enumerate(data.index)})
    axes[1].scatter(probabilistic["crps_3q"], ppos, color="#b44b3a", s=30)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_xlabel(r"Daylight $\mathrm{CRPS}_{3q}$ (p.u.)")
    axes[1].grid(axis="x", color="0.88", linewidth=0.6)
    axes[0].set_title("Point accuracy")
    axes[1].set_title("Distributional accuracy")
    fig.savefig(FIGURES / "nrel_complete_method_comparison.pdf")
    plt.close(fig)


def representative_day() -> None:
    base = pd.read_parquet(RESULTS / "nrel_physics_ladder_predictions.parquet")
    keys = ["state", "site_id", "LocalTime"]
    chronos_path = RESULTS / "nrel_chronos2_predictions.parquet"
    if chronos_path.exists():
        chronos = pd.read_parquet(chronos_path)
        cols = keys + [
            "chronos2_q050_projected",
            "chronos2_q500_projected",
            "chronos2_q950_projected",
        ]
        base = base.merge(chronos[cols], on=keys, validate="one_to_one")
    daylight = base.loc[base["daylight"]].copy()
    daily = daylight.groupby(["state", "site_id", "delivery_date"], observed=True).apply(
        lambda x: pd.Series(
            {
                "gain": np.mean(np.abs(x["target_power_normalized"] - x["source_forecast_normalized"]))
                - np.mean(np.abs(x["target_power_normalized"] - x["M5_q50"]))
            }
        ),
        include_groups=False,
    )
    median_gain = daily["gain"].median()
    chosen = (daily["gain"] - median_gain).abs().idxmin()
    day = base.loc[
        (base["state"] == chosen[0])
        & (base["site_id"] == chosen[1])
        & (base["delivery_date"] == chosen[2])
    ].sort_values("delivery_hour")
    hour = day["delivery_hour"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(hour, day["M5_q05"], day["M5_q95"], color="#3b7aaa", alpha=0.16, label="Physical QGBM 90% interval")
    ax.plot(hour, day["M5_q05"], color="#3b7aaa", linewidth=0.6, alpha=0.8)
    ax.plot(hour, day["M5_q95"], color="#3b7aaa", linewidth=0.6, alpha=0.8)
    ax.plot(hour, day["M5_q50"], color="#1f5a94", linewidth=1.5, label="Physical QGBM median")
    if "chronos2_q500_projected" in day:
        ax.fill_between(hour, day["chronos2_q050_projected"], day["chronos2_q950_projected"], facecolor="#d27c39", edgecolor="#b45f22", linewidth=0.6, hatch="///", alpha=0.11, label="Chronos-2 90% interval")
        ax.plot(hour, day["chronos2_q050_projected"], color="#b45f22", linewidth=0.6, alpha=0.8)
        ax.plot(hour, day["chronos2_q950_projected"], color="#b45f22", linewidth=0.6, alpha=0.8)
        ax.plot(hour, day["chronos2_q500_projected"], color="#b45f22", linewidth=1.3, label="Chronos-2 median")
    ax.plot(hour, day["source_forecast_normalized"], color="0.45", linestyle="--", linewidth=1.2, label="Issued DA forecast")
    ax.plot(hour, day["target_power_normalized"], color="black", linewidth=1.6, label="Realised power")
    ax.set_xlabel("Local delivery hour")
    ax.set_ylabel("Capacity-normalised power")
    ax.set_xlim(0, 23)
    ax.set_ylim(bottom=0)
    ax.grid(color="0.90", linewidth=0.6)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(FIGURES / "nrel_representative_day.pdf")
    plt.close(fig)


def transfer_skill() -> None:
    frame = pd.read_parquet(RESULTS / "nrel_state_holdout_predictions.parquet")
    records = []
    for state, group in frame.loc[frame["daylight"]].groupby("state", observed=True):
        target = group["target_power_normalized"].to_numpy(float)
        non_mae = np.mean(np.abs(target - group["nonphysics_q50"].to_numpy(float)))
        phy_mae = np.mean(np.abs(target - group["physics_q50"].to_numpy(float)))
        def interval_score(prefix: str) -> float:
            low = group[f"{prefix}_q05"].to_numpy(float)
            high = group[f"{prefix}_q95"].to_numpy(float)
            return float(np.mean(high - low + 20 * (low - target) * (target < low) + 20 * (target - high) * (target > high)))
        records.append((state, phy_mae - non_mae, interval_score("physics") - interval_score("nonphysics")))
    data = pd.DataFrame(records, columns=["state", "mae_delta", "is_delta"])
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    colors = np.where(data["mae_delta"] <= 0, "#2b7a3d", "#b33b32")
    axes[0].bar(x, data["mae_delta"], color=colors, width=0.62)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xticks(x, data["state"])
    axes[0].set_ylabel(r"$\Delta$ MAE, physical minus non-physical")
    colors = np.where(data["is_delta"] <= 0, "#2b7a3d", "#b33b32")
    axes[1].bar(x, data["is_delta"], color=colors, width=0.62)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_xticks(x, data["state"])
    axes[1].set_ylabel(r"$\Delta$ interval score")
    for ax in axes:
        ax.grid(axis="y", color="0.90", linewidth=0.6)
    fig.savefig(FIGURES / "nrel_state_transfer_skill.pdf")
    plt.close(fig)


def pvod_gate(metrics: pd.DataFrame) -> None:
    data = metrics.loc[
        (metrics["dataset"] == "PVOD daylight common sample")
        & metrics["method"].str.contains("LightGBM")
    ].copy()
    labels = ["No physics", "Geometry", "Clear sky", "Residual", "Trajectory"]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    axes[0].plot(x, data["mae"], marker="o", color="#1f5a94")
    axes[1].plot(x, data["interval_score_90"], marker="o", color="#b44b3a")
    axes[0].set_ylabel("Daylight MAE (p.u.)")
    axes[1].set_ylabel("90% interval score (p.u.)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", color="0.90", linewidth=0.6)
    fig.savefig(FIGURES / "pvod_physics_gate.pdf")
    plt.close(fig)


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "all_method_metrics.csv")
    method_comparison(metrics)
    representative_day()
    transfer_skill()
    pvod_gate(metrics)
    print("Created four article figures in", FIGURES)


if __name__ == "__main__":
    main()
