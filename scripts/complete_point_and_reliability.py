#!/usr/bin/env python3
"""Complete point metrics and create reliability and day-example figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
LEVELS = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
METHODS = {
    "lightgbm_nonphysical": "Non-physical LightGBM",
    "lightgbm_physical": "Physical LightGBM",
    "xgboost_physical": "Physical XGBoost",
}


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    denominator = np.abs(target) + np.abs(prediction)
    smape = np.divide(200.0 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0)
    total = np.sum((target - np.mean(target)) ** 2)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "smape_percent": float(np.mean(smape)),
        "r2": float(1.0 - np.sum(error**2) / total),
    }


def empirical_probabilities(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    target = frame["target_power_normalized"].to_numpy(float)
    values = []
    for level in LEVELS:
        suffix = f"q{int(round(level * 100)):02d}"
        values.append(float(np.mean(target <= frame[f"{prefix}_{suffix}"].to_numpy(float))))
    return np.asarray(values)


def season(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
            ["Winter", "Spring", "Summer"],
            default="Autumn",
        ),
        index=month.index,
    )


def add_reliability_line(ax, frame: pd.DataFrame, prefix: str, label: str) -> None:
    ax.plot(LEVELS, empirical_probabilities(frame, prefix), marker="o", markersize=3, linewidth=1.1, label=label)


def reliability_figure(nrel: pd.DataFrame, pvod: pd.DataFrame) -> dict:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
    diagnostics: dict[str, dict] = {}

    for frame, ax, name in [(nrel, axes[0, 0], "NREL"), (pvod, axes[0, 1], "PVOD")]:
        diagnostics[f"{name.lower()}_methods"] = {}
        for prefix, label in METHODS.items():
            add_reliability_line(ax, frame, prefix, label)
            diagnostics[f"{name.lower()}_methods"][label] = empirical_probabilities(frame, prefix).tolist()
        ax.set_title(f"{name}: methods")

    nrel = nrel.copy()
    nrel["month"] = pd.to_datetime(nrel["LocalTime"]).dt.month
    nrel["season"] = season(nrel["month"])
    diagnostics["nrel_season"] = {}
    for label in ["Winter", "Autumn"]:
        group = nrel.loc[nrel["season"] == label]
        add_reliability_line(axes[0, 2], group, "lightgbm_physical", label)
        diagnostics["nrel_season"][label] = empirical_probabilities(group, "lightgbm_physical").tolist()
    axes[0, 2].set_title("NREL physical: season")

    diagnostics["nrel_state"] = {}
    for label, group in nrel.groupby("state", observed=True):
        add_reliability_line(axes[1, 0], group, "lightgbm_physical", str(label))
        diagnostics["nrel_state"][str(label)] = empirical_probabilities(group, "lightgbm_physical").tolist()
    axes[1, 0].set_title("NREL physical: state")

    nrel["lead_band"] = pd.cut(nrel["delivery_hour"], [-1, 5, 11, 17, 23], labels=["00--05", "06--11", "12--17", "18--23"])
    diagnostics["nrel_delivery_hour"] = {}
    for label, group in nrel.groupby("lead_band", observed=True):
        add_reliability_line(axes[1, 1], group, "lightgbm_physical", str(label))
        diagnostics["nrel_delivery_hour"][str(label)] = empirical_probabilities(group, "lightgbm_physical").tolist()
    axes[1, 1].set_title("NREL physical: delivery hour")

    pvod = pvod.copy()
    pvod["lead_hours"] = pvod["lead_15min"] / 4.0
    pvod["lead_band"] = pd.cut(pvod["lead_hours"], [27.9, 33.9, 39.9, 45.9, 52.0], labels=["28--33", "34--39", "40--45", "46--52"])
    diagnostics["pvod_lead"] = {}
    for label, group in pvod.groupby("lead_band", observed=True):
        add_reliability_line(axes[1, 2], group, "lightgbm_nonphysical", str(label))
        diagnostics["pvod_lead"][str(label)] = empirical_probabilities(group, "lightgbm_nonphysical").tolist()
    axes[1, 2].set_title("PVOD non-physical: lead (h)")

    for ax in axes.flat:
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8, label="Ideal")
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(0.03, 0.97)
        ax.set_xticks([0.05, 0.30, 0.50, 0.70, 0.95])
        ax.set_yticks([0.05, 0.30, 0.50, 0.70, 0.95])
        ax.grid(color="0.90", linewidth=0.5)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Empirical probability")
    for ax in axes[1, :]:
        ax.set_xlabel("Nominal quantile probability")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "quantile_reliability.pdf")
    fig.savefig(FIGURES / "quantile_reliability.png", dpi=300)
    plt.close(fig)
    return diagnostics


def day_examples() -> dict:
    base = pd.read_parquet(RESULTS / "nrel_physics_ladder_predictions.parquet")
    keys = ["state", "site_id", "LocalTime"]
    chronos = pd.read_parquet(RESULTS / "nrel_chronos2_predictions.parquet")
    chronos_columns = keys + ["chronos2_q050_projected", "chronos2_q500_projected", "chronos2_q950_projected"]
    base = base.merge(chronos[chronos_columns], on=keys, validate="one_to_one")
    daylight = base.loc[base["daylight"]].copy()

    def daily_stats(group: pd.DataFrame) -> pd.Series:
        target = group.sort_values("delivery_hour")["target_power_normalized"].to_numpy(float)
        gain = np.mean(np.abs(target - group.sort_values("delivery_hour")["source_forecast_normalized"].to_numpy(float)))
        gain -= np.mean(np.abs(target - group.sort_values("delivery_hour")["M5_q50"].to_numpy(float)))
        ramp = float(np.max(np.abs(np.diff(target)))) if len(target) > 1 else 0.0
        return pd.Series({"gain": gain, "max_realised_ramp": ramp})

    daily = daylight.groupby(["state", "site_id", "delivery_date"], observed=True).apply(daily_stats, include_groups=False)
    typical_gain = daily["gain"].median()
    typical = (daily["gain"] - typical_gain).abs().idxmin()
    threshold = daily["max_realised_ramp"].quantile(0.90)
    difficult_pool = daily.loc[daily["max_realised_ramp"] >= threshold]
    difficult_gain = difficult_pool["gain"].median()
    difficult = (difficult_pool["gain"] - difficult_gain).abs().idxmin()

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharex=True)
    records = {}
    for ax, chosen, title in [(axes[0], typical, "Typical improvement day"), (axes[1], difficult, "High-ramp day")]:
        day = base.loc[(base["state"] == chosen[0]) & (base["site_id"] == chosen[1]) & (base["delivery_date"] == chosen[2])].sort_values("delivery_hour")
        hour = day["delivery_hour"].to_numpy()
        ax.fill_between(hour, day["M5_q05"], day["M5_q95"], color="#3b7aaa", alpha=0.16, label="Physical QGBM 90% interval")
        ax.plot(hour, day["M5_q50"], color="#1f5a94", linewidth=1.4, label="Physical QGBM median")
        ax.fill_between(hour, day["chronos2_q050_projected"], day["chronos2_q950_projected"], color="#d27c39", alpha=0.10, hatch="///", label="Chronos-2 90% interval")
        ax.plot(hour, day["chronos2_q500_projected"], color="#b45f22", linewidth=1.2, label="Chronos-2 median")
        ax.plot(hour, day["source_forecast_normalized"], color="0.45", linestyle="--", linewidth=1.1, label="Issued DA forecast")
        ax.plot(hour, day["target_power_normalized"], color="black", linewidth=1.5, label="Realised power")
        ax.set_title(title, loc="left", pad=4)
        ax.set_ylabel("Capacity-normalised power")
        ax.set_xlim(0, 23)
        ax.set_ylim(bottom=0)
        ax.grid(color="0.90", linewidth=0.5)
        records[title] = {"state": str(chosen[0]), "site_id": str(chosen[1]), "delivery_date": str(chosen[2]), **daily.loc[chosen].to_dict()}
    axes[1].set_xlabel("Local delivery hour")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.90, bottom=0.09, left=0.10, right=0.99, hspace=0.24)
    fig.savefig(FIGURES / "nrel_day_examples.pdf")
    fig.savefig(FIGURES / "nrel_day_examples.png", dpi=300)
    plt.close(fig)
    records["high_ramp_threshold"] = float(threshold)
    return records


def main() -> None:
    nrel_all = pd.read_parquet(RESULTS / "nrel_nine_quantile_review_predictions.parquet")
    pvod_all = pd.read_parquet(RESULTS / "pvod_nine_quantile_review_predictions.parquet")
    nrel = nrel_all.loc[nrel_all["daylight"]].copy()
    pvod = pvod_all.loc[pvod_all["daylight"]].copy()
    result = {"point_metrics": {}, "quantile_levels": LEVELS.tolist()}
    for name, frame in [("NREL", nrel), ("PVOD", pvod)]:
        result["point_metrics"][name] = {}
        target = frame["target_power_normalized"].to_numpy(float)
        for prefix, label in METHODS.items():
            result["point_metrics"][name][label] = point_metrics(target, frame[f"{prefix}_q50"].to_numpy(float))
    result["quantile_reliability"] = reliability_figure(nrel, pvod)
    result["day_examples"] = day_examples()
    (RESULTS / "point_reliability_day_review.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["point_metrics"], indent=2))
    print(json.dumps(result["day_examples"], indent=2))


if __name__ == "__main__":
    main()
