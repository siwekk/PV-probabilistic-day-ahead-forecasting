#!/usr/bin/env python3
"""Dependence-aware uncertainty for global versus rolling state-transfer calibration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260829
DRAWS = 2000
BLOCK = 7


def prepare(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    y = frame["target_power_normalized"].to_numpy(float)
    low = frame[f"physics_q05_{method}"].to_numpy(float)
    high = frame[f"physics_q95_{method}"].to_numpy(float)
    median = frame["physics_q50"].to_numpy(float)
    is90 = high - low + 20 * (low - y) * (y < low) + 20 * (y - high) * (y > high)
    return pd.DataFrame({
        "site_id": frame["site_id"].to_numpy(), "date": pd.to_datetime(frame["delivery_date"]).to_numpy(),
        "coverage": ((y >= low) & (y <= high)).astype(float), "width": high - low,
        "interval_score": is90, "wis90": (0.5 * np.abs(y - median) + 0.05 * is90) / 1.5,
    }).groupby(["date", "site_id"], observed=True).mean().reset_index()


def matrices(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    dates = np.sort(frame["date"].unique())
    sites = sorted(frame["site_id"].astype(str).unique())
    output = {}
    for metric in ["coverage", "width", "interval_score", "wis90"]:
        pivot = frame.pivot(index="date", columns="site_id", values=metric).reindex(index=dates, columns=sites)
        output[metric] = pivot.to_numpy(float)
    return dates, sites, output


def bootstrap(state_frame: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    dates, sites, global_m = matrices(prepare(state_frame, "global"))
    dates2, sites2, rolling_m = matrices(prepare(state_frame, "rolling_28d"))
    assert np.array_equal(dates, dates2) and sites == sites2
    n_days, n_sites = len(dates), len(sites)
    draws = {metric: [] for metric in global_m}
    seasonal = {season: {method: [] for method in ["global", "rolling_28d"]} for season in ["winter", "spring", "summer", "autumn"]}
    month = pd.DatetimeIndex(dates).month
    season_array = np.select([np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8])], ["winter", "spring", "summer"], default="autumn")
    for _ in range(DRAWS):
        site_idx = rng.integers(0, n_sites, size=n_sites)
        starts = rng.integers(0, n_days, size=int(np.ceil(n_days / BLOCK)))
        day_idx = np.concatenate([(np.arange(BLOCK) + start) % n_days for start in starts])[:n_days]
        for metric in draws:
            draws[metric].append(float(np.nanmean(rolling_m[metric][day_idx][:, site_idx]) - np.nanmean(global_m[metric][day_idx][:, site_idx])))
        sampled_season = season_array[day_idx]
        for season in seasonal:
            mask = sampled_season == season
            seasonal[season]["global"].append(float(np.nanmean(global_m["coverage"][day_idx[mask]][:, site_idx])))
            seasonal[season]["rolling_28d"].append(float(np.nanmean(rolling_m["coverage"][day_idx[mask]][:, site_idx])))
    result = {"rolling_minus_global": {}}
    for metric, values in draws.items():
        point = float(np.nanmean(rolling_m[metric]) - np.nanmean(global_m[metric]))
        result["rolling_minus_global"][metric] = {"estimate": point, "ci95": np.quantile(values, [0.025, 0.975]).tolist()}
    result["seasonal_coverage"] = {}
    for season, methods in seasonal.items():
        result["seasonal_coverage"][season] = {
            method: {"estimate": float(np.nanmean(matrix["coverage"][season_array == season])), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}
            for method, values in methods.items()
            for matrix in [global_m if method == "global" else rolling_m]
        }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frame = pd.read_parquet(root / "results/nrel_state_holdout_predictions.parquet")
    frame = frame.loc[frame["daylight"]].copy()
    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "draws": DRAWS, "block_days": BLOCK, "states": {}}
    for state, group in frame.groupby("state", observed=True):
        output["states"][str(state)] = bootstrap(group)
    path = root / "results/conditional_calibration_uncertainty.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
