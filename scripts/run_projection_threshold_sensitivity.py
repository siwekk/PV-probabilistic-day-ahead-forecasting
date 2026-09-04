#!/usr/bin/env python3
"""Evaluate hard-projection thresholds on frozen raw NREL Chronos-2 quantiles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics import metrics


COMMON_LEVELS = np.asarray([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
NATIVE_LEVELS = np.asarray([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
NATIVE_COLUMNS = [
    "chronos2_q050_raw",
    "chronos2_q100_raw",
    "chronos2_q250_raw",
    "chronos2_q500_raw",
    "chronos2_q750_raw",
    "chronos2_q900_raw",
    "chronos2_q950_raw",
]
DAYLIGHT_THRESHOLDS = (1.0, 5.0, 10.0, 20.0)
UPPER_BOUNDS = (1.0, 1.1, 1.2, 1.3)


def interpolate_quantiles(native: np.ndarray) -> np.ndarray:
    output = np.empty((len(native), len(COMMON_LEVELS)), dtype=np.float64)
    for column, level in enumerate(COMMON_LEVELS):
        right = int(np.searchsorted(NATIVE_LEVELS, level, side="left"))
        if right < len(NATIVE_LEVELS) and NATIVE_LEVELS[right] == level:
            output[:, column] = native[:, right]
            continue
        left = right - 1
        weight = (level - NATIVE_LEVELS[left]) / (NATIVE_LEVELS[right] - NATIVE_LEVELS[left])
        output[:, column] = native[:, left] + weight * (native[:, right] - native[:, left])
    return output


def evaluate(frame: pd.DataFrame, raw: np.ndarray, threshold: float, upper: float) -> dict:
    ordered = np.sort(raw, axis=1)
    projected = np.clip(ordered, 0.0, upper)
    projected[frame["clear_sky_ghi_w_m2"].to_numpy(float) < threshold] = 0.0
    irradiance = frame["clear_sky_ghi_w_m2"].to_numpy(float)
    target = frame["target_power_normalized"].to_numpy(float)
    solar_positive = irradiance >= 1.0
    conventional_daylight = irradiance >= 20.0
    low_light = (irradiance >= 1.0) & (irradiance < 20.0)
    result = {
        "g_min_w_m2": threshold,
        "y_max_pu": upper,
        "solar_positive": metrics(target[solar_positive], projected[solar_positive]),
        "conventional_daylight": metrics(target[conventional_daylight], projected[conventional_daylight]),
        "low_light": metrics(target[low_light], projected[low_light]),
        "low_light_rows": int(low_light.sum()),
        "low_light_positive_target_rate": float(np.mean(target[low_light] > 1e-4)),
        "low_light_mean_target": float(np.mean(target[low_light])),
        "rows_zeroed_rate": float(np.mean(irradiance < threshold)),
        "raw_quantile_above_bound_rate": float(np.mean(ordered > upper)),
    }
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prediction = pd.read_parquet(root / "results" / "nrel_chronos2_predictions.parquet")
    panel = pd.read_parquet(
        root / "data" / "processed" / "nrel_q1_panel.parquet",
        columns=["state", "site_id", "LocalTime", "clear_sky_ghi_w_m2"],
    )
    prediction["LocalTime"] = pd.to_datetime(prediction["LocalTime"])
    panel["LocalTime"] = pd.to_datetime(panel["LocalTime"])
    frame = prediction.merge(panel, on=["state", "site_id", "LocalTime"], how="left", validate="one_to_one")
    if frame["clear_sky_ghi_w_m2"].isna().any():
        raise RuntimeError("Clear-sky irradiance did not merge onto every frozen prediction row")
    raw = interpolate_quantiles(frame[NATIVE_COLUMNS].to_numpy(float))
    combinations = [
        evaluate(frame, raw, threshold, upper)
        for threshold in DAYLIGHT_THRESHOLDS
        for upper in UPPER_BOUNDS
    ]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen raw NREL Chronos-2 predictions; descriptive hard-projection sensitivity without refitting.",
        "evaluation_contract": {
            "fixed_solar_positive_support": "clear-sky GHI >= 1 W m-2",
            "conventional_daylight_support": "clear-sky GHI >= 20 W m-2",
            "low_light_support": "1 <= clear-sky GHI < 20 W m-2",
            "quantile_grid": COMMON_LEVELS.tolist(),
            "night_action": "Set every quantile to zero below G_min after ordering and clipping.",
        },
        "combinations": combinations,
    }
    target = root / "results" / "projection_threshold_sensitivity.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
