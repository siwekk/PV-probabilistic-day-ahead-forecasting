#!/usr/bin/env python3
"""Test a physically reconstructed NWP proxy residual model on PVOD.

This is an exploratory response to the failure of the preregistered clear-sky
persistence residual.  It must not be described as confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_nrel_physics_ladder import conformal_radius, score


QUANTILES = (0.05, 0.50, 0.95)
SEED = 20260828


def evaluate(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict:
    daylight = frame["daylight"].to_numpy(dtype=bool)
    return {
        "overall": score(frame, low, median, high),
        "daylight": score(frame.loc[daylight], low[daylight], median[daylight], high[daylight]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_q1_physics_study.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    frame["station_code"] = frame["station"].astype("category")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["daylight"] = frame["clearsky_ghi"] >= 20.0
    frame["solar_elevation_factor"] = np.clip(
        np.cos(np.deg2rad(frame["solar_zenith"].to_numpy(dtype=float))), 0.0, 1.0
    )
    frame["nwp_power_proxy"] = np.clip(frame["nwp_globalirrad"] / 1000.0, 0.0, 1.2)
    frame["nwp_clearness_index"] = np.clip(
        frame["nwp_globalirrad"] / np.maximum(frame["clearsky_ghi"], 20.0), 0.0, 2.0
    )
    frame["nwp_direct_fraction"] = np.clip(
        frame["nwp_directirrad"] / np.maximum(frame["nwp_globalirrad"], 20.0), 0.0, 2.0
    )
    features = [
        "station_code",
        "lead_15min",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
        "nwp_globalirrad",
        "nwp_directirrad",
        "nwp_temperature",
        "nwp_humidity",
        "nwp_windspeed",
        "nwp_winddirection_sin",
        "nwp_winddirection_cos",
        "nwp_pressure",
        "origin_power",
        "origin_power_lag_4",
        "origin_power_lag_96",
        "solar_zenith",
        "clearsky_ghi",
        "solar_elevation_factor",
        "nwp_power_proxy",
        "nwp_clearness_index",
        "nwp_direct_fraction",
    ]
    frame = frame.dropna(subset=features + ["target_power_normalized"]).copy()
    frame["physical_proxy_residual"] = (
        frame["target_power_normalized"] - frame["nwp_power_proxy"]
    )
    timestamp = frame["target_timestamp_utc"]
    split = design["split"]
    train = frame.loc[timestamp <= pd.Timestamp(split["train_end"])].copy()
    tuning = frame.loc[
        (timestamp >= pd.Timestamp(split["tuning_start"]))
        & (timestamp <= pd.Timestamp(split["tuning_end"]))
    ].copy()
    calibration = frame.loc[
        (timestamp >= pd.Timestamp(split["calibration_start"]))
        & (timestamp <= pd.Timestamp(split["calibration_end"]))
    ].copy()
    test = frame.loc[timestamp >= pd.Timestamp(split["test_start"])].copy()
    predictions = {}
    iterations = {}
    timings = {}
    for quantile in QUANTILES:
        started = time.perf_counter()
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=1400,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=100,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=32,
            verbosity=-1,
            random_state=SEED,
        )
        model.fit(
            train[features],
            train["physical_proxy_residual"],
            categorical_feature=["station_code"],
            eval_set=[(tuning[features], tuning["physical_proxy_residual"])],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        best_iteration = int(model.best_iteration_ or model.n_estimators)
        development = pd.concat([train, tuning], ignore_index=True)
        final = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            n_estimators=best_iteration,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=100,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=32,
            verbosity=-1,
            random_state=SEED,
        )
        final.fit(
            development[features],
            development["physical_proxy_residual"],
            categorical_feature=["station_code"],
        )
        cal_prediction = np.clip(
            final.predict(calibration[features]) + calibration["nwp_power_proxy"].to_numpy(),
            0.0,
            1.2,
        )
        test_prediction = np.clip(
            final.predict(test[features]) + test["nwp_power_proxy"].to_numpy(), 0.0, 1.2
        )
        predictions[quantile] = {"calibration": cal_prediction, "test": test_prediction}
        iterations[f"quantile_{quantile:.2f}"] = best_iteration
        timings[f"quantile_{quantile:.2f}_seconds"] = float(time.perf_counter() - started)

    calibration_ordered = np.sort(
        np.column_stack([predictions[q]["calibration"] for q in QUANTILES]), axis=1
    )
    test_ordered = np.sort(np.column_stack([predictions[q]["test"] for q in QUANTILES]), axis=1)
    calibration_daylight = calibration["daylight"].to_numpy(dtype=bool)
    radius = conformal_radius(
        calibration.loc[calibration_daylight],
        calibration_ordered[calibration_daylight, 0],
        calibration_ordered[calibration_daylight, 2],
    )
    low, median, high = test_ordered[:, 0], test_ordered[:, 1], test_ordered[:, 2]
    test_daylight = test["daylight"].to_numpy(dtype=bool)
    calibrated_low = low.copy()
    calibrated_high = high.copy()
    calibrated_low[test_daylight] = np.clip(low[test_daylight] - radius, 0.0, 1.2)
    calibrated_high[test_daylight] = np.clip(high[test_daylight] + radius, 0.0, 1.2)
    proxy = test["nwp_power_proxy"].to_numpy(dtype=float)
    y = test["target_power_normalized"].to_numpy(dtype=float)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory after the preregistered physical ladder failed",
        "features": features,
        "rows": {"train": len(train), "tuning": len(tuning), "calibration": len(calibration), "test": len(test)},
        "physical_proxy_daylight_mae": float(np.mean(np.abs(y[test_daylight] - proxy[test_daylight]))),
        "best_iterations": iterations,
        "timings": timings,
        "conformal_radius": radius,
        "raw": evaluate(test, low, median, high),
        "conformal": evaluate(test, calibrated_low, median, calibrated_high),
    }
    destination = root / "results" / "pvod_physical_proxy_exploratory.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
