#!/usr/bin/env python3
"""Run a matched XGBoost quantile benchmark for the complete NREL model."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from run_nrel_physics_ladder import conformal_radius, score


SEED = 20260828
QUANTILES = (0.05, 0.50, 0.95)
FEATURES = [
    "source_forecast_normalized",
    "delivery_hour",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "latitude",
    "longitude",
    "capacity_mw",
    "plant_type",
    "site_code",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "clear_sky_ghi_w_m2",
    "clear_sky_envelope",
    "trajectory_mean",
    "trajectory_max",
    "trajectory_energy_proxy",
    "trajectory_ramp",
    "trajectory_abs_ramp",
    "trajectory_max_abs_ramp",
    "trajectory_peak_hour",
]


def evaluate(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict:
    daylight = frame["daylight"].to_numpy(dtype=bool)
    return {
        "overall": score(frame, low, median, high),
        "daylight": score(frame.loc[daylight], low[daylight], median[daylight], high[daylight]),
        "night": score(frame.loc[~daylight], low[~daylight], median[~daylight], high[~daylight]),
    }


def make_model(quantile: float, n_estimators: int, early_stopping_rounds: int | None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=n_estimators,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=50,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        enable_categorical=True,
        early_stopping_rounds=early_stopping_rounds,
        random_state=SEED,
        n_jobs=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(root / "data" / "processed" / "nrel_q1_panel.parquet")
    frame["plant_type"] = frame["plant_type"].astype("category")
    frame["site_code"] = frame["site_id"].astype("category")
    local_time = pd.to_datetime(frame["LocalTime"])
    split = design["known_site_temporal_split"]
    train = frame.loc[local_time <= pd.Timestamp(split["train_end"])].copy()
    tuning = frame.loc[
        (local_time >= pd.Timestamp(split["tuning_start"]))
        & (local_time <= pd.Timestamp(split["tuning_end"]))
    ].copy()
    calibration = frame.loc[
        (local_time >= pd.Timestamp(split["calibration_start"]))
        & (local_time <= pd.Timestamp(split["calibration_end"]))
    ].copy()
    test = frame.loc[local_time >= pd.Timestamp(split["test_start"])].copy()
    target = "source_residual_normalized"
    predictions: dict[float, dict[str, np.ndarray]] = {}
    best_iterations = {}
    timings = {}
    for quantile in QUANTILES:
        started = time.perf_counter()
        model = make_model(quantile, 1600, 80)
        model.fit(
            train[FEATURES],
            train[target],
            eval_set=[(tuning[FEATURES], tuning[target])],
            verbose=False,
        )
        best_iteration = int(model.best_iteration + 1)
        development = pd.concat([train, tuning], ignore_index=True)
        final_model = make_model(quantile, best_iteration, None)
        final_model.fit(development[FEATURES], development[target], verbose=False)
        calibration_prediction = np.clip(
            final_model.predict(calibration[FEATURES])
            + calibration["source_forecast_normalized"].to_numpy(),
            0.0,
            1.2,
        )
        test_prediction = np.clip(
            final_model.predict(test[FEATURES]) + test["source_forecast_normalized"].to_numpy(),
            0.0,
            1.2,
        )
        predictions[quantile] = {
            "calibration": calibration_prediction,
            "test": test_prediction,
        }
        best_iterations[f"quantile_{quantile:.2f}"] = best_iteration
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
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "xgboost_version": xgb.__version__,
        "method": "Matched XGBoost quantile residual model with native reg:quantileerror",
        "features": FEATURES,
        "best_iterations": best_iterations,
        "timings": timings,
        "conformal_scope": "daylight calibration rows, applied to daylight test rows",
        "conformal_radius": radius,
        "raw": evaluate(test, low, median, high),
        "conformal": evaluate(test, calibrated_low, median, calibrated_high),
    }
    result_path = root / "results" / "nrel_matched_xgboost.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    prediction_path = root / "results" / "nrel_matched_xgboost_predictions.parquet"
    prediction_frame = test[
        ["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized"]
    ].reset_index(drop=True)
    prediction_frame["xgb_q05"] = low
    prediction_frame["xgb_q50"] = median
    prediction_frame["xgb_q95"] = high
    prediction_frame.to_parquet(prediction_path, index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
