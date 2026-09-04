#!/usr/bin/env python3
"""Run a complete-site and future-time NREL holdout."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_nrel_physics_ladder import conformal_radius, score


SEED = 20260828
QUANTILES = (0.05, 0.50, 0.95)


def specifications() -> dict[str, dict[str, Any]]:
    basic = [
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
        "state_code",
    ]
    physical = basic + [
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
    return {
        "nonphysical_direct": {"features": basic, "target": "target_power_normalized", "residual": False},
        "physics_residual_trajectory": {"features": physical, "target": "source_residual_normalized", "residual": True},
    }


def evaluate(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    daylight = frame["daylight"].to_numpy(dtype=bool)
    result: dict[str, Any] = {
        "overall": score(frame, low, median, high),
        "daylight": score(frame.loc[daylight], low[daylight], median[daylight], high[daylight]),
        "by_state": {},
    }
    for state in sorted(frame["state"].unique()):
        mask = frame["state"].to_numpy() == state
        state_daylight = mask & daylight
        result["by_state"][state] = score(
            frame.loc[state_daylight], low[state_daylight], median[state_daylight], high[state_daylight]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(root / "data" / "processed" / "nrel_q1_panel.parquet")
    frame["plant_type"] = frame["plant_type"].astype("category")
    frame["state_code"] = frame["state"].astype("category")
    local_time = pd.to_datetime(frame["LocalTime"])
    split = design["known_site_temporal_split"]
    fit = frame.loc[
        (frame["site_split"] == "train") & (local_time <= pd.Timestamp(split["train_end"]))
    ].copy()
    tuning = frame.loc[
        (frame["site_split"] == "train")
        & (local_time >= pd.Timestamp(split["tuning_start"]))
        & (local_time <= pd.Timestamp(split["tuning_end"]))
    ].copy()
    calibration = frame.loc[
        (frame["site_split"] == "calibration")
        & (local_time >= pd.Timestamp(split["calibration_start"]))
        & (local_time <= pd.Timestamp(split["calibration_end"]))
    ].copy()
    test = frame.loc[
        (frame["site_split"] == "test") & (local_time >= pd.Timestamp(split["test_start"]))
    ].copy()
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "protocol": (
            "Fit on training sites through July, tune on the same training sites in August, "
            "calibrate on separate sites in September and October, and test once on unseen sites "
            "in November and December. Site identifiers are excluded from all features."
        ),
        "rows": {
            "fit": int(len(fit)),
            "tuning": int(len(tuning)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "sites": {
            "fit": int(fit["site_id"].nunique()),
            "tuning": int(tuning["site_id"].nunique()),
            "calibration": int(calibration["site_id"].nunique()),
            "test": int(test["site_id"].nunique()),
        },
        "models": {},
    }
    output = test[
        ["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized", "source_forecast_normalized"]
    ].reset_index(drop=True)
    source = test["source_forecast_normalized"].to_numpy(dtype=float)
    for name, specification in specifications().items():
        features = specification["features"]
        target = specification["target"]
        predictions: dict[float, dict[str, np.ndarray]] = {}
        iterations = {}
        timings = {}
        for quantile in QUANTILES:
            started = time.perf_counter()
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=1600,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=200,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                n_jobs=32,
                verbosity=-1,
                random_state=SEED,
            )
            model.fit(
                fit[features],
                fit[target],
                categorical_feature=["plant_type", "state_code"],
                eval_set=[(tuning[features], tuning[target])],
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            best_iteration = int(model.best_iteration_ or model.n_estimators)
            development = pd.concat([fit, tuning], ignore_index=True)
            final = lgb.LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=best_iteration,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=200,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                n_jobs=32,
                verbosity=-1,
                random_state=SEED,
            )
            final.fit(
                development[features],
                development[target],
                categorical_feature=["plant_type", "state_code"],
            )
            calibration_prediction = final.predict(calibration[features])
            test_prediction = final.predict(test[features])
            if specification["residual"]:
                calibration_prediction = np.clip(
                    calibration_prediction + calibration["source_forecast_normalized"].to_numpy(),
                    0.0,
                    1.2,
                )
                test_prediction = np.clip(test_prediction + source, 0.0, 1.2)
            predictions[quantile] = {
                "calibration": calibration_prediction,
                "test": test_prediction,
            }
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
        result["models"][name] = {
            "features": features,
            "best_iterations": iterations,
            "timings": timings,
            "conformal_radius": radius,
            "raw": evaluate(test, low, median, high),
            "conformal": evaluate(test, calibrated_low, median, calibrated_high),
        }
        prefix = "nonphysics" if name.startswith("nonphysical") else "physics"
        output[f"{prefix}_q05"] = low
        output[f"{prefix}_q50"] = median
        output[f"{prefix}_q95"] = high

    result_path = root / "results" / "nrel_site_holdout.json"
    prediction_path = root / "results" / "nrel_site_holdout_predictions.parquet"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    output.to_parquet(prediction_path, index=False)
    print(
        json.dumps(
            {
                "rows": result["rows"],
                "sites": result["sites"],
                "nonphysical_daylight": result["models"]["nonphysical_direct"]["raw"]["daylight"],
                "physics_daylight": result["models"]["physics_residual_trajectory"]["raw"]["daylight"],
                "physics_conformal_daylight": result["models"]["physics_residual_trajectory"]["conformal"]["daylight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
