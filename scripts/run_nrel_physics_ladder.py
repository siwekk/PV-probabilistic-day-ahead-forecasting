#!/usr/bin/env python3
"""Run the controlled NREL physics ladder on the frozen temporal split."""

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


SEED = 20260828
QUANTILES = (0.05, 0.50, 0.95)


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = y - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def score(
    frame: pd.DataFrame,
    low: np.ndarray,
    median: np.ndarray,
    high: np.ndarray,
) -> dict[str, float | int]:
    y = frame["target_power_normalized"].to_numpy(dtype=float)
    alpha = 0.10
    interval_score = (high - low) + (2.0 / alpha) * (low - y) * (y < low) + (
        2.0 / alpha
    ) * (y - high) * (y > high)
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - median))),
        "rmse": float(np.sqrt(np.mean(np.square(y - median)))),
        "pinball_0.05": pinball(y, low, 0.05),
        "pinball_0.50": pinball(y, median, 0.50),
        "pinball_0.95": pinball(y, high, 0.95),
        "mean_pinball": float(
            np.mean([pinball(y, low, 0.05), pinball(y, median, 0.50), pinball(y, high, 0.95)])
        ),
        "coverage_90": float(np.mean((y >= low) & (y <= high))),
        "width_90": float(np.mean(high - low)),
        "interval_score_90": float(np.mean(interval_score)),
    }


def subsets(
    frame: pd.DataFrame,
    low: np.ndarray,
    median: np.ndarray,
    high: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "overall": score(frame, low, median, high),
        "daylight": score(frame.loc[frame["daylight"]], low[frame["daylight"]], median[frame["daylight"]], high[frame["daylight"]]),
        "night": score(frame.loc[~frame["daylight"]], low[~frame["daylight"]], median[~frame["daylight"]], high[~frame["daylight"]]),
        "by_state": {},
    }
    for state in sorted(frame["state"].unique()):
        mask = frame["state"].to_numpy() == state
        result["by_state"][state] = score(frame.loc[mask], low[mask], median[mask], high[mask])
    return result


def conformal_radius(frame: pd.DataFrame, low: np.ndarray, high: np.ndarray) -> float:
    y = frame["target_power_normalized"].to_numpy(dtype=float)
    nonconformity = np.maximum.reduce([low - y, y - high, np.zeros(len(y))])
    level = min(1.0, np.ceil((len(y) + 1) * 0.90) / len(y))
    return float(np.quantile(nonconformity, level, method="higher"))


def model_specifications() -> dict[str, dict[str, Any]]:
    base = [
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
    ]
    geometry = base + ["solar_zenith_deg", "solar_azimuth_deg"]
    clear_sky = geometry + ["clear_sky_ghi_w_m2", "clear_sky_envelope"]
    trajectory = clear_sky + [
        "trajectory_mean",
        "trajectory_max",
        "trajectory_energy_proxy",
        "trajectory_ramp",
        "trajectory_abs_ramp",
        "trajectory_max_abs_ramp",
        "trajectory_peak_hour",
    ]
    return {
        "M1_nonphysical_direct": {"features": base, "target": "direct", "clip": False},
        "M2_solar_geometry_direct": {"features": geometry, "target": "direct", "clip": False},
        "M3_clear_sky_direct": {"features": clear_sky, "target": "direct", "clip": False},
        "M4_physical_residual": {"features": clear_sky, "target": "residual", "clip": True},
        "M5_physical_residual_trajectory": {"features": trajectory, "target": "residual", "clip": True},
    }


def prepare_categories(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["plant_type"] = frame["plant_type"].astype("category")
    frame["site_code"] = frame["site_id"].astype("category")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    frame = prepare_categories(pd.read_parquet(root / "data" / "processed" / "nrel_q1_panel.parquet"))
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

    source = test["source_forecast_normalized"].to_numpy(dtype=float)
    y_test = test["target_power_normalized"].to_numpy(dtype=float)
    source_point = {
        "n": int(len(test)),
        "overall_mae": float(np.mean(np.abs(y_test - source))),
        "daylight_mae": float(np.mean(np.abs(y_test[test["daylight"]] - source[test["daylight"]]))),
        "rmse": float(np.sqrt(np.mean(np.square(y_test - source)))),
    }
    results: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "split": split,
        "rows": {
            "train": int(len(train)),
            "tuning": int(len(tuning)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "M0_source_forecast": source_point,
        "models": {},
    }
    prediction_frame = test[
        ["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized", "source_forecast_normalized"]
    ].reset_index(drop=True)

    for name, specification in model_specifications().items():
        features = specification["features"]
        categorical = [feature for feature in ("plant_type", "site_code") if feature in features]
        target_column = (
            "source_residual_normalized" if specification["target"] == "residual" else "target_power_normalized"
        )
        model_predictions: dict[float, dict[str, np.ndarray]] = {}
        timings = {}
        best_iterations = {}
        for quantile in QUANTILES:
            estimator = lgb.LGBMRegressor(
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
            started = time.perf_counter()
            estimator.fit(
                train[features],
                train[target_column],
                categorical_feature=categorical,
                eval_set=[(tuning[features], tuning[target_column])],
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            best_iteration = int(estimator.best_iteration_ or estimator.n_estimators)
            development = pd.concat([train, tuning], ignore_index=True)
            final_estimator = lgb.LGBMRegressor(
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
            final_estimator.fit(
                development[features], development[target_column], categorical_feature=categorical
            )
            calibration_prediction = final_estimator.predict(calibration[features])
            test_prediction = final_estimator.predict(test[features])
            if specification["target"] == "residual":
                calibration_prediction = calibration_prediction + calibration[
                    "source_forecast_normalized"
                ].to_numpy()
                test_prediction = test_prediction + source
            if specification["clip"]:
                calibration_prediction = np.clip(calibration_prediction, 0.0, 1.2)
                test_prediction = np.clip(test_prediction, 0.0, 1.2)
            model_predictions[quantile] = {
                "calibration": calibration_prediction,
                "test": test_prediction,
            }
            timings[f"quantile_{quantile:.2f}_seconds"] = float(time.perf_counter() - started)
            best_iterations[f"quantile_{quantile:.2f}"] = best_iteration

        low = model_predictions[0.05]["test"]
        median = model_predictions[0.50]["test"]
        high = model_predictions[0.95]["test"]
        crossing = (low > median) | (median > high)
        ordered = np.sort(np.column_stack([low, median, high]), axis=1)
        low, median, high = ordered[:, 0], ordered[:, 1], ordered[:, 2]
        calibration_ordered = np.sort(
            np.column_stack(
                [
                    model_predictions[0.05]["calibration"],
                    model_predictions[0.50]["calibration"],
                    model_predictions[0.95]["calibration"],
                ]
            ),
            axis=1,
        )
        calibration_daylight = calibration["daylight"].to_numpy(dtype=bool)
        radius = conformal_radius(
            calibration.loc[calibration_daylight],
            calibration_ordered[calibration_daylight, 0],
            calibration_ordered[calibration_daylight, 2],
        )
        test_daylight = test["daylight"].to_numpy(dtype=bool)
        calibrated_low = low.copy()
        calibrated_high = high.copy()
        calibrated_low[test_daylight] = np.clip(low[test_daylight] - radius, 0.0, 1.2)
        calibrated_high[test_daylight] = np.clip(high[test_daylight] + radius, 0.0, 1.2)
        results["models"][name] = {
            "features": features,
            "target_form": specification["target"],
            "physical_clip": bool(specification["clip"]),
            "best_iterations": best_iterations,
            "timings": timings,
            "quantile_crossing_fraction_before_sort": float(np.mean(crossing)),
            "conformal_scope": "daylight calibration rows, applied to daylight test rows",
            "conformal_radius": radius,
            "raw": subsets(test, low, median, high),
            "conformal": subsets(test, calibrated_low, median, calibrated_high),
        }
        prefix = name.split("_")[0]
        prediction_frame[f"{prefix}_q05"] = low
        prediction_frame[f"{prefix}_q50"] = median
        prediction_frame[f"{prefix}_q95"] = high
        prediction_frame[f"{prefix}_q05_conformal"] = calibrated_low
        prediction_frame[f"{prefix}_q95_conformal"] = calibrated_high

    result_path = root / "results" / "nrel_physics_ladder.json"
    prediction_path = root / "results" / "nrel_physics_ladder_predictions.parquet"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="ascii")
    prediction_frame.to_parquet(prediction_path, index=False)
    compact = {
        "M0_source_forecast": source_point,
        "models": {
            name: {
                "raw_daylight": values["raw"]["daylight"],
                "conformal_daylight": values["conformal"]["daylight"],
                "seconds": float(sum(values["timings"].values())),
            }
            for name, values in results["models"].items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
