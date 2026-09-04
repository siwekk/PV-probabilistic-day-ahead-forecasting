#!/usr/bin/env python3
"""Run complete-state holdouts for non-physical and physics-informed models."""

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
    nonphysical = [
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
    ]
    physical = nonphysical + [
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
        "nonphysical_direct": {"features": nonphysical, "target": "target_power_normalized", "residual": False},
        "physics_residual_trajectory": {"features": physical, "target": "source_residual_normalized", "residual": True},
    }


def evaluate_strata(
    held: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray
) -> dict[str, Any]:
    daylight = held["daylight"].to_numpy(dtype=bool)
    results: dict[str, Any] = {
        "overall": score(held, low, median, high),
        "daylight": score(held.loc[daylight], low[daylight], median[daylight], high[daylight]),
        "night": score(held.loc[~daylight], low[~daylight], median[~daylight], high[~daylight]),
        "by_plant_type": {},
    }
    for plant_type in sorted(held["plant_type"].astype(str).unique()):
        mask = held["plant_type"].astype(str).to_numpy() == plant_type
        results["by_plant_type"][plant_type] = score(
            held.loc[mask], low[mask], median[mask], high[mask]
        )
    return results


def season_label(frame: pd.DataFrame) -> np.ndarray:
    month = pd.to_datetime(frame["LocalTime"]).dt.month
    return np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["winter", "spring", "summer"], default="autumn",
    )


def elevation_label(frame: pd.DataFrame) -> np.ndarray:
    elevation = 90.0 - frame["solar_zenith_deg"].to_numpy(float)
    return np.select([elevation < 15, elevation < 35], ["low", "medium"], default="high")


def finite_radius(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    nonconformity = np.maximum.reduce([low - y, y - high, np.zeros(len(y))])
    level = min(1.0, np.ceil((len(y) + 1) * 0.90) / len(y))
    return float(np.quantile(nonconformity, level, method="higher"))


def apply_radius(low: np.ndarray, high: np.ndarray, radius: np.ndarray, daylight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    adjusted_low, adjusted_high = low.copy(), high.copy()
    adjusted_low[daylight] = np.clip(low[daylight] - radius[daylight], 0.0, 1.2)
    adjusted_high[daylight] = np.clip(high[daylight] + radius[daylight], 0.0, 1.2)
    return adjusted_low, adjusted_high


def calibration_metrics(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    work = frame.copy()
    work["season_group"] = season_label(work)
    work["elevation_group"] = elevation_label(work)
    output: dict[str, Any] = {}
    for grouping in ["overall", "season_group", "elevation_group"]:
        groups = [("all", work)] if grouping == "overall" else work.groupby(grouping, observed=True)
        values = {}
        for label, group in groups:
            idx = group.index.to_numpy()
            y = group["target_power_normalized"].to_numpy(float)
            lo, med, hi = low[idx], median[idx], high[idx]
            is90 = hi - lo + 20 * (lo - y) * (y < lo) + 20 * (y - hi) * (y > hi)
            wis90 = (0.5 * np.abs(y - med) + 0.05 * is90) / 1.5
            values[str(label)] = {
                "n": int(len(group)), "coverage_90": float(np.mean((y >= lo) & (y <= hi))),
                "width_90": float(np.mean(hi - lo)), "interval_score_90": float(np.mean(is90)),
                "wis_90_single_interval": float(np.mean(wis90)),
            }
        output[grouping] = values
    return output


def conditional_calibration(
    calibration: pd.DataFrame, held: pd.DataFrame, cal_low: np.ndarray, cal_high: np.ndarray,
    test_low: np.ndarray, median: np.ndarray, test_high: np.ndarray,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    cal_day = calibration["daylight"].to_numpy(bool)
    test_day = held["daylight"].to_numpy(bool)
    cal_y = calibration["target_power_normalized"].to_numpy(float)
    global_radius = finite_radius(cal_y[cal_day], cal_low[cal_day], cal_high[cal_day])
    methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    methods["global"] = apply_radius(test_low, test_high, np.full(len(held), global_radius), test_day)
    cal_groups = {
        "season": season_label(calibration),
        "solar_elevation": elevation_label(calibration),
        "season_solar": np.char.add(np.char.add(season_label(calibration).astype(str), "_"), elevation_label(calibration).astype(str)),
    }
    test_groups = {
        "season": season_label(held),
        "solar_elevation": elevation_label(held),
        "season_solar": np.char.add(np.char.add(season_label(held).astype(str), "_"), elevation_label(held).astype(str)),
    }
    for name in cal_groups:
        radii = np.full(len(held), global_radius)
        for label in np.unique(test_groups[name]):
            mask = cal_day & (cal_groups[name] == label)
            if np.sum(mask) >= 200:
                radii[test_groups[name] == label] = finite_radius(cal_y[mask], cal_low[mask], cal_high[mask])
        methods[name] = apply_radius(test_low, test_high, radii, test_day)
    rolling_radius = np.full(len(held), global_radius)
    dates = pd.to_datetime(held["delivery_date"])
    order = np.argsort(dates.to_numpy())
    unique_dates = np.sort(dates.unique())
    test_y = held["target_power_normalized"].to_numpy(float)
    for date in unique_dates:
        current = dates == date
        history = (dates < date) & (dates >= date - pd.Timedelta(days=28)) & test_day
        if np.sum(history) >= 500:
            rolling_radius[current] = finite_radius(test_y[history], test_low[history], test_high[history])
    methods["rolling_28d"] = apply_radius(test_low, test_high, rolling_radius, test_day)
    daylight_frame = held.loc[test_day].reset_index(drop=True)
    results = {}
    for name, (low, high) in methods.items():
        results[name] = calibration_metrics(daylight_frame, low[test_day], median[test_day], high[test_day])
    return results, methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(root / "data" / "processed" / "nrel_q1_panel.parquet")
    frame["plant_type"] = frame["plant_type"].astype("category")
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "protocol": (
            "For each rotation, fit on development-state training sites, tune on separate "
            "development-state calibration sites, calibrate intervals on remaining development-state "
            "test sites, and evaluate once on every site in the held-out state."
        ),
        "rotations": {},
    }
    prediction_parts = []

    for rotation in design["state_holdout_rotations"]:
        test_state = rotation["test_state"]
        development_states = rotation["development_states"]
        development = frame.loc[frame["state"].isin(development_states)]
        fit = development.loc[development["site_split"] == "train"]
        tuning = development.loc[development["site_split"] == "calibration"]
        calibration = development.loc[development["site_split"] == "test"]
        held = frame.loc[frame["state"] == test_state].copy()
        source = held["source_forecast_normalized"].to_numpy(dtype=float)
        y = held["target_power_normalized"].to_numpy(dtype=float)
        rotation_result: dict[str, Any] = {
            "development_states": development_states,
            "test_state": test_state,
            "rows": {
                "fit": int(len(fit)),
                "tuning": int(len(tuning)),
                "calibration": int(len(calibration)),
                "test": int(len(held)),
            },
            "test_sites": int(held["site_id"].nunique()),
            "source_forecast": {
                "overall_mae": float(np.mean(np.abs(y - source))),
                "daylight_mae": float(np.mean(np.abs(y[held["daylight"]] - source[held["daylight"]]))),
            },
            "models": {},
        }
        predictions = held[
            ["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized", "source_forecast_normalized"]
        ].reset_index(drop=True)

        for model_name, specification in specifications().items():
            features = specification["features"]
            target = specification["target"]
            quantile_predictions: dict[float, dict[str, np.ndarray]] = {}
            best_iterations = {}
            timings = {}
            for quantile in QUANTILES:
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
                started = time.perf_counter()
                model.fit(
                    fit[features],
                    fit[target],
                    categorical_feature=["plant_type"],
                    eval_set=[(tuning[features], tuning[target])],
                    callbacks=[lgb.early_stopping(80, verbose=False)],
                )
                best_iteration = int(model.best_iteration_ or model.n_estimators)
                final_model = lgb.LGBMRegressor(
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
                final_model.fit(
                    pd.concat([fit[features], tuning[features]], ignore_index=True),
                    pd.concat([fit[target], tuning[target]], ignore_index=True),
                    categorical_feature=["plant_type"],
                )
                cal_prediction = final_model.predict(calibration[features])
                test_prediction = final_model.predict(held[features])
                if specification["residual"]:
                    cal_prediction += calibration["source_forecast_normalized"].to_numpy()
                    test_prediction += source
                    cal_prediction = np.clip(cal_prediction, 0.0, 1.2)
                    test_prediction = np.clip(test_prediction, 0.0, 1.2)
                quantile_predictions[quantile] = {
                    "calibration": cal_prediction,
                    "test": test_prediction,
                }
                best_iterations[f"quantile_{quantile:.2f}"] = best_iteration
                timings[f"quantile_{quantile:.2f}_seconds"] = float(time.perf_counter() - started)

            test_ordered = np.sort(
                np.column_stack([quantile_predictions[q]["test"] for q in QUANTILES]), axis=1
            )
            calibration_ordered = np.sort(
                np.column_stack([quantile_predictions[q]["calibration"] for q in QUANTILES]), axis=1
            )
            calibration_daylight = calibration["daylight"].to_numpy(dtype=bool)
            radius = conformal_radius(
                calibration.loc[calibration_daylight],
                calibration_ordered[calibration_daylight, 0],
                calibration_ordered[calibration_daylight, 2],
            )
            low, median, high = test_ordered[:, 0], test_ordered[:, 1], test_ordered[:, 2]
            held_daylight = held["daylight"].to_numpy(dtype=bool)
            calibrated_low = low.copy()
            calibrated_high = high.copy()
            calibrated_low[held_daylight] = np.clip(low[held_daylight] - radius, 0.0, 1.2)
            calibrated_high[held_daylight] = np.clip(high[held_daylight] + radius, 0.0, 1.2)
            conditional_results, conditional_predictions = conditional_calibration(
                calibration.reset_index(drop=True), held.reset_index(drop=True),
                calibration_ordered[:, 0], calibration_ordered[:, 2],
                low, median, high,
            )
            rotation_result["models"][model_name] = {
                "features": features,
                "best_iterations": best_iterations,
                "timings": timings,
                "conformal_scope": "development-state daylight rows, applied to held-state daylight rows",
                "conformal_radius": radius,
                "raw": evaluate_strata(held, low, median, high),
                "conformal": evaluate_strata(held, calibrated_low, median, calibrated_high),
                "conditional_calibration": conditional_results,
            }
            prefix = "nonphysics" if model_name.startswith("nonphysical") else "physics"
            predictions[f"{prefix}_q05"] = low
            predictions[f"{prefix}_q50"] = median
            predictions[f"{prefix}_q95"] = high
            predictions[f"{prefix}_q05_conformal"] = calibrated_low
            predictions[f"{prefix}_q95_conformal"] = calibrated_high
            if prefix == "physics":
                for method, (method_low, method_high) in conditional_predictions.items():
                    predictions[f"physics_q05_{method}"] = method_low
                    predictions[f"physics_q95_{method}"] = method_high

        result["rotations"][test_state] = rotation_result
        prediction_parts.append(predictions)

    result_path = root / "results" / "nrel_state_holdout.json"
    prediction_path = root / "results" / "nrel_state_holdout_predictions.parquet"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    pd.concat(prediction_parts, ignore_index=True).to_parquet(prediction_path, index=False)
    compact = {
        state: {
            "source": values["source_forecast"],
            "nonphysical_daylight": values["models"]["nonphysical_direct"]["raw"]["daylight"],
            "physics_daylight": values["models"]["physics_residual_trajectory"]["raw"]["daylight"],
        }
        for state, values in result["rotations"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
