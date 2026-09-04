#!/usr/bin/env python3
"""Run the controlled physics ladder on measured PVOD power."""

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


QUANTILES = (0.05, 0.50, 0.95)


def add_trajectory_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["station", "issue_timestamp_utc", "lead_15min"]).copy()
    keys = ["station", "issue_timestamp_utc"]
    grouped = frame.groupby(keys, observed=True)["nwp_globalirrad"]
    frame["nwp_ghi_trajectory_mean"] = grouped.transform("mean")
    frame["nwp_ghi_trajectory_max"] = grouped.transform("max")
    frame["nwp_ghi_trajectory_std"] = grouped.transform("std").fillna(0.0)
    frame["nwp_ghi_trajectory_ramp"] = grouped.diff().fillna(0.0)
    frame["nwp_ghi_trajectory_abs_ramp"] = frame["nwp_ghi_trajectory_ramp"].abs()
    frame["nwp_ghi_trajectory_max_abs_ramp"] = frame.groupby(keys, observed=True)[
        "nwp_ghi_trajectory_abs_ramp"
    ].transform("max")
    peak_source = frame.assign(_nwp_ghi_for_peak=frame["nwp_globalirrad"].fillna(-np.inf))
    peak_indices = peak_source.groupby(keys, observed=True)["_nwp_ghi_for_peak"].idxmax()
    peak = frame.loc[peak_indices, keys + ["lead_15min"]].rename(
        columns={"lead_15min": "nwp_ghi_trajectory_peak_lead"}
    )
    return frame.merge(peak, on=keys, how="left", validate="many_to_one")


def specifications() -> dict[str, dict[str, Any]]:
    nonphysical = [
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
    ]
    geometry = nonphysical + ["solar_zenith"]
    clear_sky = geometry + ["clearsky_ghi", "clearsky_scaled_persistence_72h"]
    trajectory = clear_sky + [
        "nwp_ghi_trajectory_mean",
        "nwp_ghi_trajectory_max",
        "nwp_ghi_trajectory_std",
        "nwp_ghi_trajectory_ramp",
        "nwp_ghi_trajectory_abs_ramp",
        "nwp_ghi_trajectory_max_abs_ramp",
        "nwp_ghi_trajectory_peak_lead",
    ]
    return {
        "M1_nonphysical_nwp_direct": {"features": nonphysical, "target": "direct", "clip": False},
        "M2_solar_geometry_direct": {"features": geometry, "target": "direct", "clip": False},
        "M3_clear_sky_direct": {"features": clear_sky, "target": "direct", "clip": False},
        "M4_physical_residual": {"features": clear_sky, "target": "residual", "clip": True},
        "M5_physical_residual_trajectory": {"features": trajectory, "target": "residual", "clip": True},
    }


def evaluate(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict[str, Any]:
    daylight = frame["daylight"].to_numpy(dtype=bool)
    result: dict[str, Any] = {
        "overall": score(frame, low, median, high),
        "daylight": score(frame.loc[daylight], low[daylight], median[daylight], high[daylight]),
        "by_station": {},
        "by_lead_band": {},
    }
    for station in sorted(frame["station"].astype(str).unique()):
        mask = (frame["station"].astype(str).to_numpy() == station) & daylight
        result["by_station"][station] = score(frame.loc[mask], low[mask], median[mask], high[mask])
    for band in sorted(frame["lead_band"].unique()):
        mask = (frame["lead_band"].to_numpy() == band) & daylight
        result["by_lead_band"][band] = score(frame.loc[mask], low[mask], median[mask], high[mask])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_q1_physics_study.json").read_text(encoding="utf-8"))
    frame = add_trajectory_features(
        pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    )
    frame["station_code"] = frame["station"].astype("category")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["issue_timestamp_utc"] = pd.to_datetime(frame["issue_timestamp_utc"], utc=True)
    frame["daylight"] = frame["clearsky_ghi"] >= float(design["daylight_clearsky_ghi_w_m2"])
    frame["lead_band"] = pd.cut(
        frame["lead_15min"] / 4.0,
        bins=[27.99, 36.0, 44.0, 52.0],
        labels=["28-36h", "36-44h", "44-52h"],
        include_lowest=True,
    ).astype(str)
    specifications_map = specifications()
    common_features = sorted(
        {feature for specification in specifications_map.values() for feature in specification["features"]}
    )
    rows_before_common_filter = len(frame)
    frame = frame.dropna(subset=common_features + ["target_power_normalized"]).copy()
    split = design["split"]
    timestamp = frame["target_timestamp_utc"]
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
    baseline_column = "clearsky_scaled_persistence_72h"
    frame_result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(design["seed"]),
        "restriction": design["restriction"],
        "split": split,
        "rows": {
            "before_common_complete_case_filter": int(rows_before_common_filter),
            "after_common_complete_case_filter": int(len(frame)),
            "train": int(len(train)),
            "tuning": int(len(tuning)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "models": {},
    }
    predictions_output = test[
        ["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min", "daylight", "target_power_normalized"]
    ].reset_index(drop=True)

    for name, specification in specifications_map.items():
        features = specification["features"]
        required = features + ["target_power_normalized"]
        fit = train.dropna(subset=required).copy()
        tune = tuning.dropna(subset=required).copy()
        calibrate = calibration.dropna(subset=required).copy()
        held = test.dropna(subset=required).copy()
        if len(held) != len(test):
            raise ValueError(f"Model {name} would evaluate a different test sample")
        if specification["target"] == "residual":
            for part in (fit, tune, calibrate, held):
                part["physical_residual"] = part["target_power_normalized"] - part[baseline_column]
            target_column = "physical_residual"
        else:
            target_column = "target_power_normalized"
        model_predictions: dict[float, dict[str, np.ndarray]] = {}
        best_iterations = {}
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
                random_state=int(design["seed"]),
            )
            model.fit(
                fit[features],
                fit[target_column],
                categorical_feature=["station_code"],
                eval_set=[(tune[features], tune[target_column])],
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            best_iteration = int(model.best_iteration_ or model.n_estimators)
            development = pd.concat([fit, tune], ignore_index=True)
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
                random_state=int(design["seed"]),
            )
            final.fit(
                development[features], development[target_column], categorical_feature=["station_code"]
            )
            cal_prediction = final.predict(calibrate[features])
            test_prediction = final.predict(held[features])
            if specification["target"] == "residual":
                cal_prediction += calibrate[baseline_column].to_numpy()
                test_prediction += held[baseline_column].to_numpy()
            if specification["clip"]:
                cal_prediction = np.clip(cal_prediction, 0.0, 1.2)
                test_prediction = np.clip(test_prediction, 0.0, 1.2)
            model_predictions[quantile] = {
                "calibration": cal_prediction,
                "test": test_prediction,
            }
            best_iterations[f"quantile_{quantile:.2f}"] = best_iteration
            timings[f"quantile_{quantile:.2f}_seconds"] = float(time.perf_counter() - started)

        calibration_ordered = np.sort(
            np.column_stack([model_predictions[q]["calibration"] for q in QUANTILES]), axis=1
        )
        test_ordered = np.sort(
            np.column_stack([model_predictions[q]["test"] for q in QUANTILES]), axis=1
        )
        calibration_daylight = calibrate["daylight"].to_numpy(dtype=bool)
        radius = conformal_radius(
            calibrate.loc[calibration_daylight],
            calibration_ordered[calibration_daylight, 0],
            calibration_ordered[calibration_daylight, 2],
        )
        low, median, high = test_ordered[:, 0], test_ordered[:, 1], test_ordered[:, 2]
        test_daylight = held["daylight"].to_numpy(dtype=bool)
        calibrated_low = low.copy()
        calibrated_high = high.copy()
        calibrated_low[test_daylight] = np.clip(low[test_daylight] - radius, 0.0, 1.2)
        calibrated_high[test_daylight] = np.clip(high[test_daylight] + radius, 0.0, 1.2)
        frame_result["models"][name] = {
            "features": features,
            "target_form": specification["target"],
            "best_iterations": best_iterations,
            "timings": timings,
            "conformal_radius": radius,
            "raw": evaluate(held, low, median, high),
            "conformal": evaluate(held, calibrated_low, median, calibrated_high),
        }
        prefix = name.split("_")[0]
        predictions_output[f"{prefix}_q05"] = low
        predictions_output[f"{prefix}_q50"] = median
        predictions_output[f"{prefix}_q95"] = high

    result_path = root / "results" / "pvod_q1_physics_ladder.json"
    prediction_path = root / "results" / "pvod_q1_physics_ladder_predictions.parquet"
    result_path.write_text(json.dumps(frame_result, indent=2) + "\n", encoding="ascii")
    predictions_output.to_parquet(prediction_path, index=False)
    print(
        json.dumps(
            {
                name: {
                    "raw_daylight": values["raw"]["daylight"],
                    "conformal_daylight": values["conformal"]["daylight"],
                }
                for name, values in frame_result["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
