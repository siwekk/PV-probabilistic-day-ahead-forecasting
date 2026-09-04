"""Fit and calibrate source-supported day-ahead PVOD QGBM forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.50, 0.95)


def evaluate(y: np.ndarray, pred: dict[float, np.ndarray]) -> dict[str, float]:
    low, mid, high = pred[0.05], pred[0.50], pred[0.95]
    return {"mae_median": float(np.mean(np.abs(y - mid))), "coverage_90": float(np.mean((y >= low) & (y <= high))), "width_90": float(np.mean(high - low))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_day_ahead_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    split = design["temporal_split_target_utc"]
    train = frame.loc[frame["target_timestamp_utc"] <= pd.Timestamp(split["train_end"])]
    calibration = frame.loc[(frame["target_timestamp_utc"] >= pd.Timestamp(split["calibration_start"])) & (frame["target_timestamp_utc"] <= pd.Timestamp(split["calibration_end"]))]
    test = frame.loc[frame["target_timestamp_utc"] >= pd.Timestamp(split["test_start"])]
    features = ["station_code"] + design["future_features"] + design["origin_features"]
    target = "target_power_normalized"
    fit = train.dropna(subset=features + [target]); cal = calibration.dropna(subset=features + [target]); held_out = test.dropna(subset=features + [target])
    predictions: dict[str, dict[float, np.ndarray]] = {"calibration": {}, "test": {}}
    for quantile in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=20260826)
        model.fit(fit[features], fit[target], categorical_feature=["station_code"])
        predictions["calibration"][quantile] = np.clip(model.predict(cal[features]), 0.0, 1.2)
        predictions["test"][quantile] = np.clip(model.predict(held_out[features]), 0.0, 1.2)
    y_cal = cal[target].to_numpy(); y_test = held_out[target].to_numpy()
    nonconformity = np.maximum.reduce([predictions["calibration"][0.05] - y_cal, y_cal - predictions["calibration"][0.95], np.zeros(len(cal))])
    radius = float(np.quantile(nonconformity, 0.9, method="higher"))
    raw = evaluate(y_test, predictions["test"])
    calibrated = dict(predictions["test"])
    calibrated[0.05] = np.clip(calibrated[0.05] - radius, 0.0, None); calibrated[0.95] = calibrated[0.95] + radius
    result = {"study": design["name"], "restriction": design["restriction"], "n_train": int(len(fit)), "n_calibration": int(len(cal)), "n_test": int(len(held_out)), "conformal_radius": radius, "raw_test": raw, "calibrated_test": evaluate(y_test, calibrated), "features": features}
    destination = root / "results" / "pvod_day_ahead_qgbm.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
