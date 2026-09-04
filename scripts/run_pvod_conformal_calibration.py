"""Calibrate global PVOD QGBM intervals using the frozen chronological calibration block."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.50, 0.95)


def interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict[str, float]:
    return {"coverage_90": float(np.mean((y >= low) & (y <= high))), "width_90": float(np.mean(high - low))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_aligned_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_15min_features.parquet")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["station_code"] = frame["station"].map({name: i for i, name in enumerate(sorted(frame["station"].unique()))}).astype("category")
    split = design["temporal_split_utc"]
    train = frame.loc[frame["timestamp_utc"] <= pd.Timestamp(split["train_end"])]
    calibration = frame.loc[(frame["timestamp_utc"] >= pd.Timestamp(split["calibration_start"])) & (frame["timestamp_utc"] <= pd.Timestamp(split["calibration_end"]))]
    test = frame.loc[frame["timestamp_utc"] >= pd.Timestamp(split["test_start"])]
    output: dict[str, object] = {"restriction": design["restriction"], "settings": {}}

    for setting in ("aligned_nwp", "aligned_nwp_with_power_history", "aligned_nwp_with_full_history"):
        features = ["station_code"] + [feature for group in design["information_sets"][setting] for feature in design["feature_sets"][group]]
        fit = train.dropna(subset=features + ["power_normalized"])
        cal = calibration.dropna(subset=features + ["power_normalized"])
        evaluate = test.dropna(subset=features + ["power_normalized"])
        prediction: dict[str, dict[float, np.ndarray]] = {"calibration": {}, "test": {}}
        for quantile in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=20260826)
            model.fit(fit[features], fit["power_normalized"], categorical_feature=["station_code"])
            prediction["calibration"][quantile] = np.clip(model.predict(cal[features]), 0.0, 1.2)
            prediction["test"][quantile] = np.clip(model.predict(evaluate[features]), 0.0, 1.2)
        y_cal = cal["power_normalized"].to_numpy(); y_test = evaluate["power_normalized"].to_numpy()
        low_cal, high_cal = prediction["calibration"][0.05], prediction["calibration"][0.95]
        nonconformity = np.maximum.reduce([low_cal - y_cal, y_cal - high_cal, np.zeros(len(cal))])
        radius = float(np.quantile(nonconformity, 0.9, method="higher"))
        low_test, high_test = prediction["test"][0.05], prediction["test"][0.95]
        output["settings"][setting] = {"n_train": int(len(fit)), "n_calibration": int(len(cal)), "n_test": int(len(evaluate)), "radius": radius, "raw_test": interval_score(y_test, low_test, high_test), "calibrated_test": interval_score(y_test, np.clip(low_test - radius, 0.0, None), high_test + radius)}

    target = root / "results" / "pvod_aligned_qgbm_calibration.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
