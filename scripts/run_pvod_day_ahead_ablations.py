"""Run controlled NWP ablations and stratified diagnostics for PVOD day-ahead QGBM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.50, 0.95)


def metrics(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray) -> dict[str, float]:
    y = frame["target_power_normalized"].to_numpy()
    result = {"n": int(len(frame)), "mae_median": float(np.mean(np.abs(y - median))), "coverage_90": float(np.mean((y >= low) & (y <= high))), "width_90": float(np.mean(high - low))}
    for quantile, pred in ((0.05, low), (0.50, median), (0.95, high)):
        result[f"pinball_{quantile:.2f}"] = float(np.mean(np.maximum(quantile * (y - pred), (quantile - 1.0) * (y - pred))))
    return result


def subset_metrics(frame: pd.DataFrame, low: np.ndarray, median: np.ndarray, high: np.ndarray, column: str, values: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for value in values:
        mask = frame[column].astype(str).to_numpy() == value
        result[value] = metrics(frame.loc[mask], low[mask], median[mask], high[mask])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_day_ahead_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["daylight"] = np.where(frame["clearsky_ghi"] >= 20.0, "daylight", "night")
    frame["lead_band"] = pd.cut(frame["lead_15min"] / 4.0, bins=[27.99, 36.0, 44.0, 52.0], labels=["28-36h", "36-44h", "44-52h"], include_lowest=True).astype(str)
    frame["daylight_lead_band"] = frame["daylight"] + "_" + frame["lead_band"]
    split = design["temporal_split_target_utc"]
    train = frame.loc[frame["target_timestamp_utc"] <= pd.Timestamp(split["train_end"])]
    calibration = frame.loc[(frame["target_timestamp_utc"] >= pd.Timestamp(split["calibration_start"])) & (frame["target_timestamp_utc"] <= pd.Timestamp(split["calibration_end"]))]
    test = frame.loc[frame["target_timestamp_utc"] >= pd.Timestamp(split["test_start"])]
    settings = {
        "issue_time_history": ["station_code", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_zenith", "clearsky_ghi"] + design["origin_features"],
        "issue_time_history_plus_nwp": ["station_code"] + design["future_features"] + design["origin_features"],
    }
    result: dict[str, object] = {"study": design["name"], "restriction": design["restriction"], "settings": {}}
    stations = sorted(frame["station"].astype(str).unique())
    for name, features in settings.items():
        fit = train.dropna(subset=features + ["target_power_normalized"])
        cal = calibration.dropna(subset=features + ["target_power_normalized"])
        held = test.dropna(subset=features + ["target_power_normalized"])
        predictions: dict[str, dict[float, np.ndarray]] = {"cal": {}, "test": {}}
        for quantile in QUANTILES:
            model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=20260826)
            model.fit(fit[features], fit["target_power_normalized"], categorical_feature=["station_code"])
            predictions["cal"][quantile] = np.clip(model.predict(cal[features]), 0.0, 1.2)
            predictions["test"][quantile] = np.clip(model.predict(held[features]), 0.0, 1.2)
        y_cal = cal["target_power_normalized"].to_numpy()
        radius = float(np.quantile(np.maximum.reduce([predictions["cal"][0.05] - y_cal, y_cal - predictions["cal"][0.95], np.zeros(len(cal))]), 0.9, method="higher"))
        low, median, high = predictions["test"][0.05], predictions["test"][0.50], predictions["test"][0.95]
        result["settings"][name] = {
            "n_train": int(len(fit)), "n_calibration": int(len(cal)), "n_test": int(len(held)), "features": features, "conformal_radius": radius,
            "raw_overall": metrics(held, low, median, high),
            "calibrated_overall": metrics(held, np.clip(low - radius, 0.0, None), median, high + radius),
            "raw_by_station": subset_metrics(held, low, median, high, "station", stations),
            "raw_by_daylight": subset_metrics(held, low, median, high, "daylight", ["daylight", "night"]),
            "raw_by_lead_band": subset_metrics(held, low, median, high, "lead_band", ["28-36h", "36-44h", "44-52h"]),
            "raw_by_daylight_lead_band": subset_metrics(held, low, median, high, "daylight_lead_band", ["daylight_28-36h", "daylight_36-44h", "daylight_44-52h"]),
        }
    destination = root / "results" / "pvod_day_ahead_ablations.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: values["raw_overall"] for name, values in result["settings"].items()}, indent=2))


if __name__ == "__main__":
    main()
