"""Run a matched XGBoost point benchmark on the frozen PVOD day-ahead split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]


def metrics(frame, prediction):
    target = frame.target_power_normalized.to_numpy(); daylight = frame.clearsky_ghi.to_numpy() >= 20
    return {"mae_all": float(np.abs(target-prediction).mean()), "mae_daylight": float(np.abs(target[daylight]-prediction[daylight]).mean()), "n": int(len(frame)), "n_daylight": int(daylight.sum())}


def main() -> None:
    design = json.loads((ROOT/"configs"/"pvod_day_ahead_study.json").read_text()); frame = pd.read_parquet(ROOT/"data"/"processed"/"pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame.target_timestamp_utc, utc=True); split = design["temporal_split_target_utc"]; features = ["station_code", *design["future_features"], *design["origin_features"]]
    partitions = {"train": frame.loc[frame.target_timestamp_utc <= pd.Timestamp(split["train_end"])], "validation": frame.loc[(frame.target_timestamp_utc >= pd.Timestamp(split["calibration_start"])) & (frame.target_timestamp_utc <= pd.Timestamp(split["calibration_end"]))], "test": frame.loc[frame.target_timestamp_utc >= pd.Timestamp(split["test_start"])]}
    partitions = {name: value.dropna(subset=features+["target_power_normalized"]).copy() for name, value in partitions.items()}
    for value in partitions.values(): value["station_code"] = value.station_code.astype("category")
    candidates = [{"max_depth": 7, "min_child_weight": 20}, {"max_depth": 9, "min_child_weight": 50}]; trials=[]; best=None; best_mae=float("inf")
    for params in candidates:
        model = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=3000, learning_rate=.03, subsample=.9, colsample_bytree=.9, reg_lambda=2., tree_method="hist", device="cuda", enable_categorical=True, early_stopping_rounds=100, random_state=20260828, **params)
        model.fit(partitions["train"][features], partitions["train"].target_power_normalized, eval_set=[(partitions["validation"][features], partitions["validation"].target_power_normalized)], verbose=False)
        prediction = np.clip(model.predict(partitions["validation"][features]), 0, 1.2); value = metrics(partitions["validation"], prediction)["mae_all"]; trials.append({**params, "best_iteration": int(model.best_iteration), "validation_mae": value})
        if value < best_mae: best, best_mae = model, value
    prediction = np.clip(best.predict(partitions["test"][features]), 0, 1.2); result = {"study": design["name"], "restriction": design["restriction"], "method": "XGBoost point forecast with the QGBM day-ahead feature contract", "trials": trials, "test": metrics(partitions["test"], prediction)}
    output = partitions["test"][["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min", "clearsky_ghi", "target_power_normalized"]].reset_index(drop=True)
    output["xgboost_prediction"] = prediction
    output.to_parquet(ROOT/"results"/"pvod_xgboost_day_ahead_predictions.parquet", index=False)
    (ROOT/"results"/"pvod_xgboost_day_ahead.json").write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
