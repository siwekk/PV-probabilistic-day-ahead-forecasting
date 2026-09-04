"""Run the first global QGBM comparison on the frozen PVOD aligned-covariate split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.50, 0.95)


def score(y: np.ndarray, pred: dict[float, np.ndarray]) -> dict[str, float]:
    low, median, high = pred[0.05], pred[0.50], pred[0.95]
    result = {
        "mae_median": float(np.mean(np.abs(y - median))),
        "coverage_90": float(np.mean((y >= low) & (y <= high))),
        "width_90": float(np.mean(high - low)),
    }
    for quantile, values in pred.items():
        result[f"pinball_{quantile:.2f}"] = float(np.mean(np.maximum(quantile * (y - values), (quantile - 1.0) * (y - values))))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_aligned_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_15min_features.parquet")
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    station_map = {name: i for i, name in enumerate(sorted(frame["station"].unique()))}
    frame["station_code"] = frame["station"].map(station_map).astype("category")
    split = design["temporal_split_utc"]
    train = frame.loc[frame["timestamp_utc"] <= pd.Timestamp(split["train_end"])].copy()
    test = frame.loc[frame["timestamp_utc"] >= pd.Timestamp(split["test_start"])].copy()
    target = "power_normalized"
    results: dict[str, object] = {"study": design["name"], "restriction": design["restriction"], "station_code_map": station_map, "settings": {}}

    for setting, groups in design["information_sets"].items():
        features = ["station_code"]
        for group in groups:
            features.extend(design["feature_sets"][group])
        model_train = train.dropna(subset=features + [target])
        model_test = test.dropna(subset=features + [target])
        predictions: dict[float, np.ndarray] = {}
        for quantile in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=quantile, n_estimators=600, learning_rate=0.05,
                num_leaves=63, min_child_samples=100, colsample_bytree=0.9,
                reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=20260826,
            )
            model.fit(model_train[features], model_train[target], categorical_feature=["station_code"])
            predictions[quantile] = np.clip(model.predict(model_test[features]), 0.0, 1.2)
        results["settings"][setting] = {
            "features": features,
            "n_train": int(len(model_train)),
            "n_test": int(len(model_test)),
            "test": score(model_test[target].to_numpy(), predictions),
        }

    persistence = test.dropna(subset=["power_lag_1", target])
    results["persistence"] = {"n_test": int(len(persistence)), "mae": float(np.mean(np.abs(persistence[target] - persistence["power_lag_1"]))) }
    target_path = root / "results" / "pvod_aligned_qgbm_raw.json"
    target_path.parent.mkdir(exist_ok=True)
    target_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
