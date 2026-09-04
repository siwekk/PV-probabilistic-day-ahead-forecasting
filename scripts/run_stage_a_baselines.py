"""Fit first one-hour-ahead QGBM baselines using the frozen Stage A manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


QUANTILES = (0.05, 0.50, 0.95)
FEATURES = ["lag1", "lag24", "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos", "solar_zenith_deg", "clear_sky_ghi_w_m2"]


def prepare(frame: pd.DataFrame, power_kw: str) -> pd.DataFrame:
    frame = frame.copy().sort_values(["site_id", "timestamp"])
    frame["power_norm"] = frame[power_kw] / frame["rated_power_kw"]
    for target in ("power_kw", "power_norm"):
        frame[f"{target}_lag1"] = frame.groupby("site_id")[target].shift(1)
        frame[f"{target}_lag24"] = frame.groupby("site_id")[target].shift(24)
    frame["clear_sky_ghi_lag24"] = frame.groupby("site_id")["clear_sky_ghi_w_m2"].shift(24)
    for lag in (1, 24):
        frame[f"lag{lag}"] = frame[f"power_norm_lag{lag}"]
    return frame


def scores(y: np.ndarray, predictions: dict[float, np.ndarray]) -> dict[str, float]:
    result = {}
    for q, pred in predictions.items():
        result[f"pinball_{q:.2f}"] = float(np.mean(np.maximum(q * (y - pred), (q - 1) * (y - pred))))
    result["mae_median"] = float(np.mean(np.abs(y - predictions[0.50])))
    result["coverage_90"] = float(np.mean((y >= predictions[0.05]) & (y <= predictions[0.95])))
    result["width_90"] = float(np.mean(predictions[0.95] - predictions[0.05]))
    return result


def fit_quantiles(train: pd.DataFrame, target: str) -> dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.9, n_jobs=32, verbosity=-1)
        model.fit(train[FEATURES], train[target])
        models[q] = model
    return models


def predict(models: dict[float, lgb.LGBMRegressor], frame: pd.DataFrame) -> dict[float, np.ndarray]:
    return {q: model.predict(frame[FEATURES]) for q, model in models.items()}


def evaluate_setting(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict[str, dict[str, float]]:
    train = train.dropna(subset=FEATURES + [target])
    test = test.dropna(subset=FEATURES + [target])
    models = fit_quantiles(train, target)
    qgbm = scores(test[target].to_numpy(), predict(models, test))
    persistence = test[f"{target}_lag1"].to_numpy()
    smart = test[f"{target}_lag24"].to_numpy() * np.divide(
        test["clear_sky_ghi_w_m2"].to_numpy(), test["clear_sky_ghi_lag24"].to_numpy(),
        out=np.ones(len(test)), where=test["clear_sky_ghi_lag24"].to_numpy() > 20,
    )
    return {
        "qgbm": qgbm,
        "persistence_mae": float(np.mean(np.abs(test[target] - persistence))),
        "smart_persistence_mae": float(np.mean(np.abs(test[target] - smart))),
        "n_train": int(len(train)), "n_test": int(len(test)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(); project = args.project_root.resolve(); processed = project / "data" / "processed"
    manifest = json.loads((processed / "stage_a_split_manifest.json").read_text()); temporal = manifest["config"]["temporal"]
    hk = pd.read_parquet(processed / "stage_a_hkust_features.parquet"); hk["timestamp"] = pd.to_datetime(hk["timestamp"]); hk["power_kw"] = hk["power_w_capacity_qc"] / 1000; hk = prepare(hk, "power_kw")
    so = pd.read_parquet(processed / "stage_a_solete_features.parquet"); so["timestamp"] = pd.to_datetime(so["timestamp"]); so = prepare(so, "power_kw")
    seen = manifest["seen_training_sites"]; unseen = manifest["unseen_test_sites"]
    train = hk[(hk.site_id.isin(seen)) & (hk.timestamp <= temporal["train_end"])]
    temporal_test = hk[(hk.site_id.isin(seen)) & (hk.timestamp >= temporal["test_start"])]
    unseen_test = hk[(hk.site_id.isin(unseen)) & (hk.timestamp >= temporal["test_start"])]
    result = {"normalized_qgbm": {"temporal": evaluate_setting(train, temporal_test, "power_norm"), "unseen_site": evaluate_setting(train, unseen_test, "power_norm"), "external_solete": evaluate_setting(train, so, "power_norm")}, "unnormalized_qgbm": {"temporal": evaluate_setting(train, temporal_test, "power_kw"), "unseen_site": evaluate_setting(train, unseen_test, "power_kw"), "external_solete": evaluate_setting(train, so, "power_kw")}}
    output = project / "results" / "stage_a_raw_baselines.json"; output.parent.mkdir(exist_ok=True); output.write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
