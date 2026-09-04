#!/usr/bin/env python3
"""Fit matched nine-quantile LightGBM and XGBoost review benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from review_metrics import QUANTILES, conditional_metrics, diagnostics, metrics, projection_audit, rearrangement_audit
from run_nrel_physics_ladder import model_specifications
from run_pvod_q1_physics_ladder import add_trajectory_features, specifications as pvod_specifications

SEED = 20260829


def finite_radius(y: np.ndarray, low: np.ndarray, high: np.ndarray, nominal: float) -> float:
    score = np.maximum.reduce([low - y, y - high, np.zeros(len(y))])
    level = min(1.0, np.ceil((len(y) + 1) * nominal) / len(y))
    return float(np.quantile(score, level, method="higher"))


def conformalize(y_cal: np.ndarray, cal: np.ndarray, test: np.ndarray, cal_day: np.ndarray, test_day: np.ndarray) -> tuple[np.ndarray, dict]:
    output = test.copy()
    radii = {}
    for alpha, lo, hi in [(0.60, 3, 5), (0.40, 2, 6), (0.20, 1, 7), (0.10, 0, 8)]:
        radius = finite_radius(y_cal[cal_day], cal[cal_day, lo], cal[cal_day, hi], 1.0 - alpha)
        output[test_day, lo] = np.clip(output[test_day, lo] - radius, 0.0, 1.2)
        output[test_day, hi] = np.clip(output[test_day, hi] + radius, 0.0, 1.2)
        radii[str(int((1.0 - alpha) * 100))] = radius
    return np.sort(output, axis=1), radii


def fit_quantiles(kind: str, train: pd.DataFrame, tune: pd.DataFrame, development: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str, categorical: list[str], residual_base: str | None, physical_projection: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cal_columns, test_columns, raw_columns = [], [], []
    best, train_seconds, inference_seconds, model_bytes = {}, {}, {}, {}
    rss_start = psutil.Process(os.getpid()).memory_info().rss
    for q in QUANTILES:
        started = time.perf_counter()
        if kind == "lightgbm":
            model = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=1600, learning_rate=0.04, num_leaves=63, min_child_samples=150, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=SEED)
            model.fit(train[features], train[target], categorical_feature=categorical, eval_set=[(tune[features], tune[target])], callbacks=[lgb.early_stopping(80, verbose=False)])
            n_estimators = int(model.best_iteration_ or model.n_estimators)
            final = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=n_estimators, learning_rate=0.04, num_leaves=63, min_child_samples=150, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=SEED)
            final.fit(development[features], development[target], categorical_feature=categorical)
            model_bytes[str(q)] = len(final.booster_.model_to_string().encode("utf-8"))
        else:
            model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=float(q), n_estimators=1600, learning_rate=0.04, max_depth=8, min_child_weight=50, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", device="cuda", enable_categorical=True, early_stopping_rounds=80, random_state=SEED, n_jobs=32)
            model.fit(train[features], train[target], eval_set=[(tune[features], tune[target])], verbose=False)
            n_estimators = int(model.best_iteration + 1)
            final = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=float(q), n_estimators=n_estimators, learning_rate=0.04, max_depth=8, min_child_weight=50, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", device="cuda", enable_categorical=True, random_state=SEED, n_jobs=32)
            final.fit(development[features], development[target], verbose=False)
            model_bytes[str(q)] = len(final.get_booster().save_raw(raw_format="ubj"))
        train_seconds[str(q)] = time.perf_counter() - started
        best[str(q)] = n_estimators
        inference_started = time.perf_counter()
        test_pred = final.predict(test[features])
        inference_seconds[str(q)] = time.perf_counter() - inference_started
        cal_pred = final.predict(calibration[features])
        if residual_base:
            test_pred = test_pred + test[residual_base].to_numpy(float)
            cal_pred = cal_pred + calibration[residual_base].to_numpy(float)
        raw_columns.append(test_pred.copy())
        if physical_projection:
            test_pred = np.clip(test_pred, 0.0, 1.2)
            cal_pred = np.clip(cal_pred, 0.0, 1.2)
        test_columns.append(test_pred)
        cal_columns.append(cal_pred)
    raw = np.column_stack(raw_columns)
    test_matrix = np.sort(np.column_stack(test_columns), axis=1)
    cal_matrix = np.sort(np.column_stack(cal_columns), axis=1)
    if physical_projection:
        test_matrix[~test["daylight"].to_numpy(bool), :] = 0.0
        cal_matrix[~calibration["daylight"].to_numpy(bool), :] = 0.0
    compute = {
        "best_iterations": best,
        "training_seconds_by_quantile": train_seconds,
        "training_seconds_total": float(sum(train_seconds.values())),
        "inference_seconds_by_quantile": inference_seconds,
        "inference_seconds_total": float(sum(inference_seconds.values())),
        "inference_seconds_per_site_day": float(sum(inference_seconds.values()) / max(1, test.groupby([test.columns[0], "delivery_date"], observed=True).ngroups if "delivery_date" in test else len(test) / 24)),
        "serialized_bytes_total": int(sum(model_bytes.values())),
        "process_rss_increase_bytes": int(max(0, psutil.Process(os.getpid()).memory_info().rss - rss_start)),
    }
    return raw, cal_matrix, test_matrix, compute


def group_contract(frame: pd.DataFrame, source: str) -> dict[str, pd.Series]:
    timestamp = pd.to_datetime(frame["LocalTime"] if source == "nrel" else frame["target_timestamp_utc"])
    zenith_col = "solar_zenith_deg" if source == "nrel" else "solar_zenith"
    zenith = frame[zenith_col]
    season = pd.Series(np.select([timestamp.dt.month.isin([12, 1, 2]), timestamp.dt.month.isin([3, 4, 5]), timestamp.dt.month.isin([6, 7, 8])], ["winter", "spring", "summer"], default="autumn"), index=frame.index)
    groups = {
        "solar_elevation_regime": pd.cut(90.0 - zenith, [-90, 0, 15, 35, 90], labels=["night", "low", "medium", "high"], include_lowest=True),
        "season": season,
        "production_regime": pd.cut(frame["target_power_normalized"], [-np.inf, 0.02, 0.20, 0.50, np.inf], labels=["near_zero", "low", "medium", "high"]),
    }
    if source == "nrel":
        groups["target_hour"] = frame["delivery_hour"].astype(int).astype(str)
        groups["clear_sky_variability"] = pd.qcut(frame["trajectory_max_abs_ramp"].rank(method="first"), 3, labels=["low", "medium", "high"])
    else:
        groups["lead_band"] = frame["lead_band"]
        groups["clear_sky_variability"] = pd.qcut(frame["nwp_ghi_trajectory_max_abs_ramp"].rank(method="first"), 3, labels=["low", "medium", "high"])
    return groups


def run_nrel(root: Path) -> None:
    design = json.loads((root / "configs/nrel_q1_study.json").read_text())
    frame = pd.read_parquet(root / "data/processed/nrel_q1_panel.parquet")
    frame["LocalTime"] = pd.to_datetime(frame["LocalTime"])
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"])
    frame["plant_type"] = frame["plant_type"].astype("category")
    frame["site_code"] = frame["site_id"].astype("category")
    split, t = design["known_site_temporal_split"], frame["LocalTime"]
    train = frame[t <= pd.Timestamp(split["train_end"])].copy()
    tune = frame[(t >= pd.Timestamp(split["tuning_start"])) & (t <= pd.Timestamp(split["tuning_end"]))].copy()
    cal = frame[(t >= pd.Timestamp(split["calibration_start"])) & (t <= pd.Timestamp(split["calibration_end"]))].copy()
    test = frame[t >= pd.Timestamp(split["test_start"])].copy()
    development = pd.concat([train, tune], ignore_index=True)
    specs = model_specifications()
    methods = [("lightgbm_nonphysical", "lightgbm", specs["M1_nonphysical_direct"], None, False), ("lightgbm_physical", "lightgbm", specs["M5_physical_residual_trajectory"], "source_forecast_normalized", True), ("xgboost_physical", "xgboost", specs["M5_physical_residual_trajectory"], "source_forecast_normalized", True)]
    output, prediction = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": "NREL simulated PV", "quantiles": QUANTILES.tolist(), "lead_time_limitation": "Row-level issue times are absent, so target hour is reported instead of lead time.", "models": {}}, test[["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized"]].reset_index(drop=True)
    for name, kind, spec, residual, project in methods:
        target = "source_residual_normalized" if residual else "target_power_normalized"
        raw, cal_q, test_q, compute = fit_quantiles(kind, train, tune, development, cal, test, spec["features"], target, [x for x in ["plant_type", "site_code"] if x in spec["features"]], residual, project)
        conformed, radii = conformalize(cal["target_power_normalized"].to_numpy(float), cal_q, test_q, cal["daylight"].to_numpy(bool), test["daylight"].to_numpy(bool))
        day = test["daylight"].to_numpy(bool)
        output["models"][name] = {"raw_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), test_q[day]), "conformal_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), conformed[day]), "physical_diagnostics": diagnostics(test, raw, test_q), "rearrangement_audit": rearrangement_audit(test.reset_index(drop=True), raw), "projection_audit": projection_audit(test.reset_index(drop=True), raw), "conformal_radii": radii, "conditional": conditional_metrics(test.reset_index(drop=True), conformed, group_contract(test.reset_index(drop=True), "nrel")), "compute": compute}
        for index, q in enumerate(QUANTILES):
            prediction[f"{name}_q{int(q*100):02d}"] = conformed[:, index]
    (root / "results/nrel_nine_quantile_review.json").write_text(json.dumps(output, indent=2) + "\n")
    prediction.to_parquet(root / "results/nrel_nine_quantile_review_predictions.parquet", index=False)


def run_pvod(root: Path) -> None:
    design = json.loads((root / "configs/pvod_q1_physics_study.json").read_text())
    frame = add_trajectory_features(pd.read_parquet(root / "data/processed/pvod_day_ahead_panel.parquet"))
    frame["station_code"] = frame["station"].astype("category")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    frame["delivery_date"] = frame["target_timestamp_utc"].dt.date.astype(str)
    frame["daylight"] = frame["clearsky_ghi"] >= float(design["daylight_clearsky_ghi_w_m2"])
    frame["lead_band"] = pd.cut(frame["lead_15min"] / 4.0, [27.99, 36, 44, 52], labels=["28-36h", "36-44h", "44-52h"], include_lowest=True).astype(str)
    specs = pvod_specifications()
    required = sorted({x for spec in specs.values() for x in spec["features"]}) + ["target_power_normalized"]
    frame = frame.dropna(subset=required).copy()
    split, t = design["split"], frame["target_timestamp_utc"]
    train = frame[t <= pd.Timestamp(split["train_end"])].copy()
    tune = frame[(t >= pd.Timestamp(split["tuning_start"])) & (t <= pd.Timestamp(split["tuning_end"]))].copy()
    cal = frame[(t >= pd.Timestamp(split["calibration_start"])) & (t <= pd.Timestamp(split["calibration_end"]))].copy()
    test = frame[t >= pd.Timestamp(split["test_start"])].copy()
    for part in [train, tune, cal, test]:
        part["physical_residual"] = part["target_power_normalized"] - part["clearsky_scaled_persistence_72h"]
    development = pd.concat([train, tune], ignore_index=True)
    methods = [("lightgbm_nonphysical", "lightgbm", specs["M1_nonphysical_nwp_direct"], None, False), ("lightgbm_physical", "lightgbm", specs["M5_physical_residual_trajectory"], "clearsky_scaled_persistence_72h", True), ("xgboost_physical", "xgboost", specs["M5_physical_residual_trajectory"], "clearsky_scaled_persistence_72h", True)]
    output, prediction = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": "PVOD measured PV with source-documented WRF schedule", "quantiles": QUANTILES.tolist(), "models": {}}, test[["station", "issue_timestamp_utc", "target_timestamp_utc", "delivery_date", "lead_15min", "daylight", "target_power_normalized"]].reset_index(drop=True)
    for name, kind, spec, residual, project in methods:
        target = "physical_residual" if residual else "target_power_normalized"
        raw, cal_q, test_q, compute = fit_quantiles(kind, train, tune, development, cal, test, spec["features"], target, ["station_code"], residual, project)
        conformed, radii = conformalize(cal["target_power_normalized"].to_numpy(float), cal_q, test_q, cal["daylight"].to_numpy(bool), test["daylight"].to_numpy(bool))
        day = test["daylight"].to_numpy(bool)
        output["models"][name] = {"raw_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), test_q[day]), "conformal_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), conformed[day]), "physical_diagnostics": diagnostics(test, raw, test_q), "rearrangement_audit": rearrangement_audit(test.reset_index(drop=True), raw), "projection_audit": projection_audit(test.reset_index(drop=True), raw), "conformal_radii": radii, "conditional": conditional_metrics(test.reset_index(drop=True), conformed, group_contract(test.reset_index(drop=True), "pvod")), "compute": compute}
        for index, q in enumerate(QUANTILES):
            prediction[f"{name}_q{int(q*100):02d}"] = conformed[:, index]
    (root / "results/pvod_nine_quantile_review.json").write_text(json.dumps(output, indent=2) + "\n")
    prediction.to_parquet(root / "results/pvod_nine_quantile_review_predictions.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["nrel", "pvod", "all"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.dataset in {"nrel", "all"}: run_nrel(args.project_root.resolve())
    if args.dataset in {"pvod", "all"}: run_pvod(args.project_root.resolve())


if __name__ == "__main__":
    main()
