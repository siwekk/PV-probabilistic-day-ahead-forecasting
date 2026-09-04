#!/usr/bin/env python3
"""Run paired uncertainty, seasonal calibration, and PVOD failure diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics import QUANTILES, hierarchical_block_bootstrap, interval_score, pinball_matrix


def crps_rows(y: np.ndarray, qhat: np.ndarray) -> np.ndarray:
    loss = pinball_matrix(y, qhat)
    return 2.0 * np.trapz(np.column_stack([loss[:, 0], loss, loss[:, -1]]), x=np.r_[0.0, QUANTILES, 1.0], axis=1)


def qmatrix(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_q{int(q*100):02d}" for q in QUANTILES]].to_numpy(float)


def compare(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, site: str, day: str) -> dict:
    y = frame["target_power_normalized"].to_numpy(float)
    return {
        "mae": hierarchical_block_bootstrap(frame, np.abs(y - a[:, 4]), np.abs(y - b[:, 4]), site, day),
        "crps_9q": hierarchical_block_bootstrap(frame, crps_rows(y, a), crps_rows(y, b), site, day),
        "interval_score_90": hierarchical_block_bootstrap(frame, interval_score(y, a[:, 0], a[:, 8], 0.10), interval_score(y, b[:, 0], b[:, 8], 0.10), site, day),
    }


def nrel_analysis(root: Path) -> dict:
    review = pd.read_parquet(root / "results/nrel_nine_quantile_review_predictions.parquet")
    review["LocalTime"] = pd.to_datetime(review["LocalTime"])
    review["delivery_date"] = pd.to_datetime(review["delivery_date"])
    review = review.loc[review["daylight"]].copy()
    lgb = qmatrix(review, "lightgbm_physical")
    xgb = qmatrix(review, "xgboost_physical")
    output = {"lightgbm_minus_xgboost": compare(review, lgb, xgb, "site_id", "delivery_date")}

    chronos = pd.read_parquet(root / "results/nrel_chronos2_predictions.parquet")
    chronos["LocalTime"] = pd.to_datetime(chronos["LocalTime"])
    merged = review.merge(chronos, on=["state", "site_id", "LocalTime"], suffixes=("", "_chronos"), validate="one_to_one")
    y = merged["target_power_normalized"].to_numpy(float)
    lgb_merged = qmatrix(merged, "lightgbm_physical")
    chronos_q = merged[["chronos2_q050_conformal", "chronos2_q500_conformal", "chronos2_q950_conformal"]].to_numpy(float)
    output["lightgbm_minus_chronos2"] = {
        "mae": hierarchical_block_bootstrap(merged, np.abs(y - lgb_merged[:, 4]), np.abs(y - chronos_q[:, 1]), "site_id", "delivery_date"),
        "interval_score_90": hierarchical_block_bootstrap(merged, interval_score(y, lgb_merged[:, 0], lgb_merged[:, 8], 0.10), interval_score(y, chronos_q[:, 0], chronos_q[:, 2], 0.10), "site_id", "delivery_date"),
    }

    state = pd.read_parquet(root / "results/nrel_state_holdout_predictions.parquet")
    state["LocalTime"] = pd.to_datetime(state["LocalTime"])
    state = state.loc[state["daylight"]].copy()
    state["season"] = np.select([state["LocalTime"].dt.month.isin([12, 1, 2]), state["LocalTime"].dt.month.isin([3, 4, 5]), state["LocalTime"].dt.month.isin([6, 7, 8])], ["winter", "spring", "summer"], default="autumn")
    seasonal = {}
    for (held_state, season), group in state.groupby(["state", "season"], observed=True):
        y_group = group["target_power_normalized"].to_numpy(float)
        seasonal[f"{held_state}_{season}"] = {
            "n": int(len(group)),
            "coverage_90": float(np.mean((y_group >= group["physics_q05_conformal"]) & (y_group <= group["physics_q95_conformal"]))),
            "width_90": float(np.mean(group["physics_q95_conformal"] - group["physics_q05_conformal"])),
        }
    output["state_transfer_seasonal_calibration"] = seasonal
    return output


def pvod_analysis(root: Path) -> dict:
    prediction = pd.read_parquet(root / "results/pvod_nine_quantile_review_predictions.parquet")
    prediction["delivery_date"] = pd.to_datetime(prediction["delivery_date"])
    daylight = prediction.loc[prediction["daylight"]].copy()
    nonphysical = qmatrix(daylight, "lightgbm_nonphysical")
    physical = qmatrix(daylight, "lightgbm_physical")
    xgb = qmatrix(daylight, "xgboost_physical")
    comparisons = {
        "physical_minus_nonphysical_lightgbm": compare(daylight, physical, nonphysical, "station", "delivery_date"),
        "physical_lightgbm_minus_physical_xgboost": compare(daylight, physical, xgb, "station", "delivery_date"),
    }

    panel = pd.read_parquet(root / "data/processed/pvod_day_ahead_panel.parquet")
    panel["target_timestamp_utc"] = pd.to_datetime(panel["target_timestamp_utc"], utc=True)
    panel["daylight"] = panel["clearsky_ghi"] >= 20.0
    diagnostics = {"interpretation": "Associations are diagnostic and do not identify a unique cause."}
    available = set(panel.columns)
    diagnostics["metadata_availability"] = {name: bool(name in available) for name in ["capacity", "latitude", "longitude", "tilt", "azimuth", "curtailment", "snow", "shading"]}
    diagnostics["sensor_and_support"] = {
        "negative_target_rate": float(np.mean(panel["target_power_normalized"] < 0)),
        "above_1_2_target_rate": float(np.mean(panel["target_power_normalized"] > 1.2)),
        "target_missing_rate": float(panel["target_power_normalized"].isna().mean()),
    }
    stations = {}
    for station, group in panel.groupby("station", observed=True):
        day = group.loc[group["daylight"] & group["target_power_normalized"].notna() & group["nwp_globalirrad"].notna()].copy()
        if day.empty:
            continue
        hour = day["target_timestamp_utc"].dt.hour
        stations[str(station)] = {
            "n": int(len(day)),
            "target_q995": float(day["target_power_normalized"].quantile(0.995)),
            "near_capacity_rate": float(np.mean(day["target_power_normalized"] >= 0.98)),
            "nwp_ghi_target_correlation": float(day[["nwp_globalirrad", "target_power_normalized"]].corr().iloc[0, 1]),
            "morning_mean_target": float(day.loc[hour < 12, "target_power_normalized"].mean()),
            "afternoon_mean_target": float(day.loc[hour >= 12, "target_power_normalized"].mean()),
        }
    diagnostics["by_station"] = stations
    return {"comparisons": comparisons, "measured_data_failure_diagnostics": diagnostics}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "nrel": nrel_analysis(root), "pvod": pvod_analysis(root)}
    (root / "results/review_posthoc_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
