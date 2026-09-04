#!/usr/bin/env python3
"""Diagnose when the PVOD physical representation loses predictive accuracy.

This analysis is descriptive. PVOD has WRF irradiance and clear-sky GHI, but
does not contain measured POA irradiance, cloud cover, aerosol optical depth,
or verified plant orientation. The output therefore localises the physical
penalty without assigning it to an unobserved atmospheric or plant mechanism.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics import hierarchical_block_bootstrap


KEYS = ["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min"]


def score_group(group: pd.DataFrame) -> dict:
    target = group["target_power_normalized"].to_numpy(float)
    nonphysical = group["lightgbm_nonphysical_q50"].to_numpy(float)
    physical = group["lightgbm_physical_q50"].to_numpy(float)
    baseline = group["clearsky_scaled_persistence_72h"].to_numpy(float)
    nonphysical_loss = np.abs(target - nonphysical)
    physical_loss = np.abs(target - physical)
    comparison = hierarchical_block_bootstrap(
        group,
        physical_loss,
        nonphysical_loss,
        site_col="station",
        day_col="delivery_date",
        draws=2000,
        block_days=7,
        seed=20260831,
    )
    return {
        "n": int(len(group)),
        "station_days": int(group[["station", "delivery_date"]].drop_duplicates().shape[0]),
        "nonphysical_mae": float(nonphysical_loss.mean()),
        "physical_mae": float(physical_loss.mean()),
        "physical_minus_nonphysical_mae": float((physical_loss - nonphysical_loss).mean()),
        "physical_minus_nonphysical_block_interval": comparison,
        "reference_mae": float(np.mean(np.abs(target - baseline))),
        "reference_bias": float(np.mean(baseline - target)),
        "mean_wrf_clear_sky_index": float(group["wrf_clear_sky_index"].mean()),
    }


def grouped_scores(frame: pd.DataFrame, column: str) -> dict:
    return {
        str(label): score_group(group)
        for label, group in frame.groupby(column, observed=True)
        if len(group) >= 200
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prediction = pd.read_parquet(root / "results" / "pvod_nine_quantile_review_predictions.parquet")
    panel = pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")

    for column in ["issue_timestamp_utc", "target_timestamp_utc"]:
        prediction[column] = pd.to_datetime(prediction[column], utc=True)
        panel[column] = pd.to_datetime(panel[column], utc=True)
    prediction["delivery_date"] = pd.to_datetime(prediction["delivery_date"])

    predictors = panel[
        KEYS
        + [
            "nwp_globalirrad",
            "nwp_directirrad",
            "clearsky_ghi",
            "clearsky_scaled_persistence_72h",
            "solar_zenith",
        ]
    ]
    frame = prediction.merge(predictors, on=KEYS, how="left", validate="one_to_one")
    frame = frame.loc[frame["daylight"]].dropna(
        subset=[
            "target_power_normalized",
            "lightgbm_nonphysical_q50",
            "lightgbm_physical_q50",
            "nwp_globalirrad",
            "clearsky_ghi",
            "clearsky_scaled_persistence_72h",
        ]
    ).copy()

    frame["wrf_clear_sky_index"] = (
        frame["nwp_globalirrad"] / frame["clearsky_ghi"].clip(lower=20.0)
    ).clip(lower=0.0, upper=2.0)
    frame["wrf_irradiance_regime"] = pd.cut(
        frame["wrf_clear_sky_index"],
        bins=[-np.inf, 0.35, 0.75, np.inf],
        labels=["low", "intermediate", "high"],
    )
    frame["reference_absolute_error"] = np.abs(
        frame["target_power_normalized"] - frame["clearsky_scaled_persistence_72h"]
    )
    frame["reference_error_regime"] = pd.qcut(
        frame["reference_absolute_error"].rank(method="first"),
        q=3,
        labels=["low", "intermediate", "high"],
    )
    hour = frame["target_timestamp_utc"].dt.hour
    frame["solar_half"] = np.where(hour < 12, "before_12_UTC", "from_12_UTC")

    target = frame["target_power_normalized"].to_numpy(float)
    physical_loss = np.abs(target - frame["lightgbm_physical_q50"].to_numpy(float))
    nonphysical_loss = np.abs(target - frame["lightgbm_nonphysical_q50"].to_numpy(float))
    frame["physical_penalty"] = physical_loss - nonphysical_loss
    daily = (
        frame.groupby(["station", "delivery_date"], observed=True)
        .agg(
            physical_penalty=("physical_penalty", "mean"),
            reference_absolute_error=("reference_absolute_error", "mean"),
            wrf_clear_sky_index=("wrf_clear_sky_index", "mean"),
        )
        .reset_index()
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive localisation of the PVOD physical penalty; not causal attribution.",
        "unavailable_mechanism_variables": [
            "forecast cloud cover",
            "aerosol optical depth",
            "measured plane-of-array irradiance",
            "verified tilt",
            "verified azimuth",
        ],
        "overall": score_group(frame),
        "by_wrf_irradiance_regime": grouped_scores(frame, "wrf_irradiance_regime"),
        "by_reference_error_regime": grouped_scores(frame, "reference_error_regime"),
        "by_solar_half": grouped_scores(frame, "solar_half"),
        "daily_spearman_associations": {
            "penalty_vs_reference_absolute_error": float(
                daily[["physical_penalty", "reference_absolute_error"]].corr(method="spearman").iloc[0, 1]
            ),
            "penalty_vs_wrf_clear_sky_index": float(
                daily[["physical_penalty", "wrf_clear_sky_index"]].corr(method="spearman").iloc[0, 1]
            ),
            "station_days": int(len(daily)),
        },
    }
    path = root / "results" / "pvod_physical_mechanism_diagnostic.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
