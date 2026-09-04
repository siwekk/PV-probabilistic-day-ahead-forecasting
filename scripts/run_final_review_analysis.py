#!/usr/bin/env python3
"""Complete reviewer-requested PVOD, Chronos-2, and ECMWF analyses."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


COMMON_Q = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])
NATIVE_Q = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])


def pinball_rows(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    e = y[:, None] - q
    return np.maximum(COMMON_Q * e, (COMMON_Q - 1.0) * e)


def scores(y: np.ndarray, q: np.ndarray) -> dict[str, float | int]:
    q = np.maximum.accumulate(q, axis=1)
    loss = pinball_rows(y, q)
    crps = 2.0 * np.trapz(
        np.column_stack([loss[:, 0], loss, loss[:, -1]]),
        x=np.r_[0.0, COMMON_Q, 1.0], axis=1,
    )
    intervals = [(0, 8, 0.10), (1, 7, 0.20), (2, 6, 0.40), (3, 5, 0.60)]
    interval_scores = []
    for lo, hi, alpha in intervals:
        low, high = q[:, lo], q[:, hi]
        interval_scores.append(high - low + 2 / alpha * (low - y) * (y < low) + 2 / alpha * (y - high) * (y > high))
    wis = (0.5 * np.abs(y - q[:, 4]) + sum((alpha / 2) * value for value, (_, _, alpha) in zip(interval_scores, intervals))) / 4.5
    low, high = q[:, 0], q[:, 8]
    is90 = high - low + 20 * (low - y) * (y < low) + 20 * (y - high) * (y > high)
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(y - q[:, 4]))),
        "crps_9q": float(np.mean(crps)), "wis": float(np.mean(wis)),
        "coverage_90": float(np.mean((y >= low) & (y <= high))),
        "width_90": float(np.mean(high - low)), "interval_score_90": float(np.mean(is90)),
    }


def interpolate_native(frame: pd.DataFrame, stage: str) -> np.ndarray:
    native = np.column_stack([frame[f"chronos2_q{int(q*1000):03d}_{stage}"] for q in NATIVE_Q])
    native = np.maximum.accumulate(native, axis=1)
    output = np.empty((len(frame), len(COMMON_Q)))
    for row in range(len(frame)):
        output[row] = np.interp(COMMON_Q, NATIVE_Q, native[row])
    return output


def chronos_analysis(root: Path) -> dict:
    frame = pd.read_parquet(root / "results/nrel_chronos2_predictions.parquet")
    y = frame["target_power_normalized"].to_numpy(float)
    daylight = frame["daylight"].to_numpy(bool)
    stages = {}
    for stage in ["raw", "projected", "conformal"]:
        q = interpolate_native(frame, stage)
        stages[stage] = {"overall": scores(y, q), "daylight": scores(y[daylight], q[daylight]), "night": scores(y[~daylight], q[~daylight])}
    raw_native = np.column_stack([frame[f"chronos2_q{int(q*1000):03d}_raw"] for q in NATIVE_Q])
    stages["raw_diagnostics"] = {
        "crossing_row_rate": float(np.mean(np.any(np.diff(raw_native, axis=1) < 0, axis=1))),
        "negative_row_rate": float(np.mean(np.any(raw_native < 0, axis=1))),
        "above_1_2_row_rate": float(np.mean(np.any(raw_native > 1.2, axis=1))),
        "nonzero_night_median_rate": float(np.mean(np.abs(raw_native[~daylight, 4]) > 1e-9)),
        "interpolation": "Native ordered quantiles are linearly interpolated in probability space to the common nine-level grid. No extrapolation is needed because the common grid lies within 0.025 to 0.975.",
    }
    return stages


def block_effect(y: np.ndarray, physical: np.ndarray, nonphysical: np.ndarray, dates: pd.Series, rng: np.random.Generator, draws: int = 1000) -> tuple[float, float]:
    daily = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "d": np.abs(y - physical) - np.abs(y - nonphysical)}).groupby("date")["d"].mean().to_numpy()
    estimate = float(np.mean(daily))
    n, block = len(daily), 7
    values = []
    for _ in range(draws):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]
        values.append(float(np.mean(daily[idx])))
    return estimate, float(np.std(values, ddof=1))


def random_effects_meta(effect: np.ndarray, se: np.ndarray) -> dict:
    variance = se ** 2
    fixed_w = 1 / variance
    fixed = np.sum(fixed_w * effect) / np.sum(fixed_w)
    q = np.sum(fixed_w * (effect - fixed) ** 2)
    c = np.sum(fixed_w) - np.sum(fixed_w ** 2) / np.sum(fixed_w)
    tau2 = max(0.0, (q - len(effect) + 1) / c)
    w = 1 / (variance + tau2)
    mean = np.sum(w * effect) / np.sum(w)
    mean_se = np.sqrt(1 / np.sum(w))
    return {"effect": float(mean), "ci95": [float(mean - 1.96 * mean_se), float(mean + 1.96 * mean_se)], "tau2": float(tau2), "q": float(q)}


def meta_regression(effect: np.ndarray, se: np.ndarray, moderator: np.ndarray) -> dict:
    x = (moderator - np.mean(moderator)) / np.std(moderator, ddof=1)
    base = random_effects_meta(effect, se)
    w = 1 / (se ** 2 + base["tau2"])
    design = np.column_stack([np.ones(len(x)), x])
    covariance = np.linalg.inv(design.T @ (w[:, None] * design))
    beta = covariance @ design.T @ (w * effect)
    slope_se = np.sqrt(covariance[1, 1])
    return {"slope_per_sd": float(beta[1]), "ci95": [float(beta[1] - 1.96 * slope_se), float(beta[1] + 1.96 * slope_se)]}


def pvod_analysis(root: Path) -> dict:
    prediction = pd.read_parquet(root / "results/pvod_nine_quantile_review_predictions.parquet")
    prediction["target_timestamp_utc"] = pd.to_datetime(prediction["target_timestamp_utc"], utc=True)
    prediction = prediction.loc[prediction["daylight"]].copy()
    panel = pd.read_parquet(root / "data/processed/pvod_day_ahead_panel.parquet")
    panel["target_timestamp_utc"] = pd.to_datetime(panel["target_timestamp_utc"], utc=True)
    keep = ["station", "target_timestamp_utc", "nwp_globalirrad", "target_power_normalized"]
    context = panel[keep].drop_duplicates(["station", "target_timestamp_utc"])
    prediction = prediction.merge(context[["station", "target_timestamp_utc", "nwp_globalirrad"]], on=["station", "target_timestamp_utc"], how="left", validate="many_to_one")
    rng = np.random.default_rng(20260829)
    rows = []
    for station, test in prediction.groupby("station", observed=True):
        full = panel.loc[panel["station"] == station].copy()
        full["hour"] = full["target_timestamp_utc"].dt.hour + full["target_timestamp_utc"].dt.minute / 60
        valid = full.dropna(subset=["target_power_normalized", "nwp_globalirrad"])
        target = test["target_power_normalized"].to_numpy(float)
        p = test["lightgbm_physical_q50"].to_numpy(float)
        n = test["lightgbm_nonphysical_q50"].to_numpy(float)
        effect, effect_se = block_effect(target, p, n, test["delivery_date"], rng)
        q995 = float(valid["target_power_normalized"].quantile(0.995))
        top = valid["target_power_normalized"] >= 0.99 * q995
        power_centroid = np.average(valid["hour"], weights=np.clip(valid["target_power_normalized"], 0, None) + 1e-9)
        ghi_centroid = np.average(valid["hour"], weights=np.clip(valid["nwp_globalirrad"], 0, None) + 1e-9)
        rows.append({
            "station": str(station), "n_test": int(len(test)), "physical_minus_nonphysical_mae": effect,
            "effect_block_se": effect_se, "capacity_proxy_q995": q995,
            "irradiance_power_correlation": float(valid[["nwp_globalirrad", "target_power_normalized"]].corr().iloc[0, 1]),
            "missing_target_rate": float(full["target_power_normalized"].isna().mean()),
            "apparent_clipping_rate": float(top.mean()),
            "orientation_mismatch_hours": float(abs(power_centroid - ghi_centroid)),
        })
    station = pd.DataFrame(rows)
    effect = station["physical_minus_nonphysical_mae"].to_numpy()
    se = np.maximum(station["effect_block_se"].to_numpy(), 1e-6)
    moderators = ["capacity_proxy_q995", "irradiance_power_correlation", "missing_target_rate", "apparent_clipping_rate", "orientation_mismatch_hours"]
    return {
        "interpretation": "Exploratory station-level random-effects meta-analysis. There are only eight stations, proxies are not verified plant metadata, and moderator intervals are descriptive rather than causal.",
        "stations": station.to_dict(orient="records"),
        "random_effects_summary": random_effects_meta(effect, se),
        "univariable_meta_regressions": {name: meta_regression(effect, se, station[name].to_numpy(float)) for name in moderators},
        "stations_where_physical_mae_is_lower": int(np.sum(effect < 0)),
    }


def ecmwf_analysis(root: Path) -> dict:
    prediction = pd.read_parquet(root / "results/ecmwf_vintage_review_predictions.parquet")
    panel = pd.read_parquet(root / "data/processed/ecmwf_jacumba_vintage_panel.parquet")
    panel["valid_time_utc"] = pd.to_datetime(panel["valid_time_utc"], utc=True)
    prediction["valid_time_utc"] = pd.to_datetime(prediction["valid_time_utc"], utc=True)
    prediction = prediction.merge(panel[["valid_time_utc", "solar_zenith", "clear_sky_ghi"]], on="valid_time_utc", how="left", validate="one_to_one")
    prediction["utc_hour"] = prediction["valid_time_utc"].dt.hour
    prediction["solar_elevation"] = 90 - prediction["solar_zenith"]
    prediction["lead_band"] = pd.cut(prediction["lead_hours"], [11.9, 17, 23, 29, 35.1], labels=["12-17", "18-23", "24-29", "30-35"])
    daylight = prediction.loc[prediction["daylight"]].copy()
    groups = []
    for label, group in daylight.groupby("lead_band", observed=True):
        y = group["target_power_normalized"].to_numpy(float)
        groups.append({
            "lead_band": str(label), "n": int(len(group)), "utc_hours": sorted(group["utc_hour"].unique().astype(int).tolist()),
            "mean_solar_elevation": float(group["solar_elevation"].mean()), "mean_clear_sky_ghi": float(group["clear_sky_ghi"].mean()),
            "mean_target": float(y.mean()),
            "nonphysical_mae": float(np.mean(np.abs(y - group["nonphysical_direct_q50"].to_numpy(float)))),
            "physical_mae": float(np.mean(np.abs(y - group["physical_residual_q50"].to_numpy(float)))),
        })
    return {
        "schedule": {"assumed_issue_utc": "12:00 on the preceding UTC day", "delivery_window_utc": "00:00 to 23:00 on the delivery day", "lead_hours": "12 to 35", "identity": "lead_hours = 12 + delivery UTC hour"},
        "identifiability": "Lead time and UTC hour are perfectly confounded in this single-cycle archive. Daylight lead bands also differ in solar elevation and expected production, so the lead trend cannot be interpreted as a pure horizon effect.",
        "daylight_by_lead_band": groups,
        "correlation_lead_utc_hour": float(prediction[["lead_hours", "utc_hour"]].corr().iloc[0, 1]),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pvod_station_meta_analysis": pvod_analysis(root),
        "chronos2_postprocessing_decomposition": chronos_analysis(root),
        "ecmwf_timeline_and_confounding": ecmwf_analysis(root),
    }
    path = root / "results/final_review_analysis.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
