#!/usr/bin/env python3
"""Extended uncertainty analysis for adaptation, ECMWF, and conditional coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from review_metrics import QUANTILES, interval_score, metrics, pinball_matrix


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SWEEP = RESULTS / "chronos2_adaptation_sweep"
BLOCKS = (3, 7, 14, 28)
SEEDS = (20260829, 20260830, 20260831)
NATIVE_CHRONOS_LEVELS = np.asarray([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])


def qmatrix(frame: pd.DataFrame, prefix: str = "") -> np.ndarray:
    native = frame[
        [f"{prefix}q{int(round(q * 1000)):03d}" for q in NATIVE_CHRONOS_LEVELS]
    ].to_numpy(float)
    return interp1d(
        NATIVE_CHRONOS_LEVELS,
        native,
        axis=1,
        bounds_error=False,
        fill_value=(native[:, 0], native[:, -1]),
    )(QUANTILES)


def qmatrix_2digit(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    return frame[[f"{prefix}_q{int(q * 100):02d}" for q in QUANTILES]].to_numpy(float)


def crps_rows(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    loss = pinball_matrix(y, prediction)
    return 2.0 * np.trapz(
        np.column_stack([loss[:, 0], loss, loss[:, -1]]),
        x=np.r_[0.0, QUANTILES, 1.0],
        axis=1,
    )


def two_way_block_bootstrap(
    frame: pd.DataFrame,
    values: np.ndarray,
    site_col: str,
    day_col: str,
    block_days: int,
    draws: int = 4000,
    seed: int = 20260829,
) -> dict:
    work = frame[[site_col, day_col]].copy()
    work["value"] = np.asarray(values, float)
    daily = work.groupby([site_col, day_col], observed=True)["value"].mean().unstack(day_col)
    matrix = daily.to_numpy(float)
    n_sites, n_days = matrix.shape
    rng = np.random.default_rng(seed + block_days)
    estimates = np.empty(draws)
    n_blocks = int(np.ceil(n_days / block_days))
    offsets = np.arange(block_days)
    for draw in range(draws):
        sites = rng.integers(0, n_sites, size=n_sites)
        starts = rng.integers(0, n_days, size=n_blocks)
        days = ((starts[:, None] + offsets[None, :]) % n_days).ravel()[:n_days]
        estimates[draw] = np.nanmean(matrix[np.ix_(sites, days)])
    return {
        "draws": draws,
        "block_days": block_days,
        "estimate": float(np.nanmean(matrix)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def loss_contract(y: np.ndarray, prediction: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mae": np.abs(y - prediction[:, 4]),
        "crps_9q": crps_rows(y, prediction),
        "interval_score_90": interval_score(y, prediction[:, 0], prediction[:, 8], 0.10),
    }


def compare_all_blocks(
    frame: pd.DataFrame,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    site_col: str,
    day_col: str,
) -> dict:
    y = frame["target_power_normalized"].to_numpy(float)
    a = loss_contract(y, prediction_a)
    b = loss_contract(y, prediction_b)
    return {
        metric: {
            str(block): two_way_block_bootstrap(
                frame, a[metric] - b[metric], site_col, day_col, block
            )
            for block in BLOCKS
        }
        for metric in a
    }


def adaptation_analysis() -> dict:
    zero = pd.read_parquet(RESULTS / "nrel_chronos2_predictions.parquet")
    zero["LocalTime"] = pd.to_datetime(zero["LocalTime"])
    zero["delivery_date"] = pd.to_datetime(zero["delivery_date"])
    zero = zero.loc[zero["daylight"]].reset_index(drop=True)
    lgb = pd.read_parquet(RESULTS / "nrel_nine_quantile_review_predictions.parquet")
    lgb["LocalTime"] = pd.to_datetime(lgb["LocalTime"])
    lgb = lgb.loc[lgb["daylight"]].reset_index(drop=True)
    merged_base = zero.merge(
        lgb[["state", "site_id", "LocalTime", *[f"lightgbm_physical_q{int(q*100):02d}" for q in QUANTILES]]],
        on=["state", "site_id", "LocalTime"],
        validate="one_to_one",
    )
    zero_native = merged_base[
        [f"chronos2_q{int(round(q*1000)):03d}_conformal" for q in NATIVE_CHRONOS_LEVELS]
    ].to_numpy(float)
    zero_q = interp1d(NATIVE_CHRONOS_LEVELS, zero_native, axis=1, bounds_error=False, fill_value=(zero_native[:, 0], zero_native[:, -1]))(QUANTILES)
    lgb_q = qmatrix_2digit(merged_base, "lightgbm_physical")
    output = {
        "zero_shot_metrics": metrics(merged_base["target_power_normalized"].to_numpy(float), zero_q),
        "physical_lightgbm_metrics": metrics(merged_base["target_power_normalized"].to_numpy(float), lgb_q),
        "budget_ensembles": {},
        "seed_metrics": {},
        "optimization_sensitivity": {},
    }
    for days in (1, 3, 7, 14, 30):
        predictions = []
        seed_metrics = []
        for seed in SEEDS:
            identifier = f"lora_d{days}_s{seed}_lr1em05_r8_n300"
            frame = pd.read_parquet(SWEEP / f"{identifier}.parquet")
            frame["LocalTime"] = pd.to_datetime(frame["LocalTime"])
            frame = frame.loc[frame["daylight"]].reset_index(drop=True)
            aligned = merged_base[["state", "site_id", "LocalTime"]].merge(
                frame, on=["state", "site_id", "LocalTime"], validate="one_to_one"
            )
            prediction = qmatrix(aligned)
            predictions.append(prediction)
            seed_metrics.append(metrics(aligned["target_power_normalized"].to_numpy(float), prediction))
        ensemble = np.sort(np.mean(predictions, axis=0), axis=1)
        output["seed_metrics"][str(days)] = seed_metrics
        output["budget_ensembles"][str(days)] = {
            "metrics": metrics(merged_base["target_power_normalized"].to_numpy(float), ensemble),
            "minus_zero_shot": compare_all_blocks(merged_base, ensemble, zero_q, "site_id", "delivery_date"),
            "minus_physical_lightgbm": compare_all_blocks(merged_base, ensemble, lgb_q, "site_id", "delivery_date"),
        }
    sensitivity_ids = [
        "lora_d7_s20260829_lr3em06_r8_n300",
        "lora_d7_s20260829_lr1em05_r4_n300",
        "lora_d7_s20260829_lr1em05_r8_n300",
        "lora_d7_s20260829_lr1em05_r16_n300",
        "lora_d7_s20260829_lr3em05_r8_n300",
    ]
    full_ids = [f"full_d30_s{seed}_lr1em06_rall_n300" for seed in SEEDS]
    sensitivity_ids.extend(identifier for identifier in full_ids if (SWEEP / f"{identifier}.json").exists())
    for identifier in sensitivity_ids:
        result = json.loads((SWEEP / f"{identifier}.json").read_text())
        output["optimization_sensitivity"][identifier] = {
            "fit_seconds": result["fit_seconds"],
            "checkpoint_bytes": result["checkpoint_bytes"],
            "daylight": result["projected_conformal"]["daylight"],
        }
    available_full = [identifier for identifier in full_ids if (SWEEP / f"{identifier}.parquet").exists()]
    if available_full:
        full_predictions = []
        full_seed_metrics = []
        for identifier in available_full:
            full = pd.read_parquet(SWEEP / f"{identifier}.parquet")
            full["LocalTime"] = pd.to_datetime(full["LocalTime"])
            full = full.loc[full["daylight"]].reset_index(drop=True)
            aligned = merged_base[["state", "site_id", "LocalTime"]].merge(
                full, on=["state", "site_id", "LocalTime"], validate="one_to_one"
            )
            prediction = qmatrix(aligned)
            full_predictions.append(prediction)
            full_seed_metrics.append(metrics(aligned["target_power_normalized"].to_numpy(float), prediction))
        full_q = np.sort(np.mean(full_predictions, axis=0), axis=1)
        output["full_finetuning"] = {
            "identifiers": available_full,
            "seed_metrics": full_seed_metrics,
            "metrics": metrics(aligned["target_power_normalized"].to_numpy(float), full_q),
            "minus_zero_shot": compare_all_blocks(merged_base, full_q, zero_q, "site_id", "delivery_date"),
            "minus_physical_lightgbm": compare_all_blocks(merged_base, full_q, lgb_q, "site_id", "delivery_date"),
        }
    return output


def ecmwf_analysis() -> dict:
    frame = pd.read_parquet(RESULTS / "ecmwf_vintage_review_predictions.parquet")
    frame["delivery_date"] = pd.to_datetime(frame["delivery_date"])
    frame["site"] = "Jacumba"
    frame = frame.loc[frame["daylight"]].reset_index(drop=True)
    nonphysical = qmatrix_2digit(frame, "nonphysical_direct")
    physical = qmatrix_2digit(frame, "physical_residual")
    return {
        "physical_minus_nonphysical": compare_all_blocks(
            frame, physical, nonphysical, "site", "delivery_date"
        )
    }


def conditional_coverage() -> dict:
    output = {"nrel_state_season": {}, "ecmwf": {}}
    state = pd.read_parquet(RESULTS / "nrel_state_holdout_predictions.parquet")
    state["LocalTime"] = pd.to_datetime(state["LocalTime"])
    state["delivery_date"] = state["LocalTime"].dt.floor("D")
    state = state.loc[state["daylight"]].copy()
    state["season"] = np.select(
        [state["LocalTime"].dt.month.isin([12, 1, 2]), state["LocalTime"].dt.month.isin([3, 4, 5]), state["LocalTime"].dt.month.isin([6, 7, 8])],
        ["winter", "spring", "summer"],
        default="autumn",
    )
    for (held_state, season), group in state.groupby(["state", "season"], observed=True):
        covered = (
            (group["target_power_normalized"] >= group["physics_q05_conformal"])
            & (group["target_power_normalized"] <= group["physics_q95_conformal"])
        ).to_numpy(float)
        output["nrel_state_season"][f"{held_state}_{season}"] = two_way_block_bootstrap(
            group, covered, "site_id", "delivery_date", 7
        )
    vintage = pd.read_parquet(RESULTS / "ecmwf_vintage_review_predictions.parquet")
    vintage["delivery_date"] = pd.to_datetime(vintage["delivery_date"])
    vintage["valid_time_utc"] = pd.to_datetime(vintage["valid_time_utc"], utc=True)
    panel = pd.read_parquet(ROOT / "data/processed/ecmwf_jacumba_vintage_panel.parquet")
    panel["valid_time_utc"] = pd.to_datetime(panel["valid_time_utc"], utc=True)
    vintage = vintage.merge(
        panel[["valid_time_utc", "solar_zenith", "trajectory_max_abs_ramp"]],
        on="valid_time_utc",
        how="left",
        validate="one_to_one",
    )
    vintage["site"] = "Jacumba"
    vintage = vintage.loc[vintage["daylight"]].copy()
    group_labels = {
        "lead_band": pd.cut(vintage["lead_hours"], [11.9, 17, 23, 29, 35.1], labels=["12-17h", "18-23h", "24-29h", "30-35h"]),
        "solar_elevation_regime": pd.cut(90 - vintage["solar_zenith"], [-90, 0, 15, 35, 90], labels=["night", "low", "medium", "high"]),
        "season": np.select(
            [vintage["valid_time_utc"].dt.month.isin([12, 1, 2]), vintage["valid_time_utc"].dt.month.isin([3, 4, 5]), vintage["valid_time_utc"].dt.month.isin([6, 7, 8])],
            ["winter", "spring", "summer"], default="autumn"
        ),
        "clear_sky_variability": pd.qcut(vintage["trajectory_max_abs_ramp"].rank(method="first"), 3, labels=["low", "medium", "high"]),
        "production_regime": pd.cut(vintage["target_power_normalized"], [-np.inf, 0.02, 0.2, 0.5, np.inf], labels=["near_zero", "low", "medium", "high"]),
    }
    prediction = qmatrix_2digit(vintage, "nonphysical_direct")
    covered_all = (
        (vintage["target_power_normalized"].to_numpy(float) >= prediction[:, 0])
        & (vintage["target_power_normalized"].to_numpy(float) <= prediction[:, 8])
    ).astype(float)
    for family, labels in group_labels.items():
        output["ecmwf"][family] = {}
        labels = pd.Series(labels, index=vintage.index).astype(str)
        for level in sorted(labels.unique()):
            mask = labels.eq(level).to_numpy()
            if mask.sum() < 30:
                continue
            group = vintage.loc[mask].copy()
            output["ecmwf"][family][level] = two_way_block_bootstrap(
                group, covered_all[mask], "site", "delivery_date", 7
            )
    return output


def main() -> None:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_contract": {
            "draws": 4000,
            "block_lengths_days": BLOCKS,
            "site_resampling": "sites sampled with replacement",
            "temporal_resampling": "common circular day blocks preserve shared weather dependence",
        },
        "chronos2": adaptation_analysis(),
        "ecmwf": ecmwf_analysis(),
        "conditional_coverage": conditional_coverage(),
    }
    path = RESULTS / "extended_uncertainty_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
