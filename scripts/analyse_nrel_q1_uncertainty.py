#!/usr/bin/env python3
"""Compute site-balanced moving-block uncertainty for the NREL experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 20260828
BOOTSTRAP_REPLICATIONS = 2000
BLOCK_DAYS = 7


def row_interval_score(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    alpha = 0.10
    return (high - low) + (2.0 / alpha) * (low - y) * (y < low) + (2.0 / alpha) * (
        y - high
    ) * (y > high)


def row_pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def daily_differences(
    frame: pd.DataFrame,
    comparator_prefix: str,
    physics_prefix: str,
) -> dict[str, pd.DataFrame]:
    frame = frame.loc[frame["daylight"]].copy()
    y = frame["target_power_normalized"].to_numpy(dtype=float)
    comparator_mae = np.abs(y - frame[f"{comparator_prefix}_q50"].to_numpy(dtype=float))
    physics_mae = np.abs(y - frame[f"{physics_prefix}_q50"].to_numpy(dtype=float))
    comparator_interval = row_interval_score(
        y,
        frame[f"{comparator_prefix}_q05"].to_numpy(dtype=float),
        frame[f"{comparator_prefix}_q95"].to_numpy(dtype=float),
    )
    physics_interval = row_interval_score(
        y,
        frame[f"{physics_prefix}_q05"].to_numpy(dtype=float),
        frame[f"{physics_prefix}_q95"].to_numpy(dtype=float),
    )
    comparator_pinball = np.mean(
        np.column_stack(
            [
                row_pinball(y, frame[f"{comparator_prefix}_q05"].to_numpy(), 0.05),
                row_pinball(y, frame[f"{comparator_prefix}_q50"].to_numpy(), 0.50),
                row_pinball(y, frame[f"{comparator_prefix}_q95"].to_numpy(), 0.95),
            ]
        ),
        axis=1,
    )
    physics_pinball = np.mean(
        np.column_stack(
            [
                row_pinball(y, frame[f"{physics_prefix}_q05"].to_numpy(), 0.05),
                row_pinball(y, frame[f"{physics_prefix}_q50"].to_numpy(), 0.50),
                row_pinball(y, frame[f"{physics_prefix}_q95"].to_numpy(), 0.95),
            ]
        ),
        axis=1,
    )
    frame["delta_mae"] = physics_mae - comparator_mae
    frame["delta_interval_score"] = physics_interval - comparator_interval
    frame["delta_mean_pinball"] = physics_pinball - comparator_pinball
    grouped = frame.groupby(["site_id", "delivery_date"], observed=True)
    return {
        metric: grouped[metric].mean().reset_index()
        for metric in ("delta_mae", "delta_interval_score", "delta_mean_pinball")
    }


def moving_block_sample(values: np.ndarray, rng: np.random.Generator) -> float:
    count = len(values)
    if count == 0:
        return np.nan
    starts = rng.integers(0, count, size=int(np.ceil(count / BLOCK_DAYS)))
    offsets = np.arange(BLOCK_DAYS)
    sampled = np.concatenate([values[(start + offsets) % count] for start in starts])[:count]
    return float(np.mean(sampled))


def hierarchical_interval(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, Any]:
    series = {
        site: part.sort_values("delivery_date").iloc[:, -1].to_numpy(dtype=float)
        for site, part in frame.groupby("site_id", observed=True)
    }
    sites = np.array(sorted(series), dtype=object)
    site_means = np.array([np.mean(series[site]) for site in sites])
    estimates = np.empty(BOOTSTRAP_REPLICATIONS, dtype=float)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        selected = rng.choice(sites, size=len(sites), replace=True)
        estimates[replication] = np.mean(
            [moving_block_sample(series[site], rng) for site in selected]
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "estimand": "equal-site mean of daily daylight score differences, physics minus comparator",
        "sites": int(len(sites)),
        "point_estimate": float(np.mean(site_means)),
        "confidence_interval_95": [float(low), float(high)],
        "probability_improvement": float(np.mean(estimates < 0.0)),
        "bootstrap_replications": BOOTSTRAP_REPLICATIONS,
        "moving_block_days": BLOCK_DAYS,
    }


def analyse_frame(
    frame: pd.DataFrame,
    comparator_prefix: str,
    physics_prefix: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    differences = daily_differences(frame, comparator_prefix, physics_prefix)
    return {metric: hierarchical_interval(values, rng) for metric, values in differences.items()}


def render_tex(result: dict[str, Any]) -> str:
    temporal = result["known_site_temporal"]
    lines = [
        r"\section{NREL hierarchical uncertainty analysis}",
        "",
        (
            "Uncertainty was estimated with a paired hierarchical bootstrap. Sites were sampled "
            "with replacement, then seven-day moving blocks were sampled within each selected site. "
            "All primary differences use daylight rows and are defined as the physics-informed "
            "model score minus the matched non-physical model score. Negative values favour the "
            "physics-informed model."
        ),
        "",
        r"\begin{center}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Experiment and score & Difference & 95\% lower & 95\% upper \\",
        r"\hline",
    ]
    for label, values in (
        ("Known-site daylight MAE", temporal["delta_mae"]),
        ("Known-site daylight interval score", temporal["delta_interval_score"]),
        ("Known-site daylight mean pinball", temporal["delta_mean_pinball"]),
    ):
        low, high = values["confidence_interval_95"]
        lines.append(f"{label} & {values['point_estimate']:.6f} & {low:.6f} & {high:.6f} " + r"\\")
    for state, state_result in result["state_holdout"].items():
        values = state_result["delta_interval_score"]
        low, high = values["confidence_interval_95"]
        lines.append(
            f"Held-out {state} daylight interval score & {values['point_estimate']:.6f} & "
            f"{low:.6f} & {high:.6f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{center}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    rng = np.random.default_rng(SEED)
    temporal = pd.read_parquet(root / "results" / "nrel_physics_ladder_predictions.parquet")
    state = pd.read_parquet(root / "results" / "nrel_state_holdout_predictions.parquet")
    site_holdout = pd.read_parquet(root / "results" / "nrel_site_holdout_predictions.parquet")
    xgboost = pd.read_parquet(root / "results" / "nrel_matched_xgboost_predictions.parquet")
    xgboost_columns = ["site_id", "LocalTime", "xgb_q05", "xgb_q50", "xgb_q95"]
    matched = temporal.merge(
        xgboost[xgboost_columns],
        on=["site_id", "LocalTime"],
        how="inner",
        validate="one_to_one",
    )
    if len(matched) != len(temporal):
        raise ValueError("Matched XGBoost predictions do not align with the LightGBM test panel")
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "known_site_temporal": analyse_frame(temporal, "M1", "M5", rng),
        "matched_xgboost": analyse_frame(matched, "xgb", "M5", rng),
        "complete_site_future_time_holdout": analyse_frame(
            site_holdout, "nonphysics", "physics", rng
        ),
        "state_holdout": {},
    }
    for state_name in sorted(state["state"].unique()):
        result["state_holdout"][state_name] = analyse_frame(
            state.loc[state["state"] == state_name], "nonphysics", "physics", rng
        )
    result_path = root / "results" / "nrel_q1_uncertainty.json"
    tex_path = root / "docs" / "nrel_q1_uncertainty_findings.tex"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    tex_path.write_text(render_tex(result), encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
