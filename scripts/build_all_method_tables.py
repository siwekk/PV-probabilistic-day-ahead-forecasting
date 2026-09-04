#!/usr/bin/env python3
"""Build complete matched-method metric tables from saved predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def point_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    denominator = np.abs(target) + np.abs(prediction)
    smape = np.divide(
        2.0 * np.abs(error), denominator, out=np.zeros_like(error), where=denominator > 0
    )
    variation = np.sum(np.square(target - target.mean()))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(100.0 * np.mean(smape)),
        "r2": float(1.0 - np.sum(np.square(error)) / variation),
    }


def probabilistic_metrics(
    target: np.ndarray, low: np.ndarray, median: np.ndarray, high: np.ndarray
) -> dict[str, float]:
    predictions = np.sort(np.column_stack([low, median, high]), axis=1)
    levels = np.asarray([0.0, 0.05, 0.50, 0.95, 1.0])
    expanded = predictions[:, [0, 0, 1, 2, 2]]
    losses = np.column_stack(
        [
            np.maximum(q * (target - expanded[:, i]), (q - 1.0) * (target - expanded[:, i]))
            for i, q in enumerate(levels)
        ]
    )
    crps = 2.0 * np.trapz(losses, x=levels, axis=1)
    low, median, high = predictions.T
    interval_score = (
        high
        - low
        + 20.0 * (low - target) * (target < low)
        + 20.0 * (target - high) * (target > high)
    )
    return {
        **point_metrics(target, median),
        "crps_3q": float(np.mean(crps)),
        "coverage_90": float(np.mean((target >= low) & (target <= high))),
        "width_90": float(np.mean(high - low)),
        "interval_score_90": float(np.mean(interval_score)),
    }


def row(dataset: str, method: str, n: int, values: dict[str, float]) -> dict[str, Any]:
    return {"dataset": dataset, "method": method, "n": int(n), **values}


def nrel_table() -> list[dict[str, Any]]:
    base = pd.read_parquet(RESULTS / "nrel_physics_ladder_predictions.parquet")
    xgb = pd.read_parquet(RESULTS / "nrel_matched_xgboost_predictions.parquet")
    keys = ["state", "site_id", "LocalTime"]
    frame = base.merge(xgb[keys + ["xgb_q05", "xgb_q50", "xgb_q95"]], on=keys, validate="one_to_one")
    chronos_path = RESULTS / "nrel_chronos2_predictions.parquet"
    if chronos_path.exists():
        chronos = pd.read_parquet(chronos_path)
        columns = keys + [
            "chronos2_q050_projected",
            "chronos2_q500_projected",
            "chronos2_q950_projected",
        ]
        frame = frame.merge(chronos[columns], on=keys, validate="one_to_one")
    frame = frame.loc[frame["daylight"]].reset_index(drop=True)
    target = frame["target_power_normalized"].to_numpy(float)
    output = [
        row(
            "NREL known-site daylight",
            "Issued DA forecast",
            len(frame),
            point_metrics(target, frame["source_forecast_normalized"].to_numpy(float)),
        )
    ]
    labels = {
        "M1": "Non-physical direct LightGBM",
        "M2": "Solar-geometry LightGBM",
        "M3": "Clear-sky LightGBM",
        "M4": "Physical residual LightGBM",
        "M5": "Physical residual and trajectory LightGBM",
    }
    for prefix, label in labels.items():
        output.append(
            row(
                "NREL known-site daylight",
                label,
                len(frame),
                probabilistic_metrics(
                    target,
                    frame[f"{prefix}_q05"].to_numpy(float),
                    frame[f"{prefix}_q50"].to_numpy(float),
                    frame[f"{prefix}_q95"].to_numpy(float),
                ),
            )
        )
    output.append(
        row(
            "NREL known-site daylight",
            "Matched XGBoost quantile",
            len(frame),
            probabilistic_metrics(
                target,
                frame["xgb_q05"].to_numpy(float),
                frame["xgb_q50"].to_numpy(float),
                frame["xgb_q95"].to_numpy(float),
            ),
        )
    )
    if chronos_path.exists():
        output.append(
            row(
                "NREL known-site daylight",
                "Chronos-2 zero-shot, projected",
                len(frame),
                probabilistic_metrics(
                    target,
                    frame["chronos2_q050_projected"].to_numpy(float),
                    frame["chronos2_q500_projected"].to_numpy(float),
                    frame["chronos2_q950_projected"].to_numpy(float),
                ),
            )
        )
    return output


def pvod_table() -> list[dict[str, Any]]:
    base = pd.read_parquet(RESULTS / "pvod_q1_physics_ladder_predictions.parquet")
    keys = ["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min"]
    additions = (
        ("pvod_day_ahead_matched_baselines_predictions.parquet", None),
        ("pvod_xgboost_day_ahead_predictions.parquet", ["xgboost_prediction"]),
        ("pvod_patchtst_day_ahead_predictions.parquet", ["patchtst_prediction"]),
    )
    frame = base
    for filename, selected in additions:
        path = RESULTS / filename
        if path.exists():
            other = pd.read_parquet(path)
            columns = keys + (selected if selected is not None else [c for c in other.columns if c not in keys and c not in frame.columns])
            frame = frame.merge(other[columns], on=keys, validate="one_to_one")
    frame = frame.loc[frame["daylight"]].reset_index(drop=True)
    target = frame["target_power_normalized"].to_numpy(float)
    output: list[dict[str, Any]] = []
    point_columns = {
        "persistence_72h": "Three-day persistence",
        "clearsky_scaled_persistence_72h": "Clear-sky-scaled persistence",
        "xgboost_prediction": "Matched XGBoost point",
        "patchtst_prediction": "PatchTST-style trajectory model",
    }
    for column, label in point_columns.items():
        if column in frame:
            output.append(row("PVOD daylight common sample", label, len(frame), point_metrics(target, frame[column].to_numpy(float))))
    if "mlp_q50" in frame:
        output.append(
            row(
                "PVOD daylight common sample",
                "Quantile MLP",
                len(frame),
                probabilistic_metrics(target, frame["mlp_q05"].to_numpy(float), frame["mlp_q50"].to_numpy(float), frame["mlp_q95"].to_numpy(float)),
            )
        )
    labels = {
        "M1": "Non-physical direct LightGBM",
        "M2": "Solar-geometry LightGBM",
        "M3": "Clear-sky LightGBM",
        "M4": "Clear-sky residual LightGBM",
        "M5": "Residual trajectory LightGBM",
    }
    for prefix, label in labels.items():
        output.append(
            row(
                "PVOD daylight common sample",
                label,
                len(frame),
                probabilistic_metrics(target, frame[f"{prefix}_q05"].to_numpy(float), frame[f"{prefix}_q50"].to_numpy(float), frame[f"{prefix}_q95"].to_numpy(float)),
            )
        )
    return output


def emsx_table() -> list[dict[str, Any]]:
    base = pd.read_parquet(RESULTS / "emsx_sklearn_baselines_lead_96.parquet")
    keys = ["issue_time", "site_id"]
    candidates = (
        ("emsx_lstm_mlp_lead_96.parquet", ["mlp_prediction_kwh", "lstm_prediction_kwh"]),
        ("emsx_tcn_lead_96.parquet", ["tcn_prediction_kwh"]),
        ("emsx_gru_lead_96.parquet", ["gru_prediction_kwh"]),
        ("emsx_curve_random_forest_lead_96.parquet", ["curve_random_forest_prediction_kwh"]),
        ("emsx_xgboost_lead_96.parquet", ["xgboost_prediction_kwh"]),
        ("emsx_tuned_residual_qgbm_lead_96.parquet", ["tuned_residual_qgbm_prediction_kwh"]),
        ("emsx_enhanced_probabilistic_qgbm_lead_96.parquet", ["low_adaptive_kwh", "median_kwh", "high_adaptive_kwh"]),
    )
    frame = base
    for filename, columns in candidates:
        path = RESULTS / filename
        if path.exists():
            other = pd.read_parquet(path)
            frame = frame.merge(other[keys + columns], on=keys, validate="one_to_one")
    target = frame["actual_pv_kwh"].to_numpy(float)
    point_columns = {
        "forecast_pv_kwh": "Issued vendor forecast",
        "persistence_prediction_kwh": "Persistence",
        "ridge_prediction_kwh": "Ridge regression",
        "random_forest_prediction_kwh": "Random forest",
        "curve_random_forest_prediction_kwh": "Enhanced random forest",
        "mlp_prediction_kwh": "MLP",
        "lstm_prediction_kwh": "LSTM",
        "gru_prediction_kwh": "GRU",
        "tcn_prediction_kwh": "TCN",
        "xgboost_prediction_kwh": "Enhanced XGBoost",
        "tuned_residual_qgbm_prediction_kwh": "Enhanced LightGBM point",
    }
    output = [
        row("EMSx common test sample", label, len(frame), point_metrics(target, frame[column].to_numpy(float)))
        for column, label in point_columns.items()
        if column in frame
    ]
    if "median_kwh" in frame:
        output.append(
            row(
                "EMSx common test sample",
                "Enhanced probabilistic QGBM",
                len(frame),
                probabilistic_metrics(target, frame["low_adaptive_kwh"].to_numpy(float), frame["median_kwh"].to_numpy(float), frame["high_adaptive_kwh"].to_numpy(float)),
            )
        )
    return output


def value(item: dict[str, Any], key: str, digits: int) -> str:
    return "--" if key not in item else f"{item[key]:.{digits}f}"


def latex_table(rows: list[dict[str, Any]], label: str, caption: str, unit_digits: int) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & MAE & RMSE & sMAPE (\\%) & $R^2$ & $\\mathrm{CRPS}_{3q}$ & Coverage & IS \\\\",
        "\\midrule",
    ]
    for item in rows:
        lines.append(
            f"{item['method']} & {value(item, 'mae', unit_digits)} & {value(item, 'rmse', unit_digits)} & "
            f"{value(item, 'smape_percent', 2)} & {value(item, 'r2', 4)} & {value(item, 'crps_3q', unit_digits)} & "
            f"{value(item, 'coverage_90', 4)} & {value(item, 'interval_score_90', unit_digits)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    groups = {
        "nrel": nrel_table(),
        "pvod": pvod_table(),
        "emsx": emsx_table(),
    }
    all_rows = [item for values in groups.values() for item in values]
    RESULTS.mkdir(exist_ok=True)
    PAPER.mkdir(exist_ok=True)
    (RESULTS / "all_method_metrics.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="ascii")
    pd.DataFrame(all_rows).to_csv(RESULTS / "all_method_metrics.csv", index=False)
    tex = "\n".join(
        [
            latex_table(
                groups["nrel"],
                "tab:all_nrel_methods",
                "Complete NREL known-site daylight comparison on a common test sample. CRPS$_{3q}$ uses the common 0.05, 0.50, and 0.95 quantiles with constant-tail trapezoidal integration. Dashes denote point forecasts without a predictive distribution.",
                5,
            ),
            latex_table(
                groups["pvod"],
                "tab:all_pvod_methods",
                "Complete PVOD daylight comparison on the intersection of test records available to every sequence and tabular model.",
                5,
            ),
            latex_table(
                groups["emsx"],
                "tab:all_emsx_methods",
                "Complete EMSx comparison on 851,958 common test records. Errors and interval score are expressed in kWh.",
                2,
            ),
        ]
    )
    (PAPER / "all_method_results_tables.tex").write_text(tex, encoding="ascii")
    print(json.dumps({name: len(values) for name, values in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
