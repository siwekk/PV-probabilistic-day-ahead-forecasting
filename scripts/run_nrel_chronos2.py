#!/usr/bin/env python3
"""Evaluate zero-shot Chronos-2 on the frozen NREL temporal experiment."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline


MODEL_ID = "amazon/chronos-2"
QUANTILES = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
COVARIATES = (
    "source_forecast_normalized",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "clear_sky_ghi_w_m2",
    "clear_sky_envelope",
)


def pinball(target: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    error = target - prediction
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    median = prediction[:, QUANTILES.index(0.50)]
    low = prediction[:, QUANTILES.index(0.05)]
    high = prediction[:, QUANTILES.index(0.95)]
    error = target - median
    denominator = np.abs(target) + np.abs(median)
    smape_terms = np.divide(
        2.0 * np.abs(error),
        denominator,
        out=np.zeros_like(error),
        where=denominator > 0,
    )
    target_variation = np.sum(np.square(target - target.mean()))
    interval_score = (
        high
        - low
        + 20.0 * (low - target) * (target < low)
        + 20.0 * (target - high) * (target > high)
    )
    losses = np.column_stack(
        [
            np.maximum(q * (target - prediction[:, column]), (q - 1.0) * (target - prediction[:, column]))
            for column, q in enumerate(QUANTILES)
        ]
    )
    crps = 2.0 * np.trapz(losses, x=np.asarray(QUANTILES), axis=1)
    return {
        "n": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "smape_percent": float(100.0 * np.mean(smape_terms)),
        "r2": float(1.0 - np.sum(np.square(error)) / target_variation),
        "mean_pinball": float(np.mean([pinball(target, prediction[:, i], q) for i, q in enumerate(QUANTILES)])),
        "crps_approx": float(np.mean(crps)),
        "coverage_90": float(np.mean((target >= low) & (target <= high))),
        "width_90": float(np.mean(high - low)),
        "interval_score_90": float(np.mean(interval_score)),
    }


def project(frame: pd.DataFrame, prediction: np.ndarray) -> np.ndarray:
    output = np.sort(np.clip(prediction, 0.0, 1.2), axis=1)
    night = ~frame["daylight"].to_numpy(dtype=bool)
    output[night, :] = 0.0
    return output


def conformal_radius(target: np.ndarray, prediction: np.ndarray) -> float:
    low = prediction[:, QUANTILES.index(0.05)]
    high = prediction[:, QUANTILES.index(0.95)]
    nonconformity = np.maximum.reduce([low - target, target - high, np.zeros(len(target))])
    level = min(1.0, np.ceil((len(target) + 1) * 0.90) / len(target))
    return float(np.quantile(nonconformity, level, method="higher"))


def make_tasks(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    context_hours: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    inputs: list[dict[str, Any]] = []
    futures: list[pd.DataFrame] = []
    for (_, _), site in frame.groupby(["state", "site_id"], observed=True, sort=True):
        site = site.sort_values("LocalTime").reset_index(drop=True)
        site_time = pd.to_datetime(site["LocalTime"])
        selected = site.loc[(site_time >= start) & (site_time <= end)]
        for _, future in selected.groupby("delivery_date", observed=True, sort=True):
            future = future.sort_values("LocalTime")
            if len(future) != 24:
                continue
            forecast_start = pd.Timestamp(future["LocalTime"].iloc[0])
            context = site.loc[site_time < forecast_start].tail(context_hours)
            if len(context) != context_hours:
                continue
            context_time = pd.to_datetime(context["LocalTime"])
            if context_time.diff().dropna().ne(pd.Timedelta(hours=1)).any():
                continue
            required = ["target_power_normalized", *COVARIATES]
            if context[required].isna().any().any() or future[[*COVARIATES]].isna().any().any():
                continue
            inputs.append(
                {
                    "target": context["target_power_normalized"].to_numpy(dtype=np.float32),
                    "past_covariates": {
                        name: context[name].to_numpy(dtype=np.float32) for name in COVARIATES
                    },
                    "future_covariates": {
                        name: future[name].to_numpy(dtype=np.float32) for name in COVARIATES
                    },
                }
            )
            futures.append(future)
    return inputs, futures


def predict(
    pipeline: Chronos2Pipeline,
    inputs: list[dict[str, Any]],
    chunk_size: int,
    model_batch_size: int,
    context_hours: int,
    label: str,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(inputs), chunk_size):
        chunk = inputs[start : start + chunk_size]
        quantiles, _ = pipeline.predict_quantiles(
            inputs=chunk,
            prediction_length=24,
            quantile_levels=list(QUANTILES),
            batch_size=model_batch_size,
            context_length=context_hours,
        )
        for forecast in quantiles:
            array = forecast.detach().cpu().numpy() if torch.is_tensor(forecast) else np.asarray(forecast)
            output.append(np.squeeze(array, axis=0))
        print(f"{label}: {min(start + len(chunk), len(inputs))}/{len(inputs)} tasks", flush=True)
    return np.concatenate(output, axis=0)


def subset_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    daylight = frame["daylight"].to_numpy(dtype=bool)
    target = frame["target_power_normalized"].to_numpy(dtype=float)
    return {
        "overall": metrics(target, prediction),
        "daylight": metrics(target[daylight], prediction[daylight]),
        "night": metrics(target[~daylight], prediction[~daylight]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--context-hours", type=int, default=336)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--model-batch-size", type=int, default=256)
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    split = design["known_site_temporal_split"]
    frame = pd.read_parquet(root / "data" / "processed" / "nrel_q1_panel.parquet")
    frame["LocalTime"] = pd.to_datetime(frame["LocalTime"])
    calibration_inputs, calibration_frames = make_tasks(
        frame,
        pd.Timestamp(split["calibration_start"]),
        pd.Timestamp(split["calibration_end"]),
        args.context_hours,
    )
    test_inputs, test_frames = make_tasks(
        frame,
        pd.Timestamp(split["test_start"]),
        frame["LocalTime"].max(),
        args.context_hours,
    )
    print(
        f"prepared calibration={len(calibration_inputs)} and test={len(test_inputs)} daily tasks",
        flush=True,
    )
    started = time.perf_counter()
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cuda")
    loaded_seconds = time.perf_counter() - started
    calibration_raw = predict(
        pipeline,
        calibration_inputs,
        args.chunk_size,
        args.model_batch_size,
        args.context_hours,
        "calibration",
    )
    test_raw = predict(
        pipeline,
        test_inputs,
        args.chunk_size,
        args.model_batch_size,
        args.context_hours,
        "test",
    )
    calibration_frame = pd.concat(calibration_frames, ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)
    calibration_projected = project(calibration_frame, calibration_raw)
    test_projected = project(test_frame, test_raw)
    calibration_daylight = calibration_frame["daylight"].to_numpy(dtype=bool)
    radius = conformal_radius(
        calibration_frame.loc[calibration_daylight, "target_power_normalized"].to_numpy(dtype=float),
        calibration_projected[calibration_daylight],
    )
    test_conformal = test_projected.copy()
    test_daylight = test_frame["daylight"].to_numpy(dtype=bool)
    low_index = QUANTILES.index(0.05)
    high_index = QUANTILES.index(0.95)
    test_conformal[test_daylight, low_index] = np.clip(
        test_conformal[test_daylight, low_index] - radius, 0.0, 1.2
    )
    test_conformal[test_daylight, high_index] = np.clip(
        test_conformal[test_daylight, high_index] + radius, 0.0, 1.2
    )
    test_conformal = np.sort(test_conformal, axis=1)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Chronos-2 zero-shot covariate-informed quantile forecast",
        "model_id": MODEL_ID,
        "model_commit": getattr(pipeline.model.config, "_commit_hash", None),
        "chronos_version": __import__("importlib.metadata").metadata.version("chronos-forecasting"),
        "device": "cuda",
        "context_hours": args.context_hours,
        "prediction_hours": 24,
        "quantiles": QUANTILES,
        "covariates": COVARIATES,
        "tasks": {"calibration": len(calibration_inputs), "test": len(test_inputs)},
        "load_seconds": loaded_seconds,
        "total_seconds": time.perf_counter() - started,
        "conformal_radius": radius,
        "raw": subset_metrics(test_frame, test_raw),
        "projected": subset_metrics(test_frame, test_projected),
        "projected_conformal": subset_metrics(test_frame, test_conformal),
    }
    (root / "results" / "nrel_chronos2.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    prediction_frame = test_frame[
        [
            "state",
            "site_id",
            "LocalTime",
            "delivery_date",
            "delivery_hour",
            "daylight",
            "target_power_normalized",
            "source_forecast_normalized",
        ]
    ].reset_index(drop=True)
    for column, quantile in enumerate(QUANTILES):
        label = f"q{int(round(1000 * quantile)):03d}"
        prediction_frame[f"chronos2_{label}_raw"] = test_raw[:, column]
        prediction_frame[f"chronos2_{label}_projected"] = test_projected[:, column]
        prediction_frame[f"chronos2_{label}_conformal"] = test_conformal[:, column]
    prediction_frame.to_parquet(root / "results" / "nrel_chronos2_predictions.parquet", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
