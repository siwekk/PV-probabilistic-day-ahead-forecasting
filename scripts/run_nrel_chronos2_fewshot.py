#!/usr/bin/env python3
"""Evaluate seven-day-per-site LoRA adaptation of Chronos-2 on NREL."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

from run_nrel_chronos2 import COVARIATES, MODEL_ID, QUANTILES, conformal_radius, make_tasks, predict, project, subset_metrics


def adaptation_sequences(frame: pd.DataFrame, end: pd.Timestamp, context_hours: int, supervised_days: int) -> list[dict]:
    inputs = []
    for _, site in frame.groupby(["state", "site_id"], observed=True, sort=True):
        site = site.sort_values("LocalTime")
        available = site.loc[pd.to_datetime(site["LocalTime"]) <= end].copy()
        future_days = sorted(available["delivery_date"].unique())[-supervised_days:]
        if len(future_days) != supervised_days:
            continue
        for delivery_date in future_days:
            future = available.loc[available["delivery_date"] == delivery_date].sort_values("LocalTime")
            if len(future) != 24:
                continue
            forecast_start = pd.Timestamp(future["LocalTime"].iloc[0])
            context = available.loc[pd.to_datetime(available["LocalTime"]) < forecast_start].tail(context_hours)
            selected = pd.concat([context, future], ignore_index=True)
            if len(context) != context_hours or pd.to_datetime(selected["LocalTime"]).diff().dropna().ne(pd.Timedelta(hours=1)).any():
                continue
            required = ["target_power_normalized", *COVARIATES]
            if selected[required].isna().any().any():
                continue
            inputs.append({
                "target": selected["target_power_normalized"].to_numpy(np.float32),
                "past_covariates": {name: selected[name].to_numpy(np.float32) for name in COVARIATES},
                "future_covariates": {name: future[name].to_numpy(np.float32) for name in COVARIATES},
            })
    return inputs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    design = json.loads((root / "configs/nrel_q1_study.json").read_text())
    split = design["known_site_temporal_split"]
    frame = pd.read_parquet(root / "data/processed/nrel_q1_panel.parquet")
    frame["LocalTime"] = pd.to_datetime(frame["LocalTime"])
    context_hours, supervised_days = 336, 7
    train_inputs = adaptation_sequences(frame, pd.Timestamp(split["train_end"]), context_hours, supervised_days)
    validation_inputs = adaptation_sequences(frame, pd.Timestamp(split["tuning_end"]), context_hours, supervised_days)
    calibration_inputs, calibration_frames = make_tasks(frame, pd.Timestamp(split["calibration_start"]), pd.Timestamp(split["calibration_end"]), context_hours)
    test_inputs, test_frames = make_tasks(frame, pd.Timestamp(split["test_start"]), frame["LocalTime"].max(), context_hours)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cuda")
    output_dir = root / "checkpoints/chronos2_nrel_7day_lora"
    started = time.perf_counter()
    adapted = pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs,
        prediction_length=24,
        finetune_mode="lora",
        context_length=context_hours,
        learning_rate=1e-5,
        num_steps=300,
        batch_size=32,
        min_past=context_hours,
        output_dir=output_dir,
        logging_steps=25,
        save_steps=100,
        eval_steps=100,
    )
    fit_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    calibration_raw = predict(adapted, calibration_inputs, 256, 256, context_hours, "fewshot calibration")
    test_raw = predict(adapted, test_inputs, 256, 256, context_hours, "fewshot test")
    inference_seconds = time.perf_counter() - inference_started
    calibration_frame = pd.concat(calibration_frames, ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)
    calibration_projected = project(calibration_frame, calibration_raw)
    test_projected = project(test_frame, test_raw)
    cal_day = calibration_frame["daylight"].to_numpy(bool)
    radius = conformal_radius(calibration_frame.loc[cal_day, "target_power_normalized"].to_numpy(float), calibration_projected[cal_day])
    conformed = test_projected.copy()
    test_day = test_frame["daylight"].to_numpy(bool)
    conformed[test_day, QUANTILES.index(0.05)] = np.clip(conformed[test_day, QUANTILES.index(0.05)] - radius, 0.0, 1.2)
    conformed[test_day, QUANTILES.index(0.95)] = np.clip(conformed[test_day, QUANTILES.index(0.95)] + radius, 0.0, 1.2)
    conformed = np.sort(conformed, axis=1)
    checkpoint_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "Chronos-2 LoRA adaptation with seven supervised days per NREL site",
        "model_id": MODEL_ID,
        "adaptation_contract": {"sites": len(train_inputs), "supervised_days_per_site": supervised_days, "context_hours": context_hours, "steps": 300, "learning_rate": 1e-5, "batch_size": 32},
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_site_day": inference_seconds / max(1, len(calibration_inputs) + len(test_inputs)),
        "checkpoint_bytes": checkpoint_bytes,
        "conformal_radius": radius,
        "raw": subset_metrics(test_frame, test_raw),
        "projected": subset_metrics(test_frame, test_projected),
        "projected_conformal": subset_metrics(test_frame, conformed),
    }
    (root / "results/nrel_chronos2_7day_lora.json").write_text(json.dumps(result, indent=2) + "\n")
    prediction = test_frame[["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized"]].reset_index(drop=True)
    for column, quantile in enumerate(QUANTILES):
        prediction[f"chronos2_lora_q{int(round(1000 * quantile)):03d}"] = conformed[:, column]
    prediction.to_parquet(root / "results/nrel_chronos2_7day_lora_predictions.parquet", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
