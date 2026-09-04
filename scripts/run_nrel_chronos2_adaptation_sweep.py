#!/usr/bin/env python3
"""Run a reproducible Chronos-2 adaptation-budget and optimization sweep."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from run_nrel_chronos2 import (
    COVARIATES,
    MODEL_ID,
    QUANTILES,
    conformal_radius,
    make_tasks,
    predict,
    project,
    subset_metrics,
)
from run_nrel_chronos2_fewshot import adaptation_sequences


BUDGETS = (1, 3, 7, 14, 30)
SEEDS = (20260829, 20260830, 20260831)


def run_specs(include_full: bool) -> list[dict]:
    specs = [
        {"days": days, "seed": seed, "mode": "lora", "lr": 1e-5, "rank": 8, "steps": 300}
        for days in BUDGETS
        for seed in SEEDS
    ]
    specs.extend(
        {"days": 7, "seed": SEEDS[0], "mode": "lora", "lr": lr, "rank": 8, "steps": 300}
        for lr in (3e-6, 3e-5)
    )
    specs.extend(
        {"days": 7, "seed": SEEDS[0], "mode": "lora", "lr": 1e-5, "rank": rank, "steps": 300}
        for rank in (4, 16)
    )
    if include_full:
        specs.extend(
            {"days": 30, "seed": seed, "mode": "full", "lr": 1e-6, "rank": None, "steps": 300}
            for seed in SEEDS
        )
    unique = {spec_id(spec): spec for spec in specs}
    return list(unique.values())


def spec_id(spec: dict) -> str:
    lr_text = f"{spec['lr']:.0e}".replace("-", "m")
    rank_text = "all" if spec["rank"] is None else str(spec["rank"])
    return f"{spec['mode']}_d{spec['days']}_s{spec['seed']}_lr{lr_text}_r{rank_text}_n{spec['steps']}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    pipeline: Chronos2Pipeline,
    calibration_inputs: list,
    calibration_frames: list[pd.DataFrame],
    test_inputs: list,
    test_frames: list[pd.DataFrame],
    context_hours: int,
    label: str,
) -> tuple[dict, pd.DataFrame]:
    started = time.perf_counter()
    calibration_raw = predict(pipeline, calibration_inputs, 256, 256, context_hours, f"{label} calibration")
    test_raw = predict(pipeline, test_inputs, 256, 256, context_hours, f"{label} test")
    inference_seconds = time.perf_counter() - started
    calibration_frame = pd.concat(calibration_frames, ignore_index=True)
    test_frame = pd.concat(test_frames, ignore_index=True)
    calibration_projected = project(calibration_frame, calibration_raw)
    test_projected = project(test_frame, test_raw)
    cal_day = calibration_frame["daylight"].to_numpy(bool)
    radius = conformal_radius(
        calibration_frame.loc[cal_day, "target_power_normalized"].to_numpy(float),
        calibration_projected[cal_day],
    )
    conformed = test_projected.copy()
    test_day = test_frame["daylight"].to_numpy(bool)
    conformed[test_day, QUANTILES.index(0.05)] = np.clip(
        conformed[test_day, QUANTILES.index(0.05)] - radius, 0.0, 1.2
    )
    conformed[test_day, QUANTILES.index(0.95)] = np.clip(
        conformed[test_day, QUANTILES.index(0.95)] + radius, 0.0, 1.2
    )
    conformed = np.sort(conformed, axis=1)
    prediction = test_frame[
        ["state", "site_id", "LocalTime", "delivery_date", "delivery_hour", "daylight", "target_power_normalized"]
    ].reset_index(drop=True)
    for column, quantile in enumerate(QUANTILES):
        prediction[f"q{int(round(1000 * quantile)):03d}"] = conformed[:, column].astype(np.float32)
    result = {
        "inference_seconds": inference_seconds,
        "inference_seconds_per_site_day": inference_seconds / max(1, len(calibration_inputs) + len(test_inputs)),
        "conformal_radius": radius,
        "raw": subset_metrics(test_frame, test_raw),
        "projected": subset_metrics(test_frame, test_projected),
        "projected_conformal": subset_metrics(test_frame, conformed),
    }
    return result, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--only", default=None, help="Run only one exact specification identifier")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = root / "results/chronos2_adaptation_sweep"
    checkpoint_root = root / "checkpoints/chronos2_adaptation_sweep"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    design = json.loads((root / "configs/nrel_q1_study.json").read_text())
    split = design["known_site_temporal_split"]
    frame = pd.read_parquet(root / "data/processed/nrel_q1_panel.parquet")
    frame["LocalTime"] = pd.to_datetime(frame["LocalTime"])
    context_hours = 336
    calibration_inputs, calibration_frames = make_tasks(
        frame, pd.Timestamp(split["calibration_start"]), pd.Timestamp(split["calibration_end"]), context_hours
    )
    test_inputs, test_frames = make_tasks(
        frame, pd.Timestamp(split["test_start"]), frame["LocalTime"].max(), context_hours
    )
    base = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cuda")
    completed = []
    sequence_cache: dict[int, tuple[list, list]] = {}
    for spec in run_specs(args.include_full):
        identifier = spec_id(spec)
        if args.only and identifier != args.only:
            continue
        result_path = output_root / f"{identifier}.json"
        prediction_path = output_root / f"{identifier}.parquet"
        if result_path.exists() and prediction_path.exists():
            print(f"SKIP {identifier}", flush=True)
            completed.append(identifier)
            continue
        print(f"START {identifier}", flush=True)
        set_seed(spec["seed"])
        if spec["days"] not in sequence_cache:
            sequence_cache[spec["days"]] = (
                adaptation_sequences(frame, pd.Timestamp(split["train_end"]), context_hours, spec["days"]),
                adaptation_sequences(frame, pd.Timestamp(split["tuning_end"]), context_hours, spec["days"]),
            )
        train_inputs, validation_inputs = sequence_cache[spec["days"]]
        checkpoint_dir = checkpoint_root / identifier
        lora_config = None
        if spec["mode"] == "lora":
            lora_config = {
                "r": spec["rank"],
                "lora_alpha": 2 * spec["rank"],
                "target_modules": [
                    "self_attention.q",
                    "self_attention.v",
                    "self_attention.k",
                    "self_attention.o",
                    "output_patch_embedding.output_layer",
                ],
            }
        fit_started = time.perf_counter()
        adapted = base.fit(
            inputs=train_inputs,
            validation_inputs=validation_inputs,
            prediction_length=24,
            finetune_mode=spec["mode"],
            lora_config=lora_config,
            context_length=context_hours,
            learning_rate=spec["lr"],
            num_steps=spec["steps"],
            batch_size=32 if spec["mode"] == "lora" else 4,
            min_past=context_hours,
            output_dir=checkpoint_dir,
            logging_steps=25,
            save_steps=100,
            eval_steps=100,
            seed=spec["seed"],
            data_seed=spec["seed"],
            gradient_accumulation_steps=1 if spec["mode"] == "lora" else 8,
        )
        fit_seconds = time.perf_counter() - fit_started
        evaluation, prediction = evaluate(
            adapted, calibration_inputs, calibration_frames, test_inputs, test_frames, context_hours, identifier
        )
        checkpoint_bytes = sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "identifier": identifier,
            "model_id": MODEL_ID,
            "specification": spec,
            "training_examples": len(train_inputs),
            "validation_examples": len(validation_inputs),
            "sites": int(frame[["state", "site_id"]].drop_duplicates().shape[0]),
            "context_hours": context_hours,
            "checkpoint_selection": "lowest validation loss among steps 100, 200, and 300",
            "fit_seconds": fit_seconds,
            "checkpoint_bytes": checkpoint_bytes,
            **evaluation,
        }
        prediction.to_parquet(prediction_path, index=False, compression="zstd")
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print("DONE", identifier, json.dumps(result["projected_conformal"]["daylight"]), flush=True)
        completed.append(identifier)
        del adapted, prediction
        gc.collect()
        torch.cuda.empty_cache()
    (output_root / "manifest.json").write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "completed": completed}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
