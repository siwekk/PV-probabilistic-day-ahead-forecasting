#!/usr/bin/env python3
"""Tune PVOD MLP and PatchTST-style baselines, then ensemble three final seeds."""

from __future__ import annotations

import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_pvod_patchtst import build_sequences


ROOT = Path(__file__).resolve().parents[1]
TUNING_SEED = 20260901
FINAL_SEEDS = (20260901, 20260902, 20260903)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class QuantileMLP(nn.Module):
    def __init__(self, features: int, hidden: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sort(self.layers(values), dim=1).values


class PatchModel(nn.Module):
    def __init__(self, features: int, width: int, layers: int) -> None:
        super().__init__()
        self.patch = nn.Conv1d(features, width, kernel_size=8, stride=8)
        self.position = nn.Parameter(torch.zeros(1, 12, width))
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=2 * width,
            dropout=0.10,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(width, 8)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(values.transpose(1, 2)).transpose(1, 2) + self.position
        return self.head(self.encoder(tokens)).reshape(values.shape[0], 96)


def pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.05, 0.50, 0.95], device=prediction.device)
    error = target[:, None] - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def fit_mlp(
    arrays: dict[str, np.ndarray],
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[float, int, np.ndarray]:
    set_seed(seed)
    model = QuantileMLP(arrays["x_train"].shape[1], tuple(config["hidden"]), config["dropout"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays["x_train"]), torch.from_numpy(arrays["y_train"])),
        batch_size=4096,
        shuffle=True,
        generator=generator,
    )
    x_val = torch.from_numpy(arrays["x_validation"]).to(device)
    y_val = torch.from_numpy(arrays["y_validation"]).to(device)
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(100):
        model.train()
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = pinball(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            value = float(pinball(model(x_val), y_val).item())
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(arrays["x_test"]).to(device)).cpu().numpy()
    return best_loss, epoch + 1, np.clip(prediction, 0.0, 1.2)


def fit_patch(
    arrays: dict[str, np.ndarray],
    config: dict,
    seed: int,
    device: torch.device,
) -> tuple[float, int, np.ndarray]:
    set_seed(seed)
    model = PatchModel(arrays["x_train"].shape[2], config["width"], config["layers"]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arrays["x_train"]), torch.from_numpy(arrays["y_train"])),
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    x_val = torch.from_numpy(arrays["x_validation"]).to(device)
    y_val = torch.from_numpy(arrays["y_validation"]).to(device)
    best_loss = float("inf")
    best_state = None
    stale = 0
    for epoch in range(60):
        model.train()
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = torch.nn.functional.l1_loss(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            value = float(torch.nn.functional.l1_loss(model(x_val), y_val).item())
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(arrays["x_test"]).to(device)).cpu().numpy()
    return best_loss, epoch + 1, np.clip(prediction, 0.0, 1.2)


def point_summary(target: np.ndarray, prediction: np.ndarray, daylight: np.ndarray) -> dict:
    loss = np.abs(target - prediction)
    return {
        "mae_all": float(loss.mean()),
        "mae_daylight": float(loss[daylight].mean()),
        "seed_mae_daylight_sd": None,
    }


def main() -> None:
    started = time.perf_counter()
    design = json.loads((ROOT / "configs" / "pvod_day_ahead_study.json").read_text())
    frame = pd.read_parquet(ROOT / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    split = design["temporal_split_target_utc"]
    features = ["station_code", *design["future_features"], *design["origin_features"]]
    train = frame.loc[frame["target_timestamp_utc"] <= pd.Timestamp(split["train_end"])].dropna(
        subset=features + ["target_power_normalized"]
    )
    validation = frame.loc[
        (frame["target_timestamp_utc"] >= pd.Timestamp(split["calibration_start"]))
        & (frame["target_timestamp_utc"] <= pd.Timestamp(split["calibration_end"]))
    ].dropna(subset=features + ["target_power_normalized"])
    test = frame.loc[frame["target_timestamp_utc"] >= pd.Timestamp(split["test_start"])].dropna(
        subset=features + ["target_power_normalized"]
    )
    scaler = StandardScaler().fit(train[features].astype(float))
    mlp_arrays = {
        "x_train": scaler.transform(train[features].astype(float)).astype(np.float32),
        "y_train": train["target_power_normalized"].to_numpy(np.float32),
        "x_validation": scaler.transform(validation[features].astype(float)).astype(np.float32),
        "y_validation": validation["target_power_normalized"].to_numpy(np.float32),
        "x_test": scaler.transform(test[features].astype(float)).astype(np.float32),
        "y_test": test["target_power_normalized"].to_numpy(np.float32),
    }

    trajectory_features = [*design["future_features"], *design["origin_features"]]
    sequences = build_sequences(frame, trajectory_features, split)
    groups = {name: [item for item in sequences if item[0] == name] for name in ("train", "validation", "test")}
    trajectory_scaler = StandardScaler().fit(np.concatenate([item[1] for item in groups["train"]], axis=0))
    patch_arrays = {}
    for name, values in groups.items():
        patch_arrays[f"x_{name}"] = np.stack(
            [trajectory_scaler.transform(item[1]).astype(np.float32) for item in values]
        )
        patch_arrays[f"y_{name}"] = np.stack([item[2].astype(np.float32) for item in values])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp_grid = [
        {"hidden": list(hidden), "dropout": dropout, "learning_rate": learning_rate}
        for hidden in ((64, 32), (128, 64), (256, 128))
        for dropout in (0.0, 0.1)
        for learning_rate in (3e-4, 1e-3)
    ]
    patch_grid = [
        {"width": width, "layers": layers, "learning_rate": learning_rate}
        for width in (32, 64)
        for layers in (1, 2)
        for learning_rate in (3e-4, 1e-3)
    ]

    mlp_search = []
    for config in mlp_grid:
        loss, epochs, _ = fit_mlp(mlp_arrays, config, TUNING_SEED, device)
        mlp_search.append({"config": config, "validation_pinball": loss, "epochs": epochs})
        print("MLP", mlp_search[-1], flush=True)
    patch_search = []
    for config in patch_grid:
        loss, epochs, _ = fit_patch(patch_arrays, config, TUNING_SEED, device)
        patch_search.append({"config": config, "validation_mae": loss, "epochs": epochs})
        print("PATCH", patch_search[-1], flush=True)
    best_mlp = min(mlp_search, key=lambda item: item["validation_pinball"])["config"]
    best_patch = min(patch_search, key=lambda item: item["validation_mae"])["config"]

    mlp_predictions = []
    patch_predictions = []
    final_runs = {"mlp": [], "patchtst_style": []}
    for seed in FINAL_SEEDS:
        loss, epochs, prediction = fit_mlp(mlp_arrays, best_mlp, seed, device)
        mlp_predictions.append(prediction)
        final_runs["mlp"].append({"seed": seed, "validation_pinball": loss, "epochs": epochs})
        loss, epochs, prediction = fit_patch(patch_arrays, best_patch, seed, device)
        patch_predictions.append(prediction)
        final_runs["patchtst_style"].append({"seed": seed, "validation_mae": loss, "epochs": epochs})

    mlp_seed_array = np.stack(mlp_predictions)
    patch_seed_array = np.stack(patch_predictions)
    mlp_ensemble = np.sort(mlp_seed_array.mean(axis=0), axis=1)
    patch_ensemble = patch_seed_array.mean(axis=0)
    mlp_daylight = test["clearsky_ghi"].to_numpy(float) >= 20.0
    patch_daylight = np.stack([item[3] for item in groups["test"]]) >= 20.0
    mlp_summary = point_summary(mlp_arrays["y_test"], mlp_ensemble[:, 1], mlp_daylight)
    patch_summary = point_summary(patch_arrays["y_test"], patch_ensemble, patch_daylight)
    mlp_summary["seed_mae_daylight_sd"] = float(
        np.std([np.mean(np.abs(mlp_arrays["y_test"][mlp_daylight] - value[:, 1][mlp_daylight])) for value in mlp_seed_array], ddof=1)
    )
    patch_summary["seed_mae_daylight_sd"] = float(
        np.std([np.mean(np.abs(patch_arrays["y_test"][patch_daylight] - value[patch_daylight])) for value in patch_seed_array], ddof=1)
    )

    mlp_output = test[
        ["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min", "clearsky_ghi", "target_power_normalized", "persistence_72h", "clearsky_scaled_persistence_72h"]
    ].reset_index(drop=True)
    mlp_output[["mlp_q05", "mlp_q50", "mlp_q95"]] = mlp_ensemble
    mlp_output[["mlp_q05_conformal", "mlp_q50_conformal", "mlp_q95_conformal"]] = mlp_ensemble
    mlp_output.to_parquet(ROOT / "results" / "pvod_day_ahead_matched_baselines_predictions.parquet", index=False)

    patch_output = pd.concat([item[4] for item in groups["test"]], ignore_index=True)
    patch_output["patchtst_prediction"] = patch_ensemble.reshape(-1)
    patch_output.to_parquet(ROOT / "results" / "pvod_patchtst_day_ahead_predictions.parquet", index=False)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "selection_rule": "Minimum validation loss at the best checkpoint; test targets were evaluated only after selection.",
        "mlp": {
            "grid": mlp_search,
            "selected": best_mlp,
            "early_stopping": {"maximum_epochs": 100, "patience": 12, "minimum_improvement": 1e-6},
            "final_runs": final_runs["mlp"],
            "ensemble_test": mlp_summary,
        },
        "patchtst_style": {
            "grid": patch_search,
            "selected": best_patch,
            "early_stopping": {"maximum_epochs": 60, "patience": 10, "minimum_improvement": 1e-6},
            "final_runs": final_runs["patchtst_style"],
            "ensemble_test": patch_summary,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (ROOT / "results" / "pvod_neural_baseline_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
