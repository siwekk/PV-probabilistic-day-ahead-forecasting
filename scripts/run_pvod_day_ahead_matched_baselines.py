"""Evaluate valid persistence baselines and a compact quantile MLP on PVOD day-ahead data."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def point_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    y = frame["target_power_normalized"].to_numpy()
    daylight = frame["clearsky_ghi"].to_numpy() >= 20.0
    return {"n": int(len(frame)), "mae_all": float(np.mean(np.abs(y - prediction))), "n_daylight": int(daylight.sum()), "mae_daylight": float(np.mean(np.abs(y[daylight] - prediction[daylight])))}


def interval_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    low, median, high = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    values = point_metrics(frame, median)
    y = frame["target_power_normalized"].to_numpy()
    values.update({"coverage_90": float(np.mean((y >= low) & (y <= high))), "width_90": float(np.mean(high - low))})
    return values


class QuantileMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.10), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sort(self.layers(x), dim=1).values


def pinball(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    quantiles = torch.tensor([0.05, 0.50, 0.95], device=prediction.device)
    error = target[:, None] - prediction
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_day_ahead_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    split = design["temporal_split_target_utc"]
    train = frame.loc[frame["target_timestamp_utc"] <= pd.Timestamp(split["train_end"])]
    calibration = frame.loc[(frame["target_timestamp_utc"] >= pd.Timestamp(split["calibration_start"])) & (frame["target_timestamp_utc"] <= pd.Timestamp(split["calibration_end"]))]
    test = frame.loc[frame["target_timestamp_utc"] >= pd.Timestamp(split["test_start"])]
    base_mask = test["target_power_normalized"].notna() & test["persistence_72h"].notna() & test["clearsky_scaled_persistence_72h"].notna()
    baselines = {"three_day_persistence": test.loc[base_mask, "persistence_72h"].to_numpy(), "clearsky_scaled_three_day_persistence": test.loc[base_mask, "clearsky_scaled_persistence_72h"].to_numpy()}
    result: dict[str, object] = {"study": design["name"], "restriction": design["restriction"], "baselines": {name: point_metrics(test.loc[base_mask], prediction) for name, prediction in baselines.items()}}

    features = ["station_code"] + design["future_features"] + design["origin_features"]
    fit = train.dropna(subset=features + ["target_power_normalized"])
    cal = calibration.dropna(subset=features + ["target_power_normalized"])
    held = test.dropna(subset=features + ["target_power_normalized"])
    scaler = StandardScaler().fit(fit[features].astype(float))
    x_train = scaler.transform(fit[features].astype(float)).astype(np.float32); y_train = fit["target_power_normalized"].to_numpy(np.float32)
    x_cal = scaler.transform(cal[features].astype(float)).astype(np.float32); x_test = scaler.transform(held[features].astype(float)).astype(np.float32)
    torch.manual_seed(20260826); np.random.seed(20260826); random.seed(20260826)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuantileMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=4096, shuffle=True, generator=torch.Generator().manual_seed(20260826))
    best_state = None; best_loss = float("inf"); patience = 0
    cal_tensor = torch.from_numpy(x_cal).to(device)
    for epoch in range(100):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(); loss = pinball(model(x_batch.to(device)), y_batch.to(device)); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(pinball(model(cal_tensor), torch.from_numpy(cal["target_power_normalized"].to_numpy(np.float32)).to(device)).item())
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred_cal = np.clip(model(cal_tensor).cpu().numpy(), 0.0, 1.2)
        pred_test = np.clip(model(torch.from_numpy(x_test).to(device)).cpu().numpy(), 0.0, 1.2)
    y_cal = cal["target_power_normalized"].to_numpy()
    radius = float(np.quantile(np.maximum.reduce([pred_cal[:, 0] - y_cal, y_cal - pred_cal[:, 2], np.zeros(len(cal))]), 0.9, method="higher"))
    calibrated = pred_test.copy(); calibrated[:, 0] = np.clip(calibrated[:, 0] - radius, 0.0, None); calibrated[:, 2] += radius
    result["quantile_mlp"] = {"device": str(device), "epochs_completed": epoch + 1, "best_calibration_pinball": best_loss, "n_train": int(len(fit)), "n_calibration": int(len(cal)), "n_test": int(len(held)), "features": features, "conformal_radius": radius, "raw_test": interval_metrics(held, pred_test), "calibrated_test": interval_metrics(held, calibrated)}
    output = held[["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min", "clearsky_ghi", "target_power_normalized", "persistence_72h", "clearsky_scaled_persistence_72h"]].reset_index(drop=True)
    output[["mlp_q05", "mlp_q50", "mlp_q95"]] = pred_test
    output[["mlp_q05_conformal", "mlp_q50_conformal", "mlp_q95_conformal"]] = calibrated
    output.to_parquet(root / "results" / "pvod_day_ahead_matched_baselines_predictions.parquet", index=False)
    target = root / "results" / "pvod_day_ahead_matched_baselines.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
