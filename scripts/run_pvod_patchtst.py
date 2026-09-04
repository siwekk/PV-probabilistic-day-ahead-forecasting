"""Evaluate a PatchTST-style NWP-trajectory model on PVOD day-ahead windows."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class PatchTST(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__(); self.patch = nn.Conv1d(features, 64, kernel_size=8, stride=8); self.position = nn.Parameter(torch.zeros(1, 12, 64)); layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=.1, batch_first=True); self.encoder = nn.TransformerEncoder(layer, num_layers=2); self.head = nn.Linear(64, 8)
    def forward(self, value):
        tokens = self.patch(value.transpose(1, 2)).transpose(1, 2)+self.position; return self.head(self.encoder(tokens)).reshape(value.shape[0], 96)


def build_sequences(frame, features, split):
    output=[]
    for (station, issue), group in frame.groupby(["station", "issue_timestamp_utc"], observed=True, sort=False):
        group = group.sort_values("lead_15min")
        if len(group) != 96 or group[features+["target_power_normalized"]].isna().any().any(): continue
        last = pd.Timestamp(group.target_timestamp_utc.iloc[-1]); first = pd.Timestamp(group.target_timestamp_utc.iloc[0])
        label = "train" if last <= pd.Timestamp(split["train_end"]) else "validation" if first >= pd.Timestamp(split["calibration_start"]) and last <= pd.Timestamp(split["calibration_end"]) else "test" if first >= pd.Timestamp(split["test_start"]) else None
        if label:
            metadata = group[["station", "issue_timestamp_utc", "target_timestamp_utc", "lead_15min", "clearsky_ghi", "target_power_normalized"]].copy()
            output.append((label, group[features].to_numpy(float), group.target_power_normalized.to_numpy(float), group.clearsky_ghi.to_numpy(float), metadata))
    return output


def main() -> None:
    random.seed(20260828); np.random.seed(20260828); torch.manual_seed(20260828)
    design = json.loads((ROOT/"configs"/"pvod_day_ahead_study.json").read_text()); frame = pd.read_parquet(ROOT/"data"/"processed"/"pvod_day_ahead_panel.parquet")
    frame["target_timestamp_utc"] = pd.to_datetime(frame.target_timestamp_utc, utc=True); features = [*design["future_features"], *design["origin_features"]]; sequences = build_sequences(frame, features, design["temporal_split_target_utc"])
    groups = {name: [item for item in sequences if item[0] == name] for name in ("train", "validation", "test")}; scaler = StandardScaler().fit(np.concatenate([item[1] for item in groups["train"]], axis=0))
    arrays = {name: (np.stack([scaler.transform(item[1]).astype(np.float32) for item in values]), np.stack([item[2].astype(np.float32) for item in values]), np.stack([item[3] for item in values])) for name, values in groups.items()}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=PatchTST(len(features)).to(device); optimiser=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); best=None; best_loss=float("inf"); stale=0
    for epoch in range(60):
        model.train(); loader=DataLoader(TensorDataset(torch.from_numpy(arrays["train"][0]), torch.from_numpy(arrays["train"][1])), batch_size=64, shuffle=True)
        for x,y in loader: optimiser.zero_grad(); loss=torch.nn.functional.l1_loss(model(x.to(device)), y.to(device)); loss.backward(); optimiser.step()
        model.eval();
        with torch.no_grad(): loss=float(torch.nn.functional.l1_loss(model(torch.from_numpy(arrays["validation"][0]).to(device)), torch.from_numpy(arrays["validation"][1]).to(device)).item())
        if loss < best_loss: best,best_loss,stale={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},loss,0
        else: stale+=1
        if stale>=10: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): prediction=np.clip(model(torch.from_numpy(arrays["test"][0]).to(device)).cpu().numpy(), 0, 1.2)
    target=arrays["test"][1]; daylight=arrays["test"][2]>=20; result={"study":design["name"],"restriction":design["restriction"],"method":"PatchTST-style model of the issued 24-hour NWP trajectory and origin-power features","device":str(device),"sequences":{name:len(value) for name,value in groups.items()},"fit":{"epochs":epoch+1,"validation_mae":best_loss},"test":{"mae_all":float(np.abs(target-prediction).mean()),"mae_daylight":float(np.abs(target[daylight]-prediction[daylight]).mean()),"n":int(target.size),"n_daylight":int(daylight.sum())}}
    output = pd.concat([item[4] for item in groups["test"]], ignore_index=True)
    output["patchtst_prediction"] = prediction.reshape(-1)
    output.to_parquet(ROOT/"results"/"pvod_patchtst_day_ahead_predictions.parquet", index=False)
    (ROOT/"results"/"pvod_patchtst_day_ahead.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
