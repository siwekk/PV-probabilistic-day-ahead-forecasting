#!/usr/bin/env python3
"""Print the schemas of frozen panels needed for the final reviewer response."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for name in [
    "nrel_nine_quantile_review_predictions.parquet",
    "pvod_nine_quantile_review_predictions.parquet",
    "nrel_state_holdout_predictions.parquet",
    "pvod_q1_physics_ladder_predictions.parquet",
]:
    path = ROOT / "results" / name
    frame = pd.read_parquet(path)
    print(name)
    print(frame.shape)
    print(frame.columns.tolist())

panel = pd.read_parquet(ROOT / "data" / "processed" / "pvod_day_ahead_panel.parquet")
print("pvod_day_ahead_panel.parquet")
print(panel.shape)
print(panel.columns.tolist())
