#!/usr/bin/env python3
"""Check ordering of saved Chronos-2 quantiles."""

from pathlib import Path

import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]
frame = pd.read_parquet(root / "results" / "nrel_chronos2_predictions.parquet")
columns = [
    "chronos2_q050_projected",
    "chronos2_q500_projected",
    "chronos2_q950_projected",
]
values = frame[columns].to_numpy(float)
ordered = (values[:, 0] <= values[:, 1]) & (values[:, 1] <= values[:, 2])
print("shape", frame.shape)
print("ordered_fraction", float(np.mean(ordered)))
print(frame.loc[~ordered, ["state", "site_id", "LocalTime", *columns]].head(20).to_string(index=False))
