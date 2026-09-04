#!/usr/bin/env python3
"""Evaluate physical output projection on the best measured PVOD model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_nrel_physics_ladder import score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    frame = pd.read_parquet(root / "results" / "pvod_q1_physics_ladder_predictions.parquet")
    raw = np.column_stack([frame["M1_q05"], frame["M1_q50"], frame["M1_q95"]]).astype(float)
    ordered = np.sort(raw, axis=1)
    projected = np.clip(ordered, 0.0, 1.2)
    night = ~frame["daylight"].to_numpy(dtype=bool)
    projected[night, :] = 0.0
    daylight = frame["daylight"].to_numpy(dtype=bool)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Post-model physical projection of the best non-physical NWP quantile model. "
            "Quantiles are ordered, clipped to [0, 1.2] per unit, and set to zero outside daylight."
        ),
        "violations_before_projection": {
            "negative_q05": int(np.sum(ordered[:, 0] < 0.0)),
            "median_outside_capacity_support": int(
                np.sum((ordered[:, 1] < 0.0) | (ordered[:, 1] > 1.2))
            ),
            "q95_above_capacity_support": int(np.sum(ordered[:, 2] > 1.2)),
            "nonzero_night_median": int(np.sum(np.abs(ordered[night, 1]) > 1e-12)),
        },
        "raw": {
            "overall": score(frame, ordered[:, 0], ordered[:, 1], ordered[:, 2]),
            "daylight": score(
                frame.loc[daylight],
                ordered[daylight, 0],
                ordered[daylight, 1],
                ordered[daylight, 2],
            ),
        },
        "projected": {
            "overall": score(frame, projected[:, 0], projected[:, 1], projected[:, 2]),
            "daylight": score(
                frame.loc[daylight],
                projected[daylight, 0],
                projected[daylight, 1],
                projected[daylight, 2],
            ),
        },
    }
    destination = root / "results" / "pvod_physical_projection.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
