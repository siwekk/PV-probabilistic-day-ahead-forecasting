#!/usr/bin/env python3
"""Consolidate projection-only feasibility audits across experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from review_metrics import projection_audit


NATIVE = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
COMMON = np.array([0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95])


def chronos(root: Path) -> dict:
    frame = pd.read_parquet(root / "results/nrel_chronos2_predictions.parquet")
    native = np.column_stack([frame[f"chronos2_q{int(q*1000):03d}_raw"] for q in NATIVE])
    common = np.empty((len(frame), len(COMMON)))
    for row in range(len(frame)):
        common[row] = np.interp(COMMON, NATIVE, native[row])
    return projection_audit(frame, common)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = {
        "NREL": json.loads((root / "results/nrel_nine_quantile_review.json").read_text()),
        "PVOD": json.loads((root / "results/pvod_nine_quantile_review.json").read_text()),
        "ECMWF": json.loads((root / "results/ecmwf_vintage_review.json").read_text()),
    }
    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "projection_definition": "Order adjacent quantiles, clip to [0,1.2] p.u., and set all night quantiles to zero. No conformal calibration is included.", "audits": {}}
    for source, data in sources.items():
        output["audits"][source] = {name: model["projection_audit"] for name, model in data["models"].items()}
    output["audits"]["NREL"]["chronos2_zero_shot"] = chronos(root)
    output["not_auditable"] = {"EMSx": "Capacity and solar-support metadata are unavailable, so the common capacity and night projection cannot be evaluated."}
    path = root / "results/physical_feasibility_audit.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
