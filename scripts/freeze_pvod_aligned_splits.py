"""Freeze counts for the PVOD aligned-covariate temporal split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_aligned_study.json").read_text())
    frame = pd.read_parquet(root / "data" / "processed" / "pvod_15min_features.parquet")
    timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True)
    split = design["temporal_split_utc"]
    masks = {
        "train": timestamp <= pd.Timestamp(split["train_end"]),
        "calibration": (timestamp >= pd.Timestamp(split["calibration_start"])) & (timestamp <= pd.Timestamp(split["calibration_end"])),
        "test": timestamp >= pd.Timestamp(split["test_start"])
    }
    output = {"design": design, "counts": {}}
    for label, mask in masks.items():
        output["counts"][label] = {station: int(value) for station, value in frame.loc[mask].groupby("station").size().items()}
    target = root / "data" / "processed" / "pvod_aligned_split_manifest.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["counts"], indent=2))


if __name__ == "__main__":
    main()
