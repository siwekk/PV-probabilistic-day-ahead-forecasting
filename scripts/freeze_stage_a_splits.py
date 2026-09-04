"""Create a deterministic Stage A split manifest from quality-controlled HKUST data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def stable_site_order(site_id: str) -> str:
    return hashlib.sha256(site_id.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project_root.resolve()
    config = json.loads((project / "configs" / "stage_a_splits.json").read_text())
    panel = pd.read_parquet(
        project / "data" / "processed" / "stage_a_hkust_features.parquet",
        columns=["site_id", "timestamp", "power_w_capacity_qc"],
    )
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    sites = sorted(panel["site_id"].unique(), key=stable_site_order)
    unseen = sites[: config["unseen_site_test_count"]]
    seen = sites[config["unseen_site_test_count"] :]
    temporal = config["temporal"]
    valid = panel["power_w_capacity_qc"].notna()
    counts = {
        "temporal_train": int((valid & (panel["timestamp"] <= temporal["train_end"])).sum()),
        "temporal_calibration": int(
            (valid & (panel["timestamp"] >= temporal["calibration_start"]) & (panel["timestamp"] <= temporal["calibration_end"])).sum()
        ),
        "temporal_test": int((valid & (panel["timestamp"] >= temporal["test_start"])).sum()),
        "unseen_site_test": int(
            (valid & panel["site_id"].isin(unseen) & (panel["timestamp"] >= temporal["test_start"])).sum()
        ),
    }
    manifest = {"config": config, "seen_training_sites": seen, "unseen_test_sites": unseen, "valid_hour_counts": counts}
    output = project / "data" / "processed" / "stage_a_split_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
