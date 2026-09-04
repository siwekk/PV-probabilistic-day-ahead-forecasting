"""Build non-overlapping PVOD day-ahead examples with only issue-time observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def lookup(base: pd.DataFrame, queries: pd.DataFrame, key: str, output: str) -> pd.DataFrame:
    right = base[["station", "timestamp_utc", "power_normalized"]].rename(columns={"timestamp_utc": key, "power_normalized": output})
    return queries.merge(right, on=["station", key], how="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "pvod_day_ahead_study.json").read_text())
    base = pd.read_parquet(root / "data" / "processed" / "pvod_15min_features.parquet")
    base["timestamp_utc"] = pd.to_datetime(base["timestamp_utc"], utc=True)
    base = base.sort_values(["station", "timestamp_utc"])
    start = base["timestamp_utc"].min().floor("D") + pd.Timedelta(hours=12)
    end = base["timestamp_utc"].max().floor("D") + pd.Timedelta(hours=12)
    origins = pd.date_range(start, end, freq="D", tz="UTC")
    leads = np.arange(design["evaluated_leads_15min"][0], design["evaluated_leads_15min"][1] + 1)
    stations = sorted(base["station"].unique())
    examples = pd.MultiIndex.from_product([stations, origins, leads], names=["station", "issue_timestamp_utc", "lead_15min"]).to_frame(index=False)
    examples["target_timestamp_utc"] = examples["issue_timestamp_utc"] + pd.to_timedelta(examples["lead_15min"] * 15, unit="min")
    target = base.rename(columns={"timestamp_utc": "target_timestamp_utc", "power_normalized": "target_power_normalized"})
    preserve = ["station", "target_timestamp_utc", "target_power_normalized"] + design["future_features"]
    examples = examples.merge(target[preserve], on=["station", "target_timestamp_utc"], how="left")
    examples = lookup(base, examples, "issue_timestamp_utc", "origin_power")
    examples["origin_lag4_timestamp"] = examples["issue_timestamp_utc"] - pd.Timedelta(minutes=60)
    examples["origin_lag96_timestamp"] = examples["issue_timestamp_utc"] - pd.Timedelta(hours=24)
    examples = lookup(base, examples, "origin_lag4_timestamp", "origin_power_lag_4")
    examples = lookup(base, examples, "origin_lag96_timestamp", "origin_power_lag_96")
    examples["target_minus_72h_timestamp"] = examples["target_timestamp_utc"] - pd.Timedelta(hours=72)
    historical = base[["station", "timestamp_utc", "power_normalized", "clearsky_ghi"]].rename(columns={"timestamp_utc": "target_minus_72h_timestamp", "power_normalized": "persistence_72h", "clearsky_ghi": "clearsky_ghi_72h"})
    examples = examples.merge(historical, on=["station", "target_minus_72h_timestamp"], how="left")
    examples["clearsky_scaled_persistence_72h"] = np.where(
        examples["clearsky_ghi"] < 20.0,
        0.0,
        examples["persistence_72h"] * examples["clearsky_ghi"] / examples["clearsky_ghi_72h"],
    )
    examples["clearsky_scaled_persistence_72h"] = examples["clearsky_scaled_persistence_72h"].clip(0.0, 1.2)
    examples["station_code"] = examples["station"].map({station: i for i, station in enumerate(stations)}).astype("category")
    output = root / "data" / "processed" / "pvod_day_ahead_panel.parquet"
    examples.to_parquet(output, index=False)
    summary = {"rows": int(len(examples)), "rows_with_target": int(examples["target_power_normalized"].notna().sum()), "issues": int(len(origins)), "leads_per_issue": int(len(leads)), "target": str(output.relative_to(root))}
    (output.parent / "pvod_day_ahead_panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
