"""Summarise the NREL PVDAQ catalogue for reproducible site selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "data" / "raw" / "pvdaq" / "systems_20250729.csv"
OUTPUT = ROOT / "data" / "processed" / "pvdaq_catalogue_audit.json"


def main() -> None:
    frame = pd.read_csv(CATALOGUE)
    quality = frame["qa_status"].fillna("missing").astype(str).str.lower()
    eligible = frame[(quality == "pass") & (frame["years"] >= 2)].copy()
    eligible = eligible.sort_values(
        ["years", "available_sensor_channels", "dataset_size_mb"],
        ascending=False,
    )

    fields = [
        "system_id",
        "system_public_name",
        "site_location",
        "timezone_or_utc_offset",
        "latitude",
        "longitude",
        "dc_capacity_kW",
        "tracking",
        "type",
        "first_timestamp",
        "last_timestamp",
        "years",
        "number_records",
        "dataset_size_mb",
        "available_sensor_channels",
        "qa_status",
    ]
    report = {
        "source": "NREL PVDAQ public data lake catalogue, 2025-07-29",
        "systems": int(len(frame)),
        "qa_status_counts": quality.value_counts().to_dict(),
        "eligible_systems": int(len(eligible)),
        "recommended_candidates": eligible[fields].head(30).to_dict(orient="records"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"systems": report["systems"], "eligible": report["eligible_systems"]}, indent=2))


if __name__ == "__main__":
    main()
