"""Test whether assumed local timestamps align measured PV output with daylight."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def daylight_diagnostic(frame: pd.DataFrame, power: str, capacity_kw: str) -> dict[str, float | int | str]:
    valid = frame.dropna(subset=[power, capacity_kw, "solar_zenith_deg"]).copy()
    active = valid[valid[power] > 0.02 * valid[capacity_kw] * (1000 if power.endswith("_w_capacity_qc") else 1)]
    daylight_violations = active[active["solar_zenith_deg"] >= 90]
    median_by_hour = valid.groupby(valid["timestamp"].dt.hour)[power].median()
    return {
        "rows": int(len(valid)),
        "active_rows": int(len(active)),
        "active_at_night_rows": int(len(daylight_violations)),
        "active_at_night_fraction": float(len(daylight_violations) / len(active)) if len(active) else 0.0,
        "median_power_peak_hour": int(median_by_hour.idxmax()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    processed = args.project_root.resolve() / "data" / "processed"
    hkust = pd.read_parquet(processed / "stage_a_hkust_features.parquet")
    solete = pd.read_parquet(processed / "stage_a_solete_features.parquet")
    hkust["timestamp"] = pd.to_datetime(hkust["timestamp"])
    solete["timestamp"] = pd.to_datetime(solete["timestamp"])

    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hkust": {
            "assumed_timezone": "Asia/Hong_Kong",
            **daylight_diagnostic(hkust, "power_w_capacity_qc", "rated_power_kw"),
        },
        "solete": {
            "assumed_timezone": "UTC timestamps converted to Europe/Copenhagen",
            **daylight_diagnostic(solete, "power_kw_capacity_qc", "rated_power_kw"),
            "raw_timestamp_interpretation": "UTC, converted to Europe/Copenhagen for calendar features",
        },
    }
    output = processed / "timestamp_diagnostic.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
