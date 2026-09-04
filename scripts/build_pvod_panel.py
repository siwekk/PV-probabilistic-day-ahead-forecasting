"""Build a leakage-safe, 15-minute PVOD panel from frozen raw station records."""
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
    design = json.loads((root / "configs" / "pvod_study_design.json").read_text())
    raw = root / "data" / "raw" / "pvod" / "records"
    metadata = pd.read_csv(raw / "metadata.csv", encoding="utf-8-sig")
    main_stations = set(design["main_stations"])
    frames: list[pd.DataFrame] = []
    audit: dict[str, object] = {"stations": {}}

    for source in sorted(raw.glob("station*.csv")):
        station = source.stem
        if station not in main_stations:
            continue
        frame = pd.read_csv(source, parse_dates=["date_time"])
        duplicate = frame.duplicated("date_time", keep=False)
        removed_duplicates = int(duplicate.sum())
        frame = frame.loc[~duplicate].copy()
        meta = metadata.loc[metadata["Station_ID"] == station].iloc[0]
        capacity_mw = float(meta["Capacity"]) / 1000.0
        if station == "station04":
            capacity_mw = float(frame["power"].max()) * (1.0 + float(design["station04_capacity_rule"]["margin"]))
        frame = frame.rename(columns={"date_time": "timestamp_utc", "power": "power_mw"})
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame["timestamp_local"] = frame["timestamp_utc"].dt.tz_convert(design["feature_timezone"])
        frame["station"] = station
        frame["capacity_mw"] = capacity_mw
        frame["power_normalized"] = frame["power_mw"] / capacity_mw
        frame["latitude"] = float(meta["Latitude"])
        frame["longitude"] = float(meta["Longitude"])
        frame["hour_local"] = frame["timestamp_local"].dt.hour
        frame["minute_local"] = frame["timestamp_local"].dt.minute
        frame["dayofyear_local"] = frame["timestamp_local"].dt.dayofyear
        frames.append(frame)
        audit["stations"][station] = {"rows_kept": int(len(frame)), "rows_removed_duplicate_timestamp": removed_duplicates, "capacity_mw": capacity_mw}

    panel = pd.concat(frames, ignore_index=True).sort_values(["station", "timestamp_utc"])
    target = root / "data" / "processed" / "pvod_15min_panel.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(target, index=False)
    audit["rows_total"] = int(len(panel))
    audit["target"] = str(target.relative_to(root))
    (target.parent / "pvod_panel_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
