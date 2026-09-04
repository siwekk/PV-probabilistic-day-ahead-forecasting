"""Profile coverage, cadence, missingness, and numerical ranges of PV sources."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def cadence_minutes(timestamps: pd.Series) -> float | None:
    ordered = pd.Series(pd.to_datetime(timestamps, errors="coerce").dropna().unique()).sort_values()
    if len(ordered) < 2:
        return None
    differences = ordered.diff().dropna().dt.total_seconds().div(60)
    if differences.empty:
        return None
    return float(differences.mode().iloc[0])


def numeric_summary(values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return {"non_missing": 0}
    quantiles = valid.quantile([0.01, 0.5, 0.99])
    return {
        "non_missing": int(valid.size),
        "negative_count": int((valid < 0).sum()),
        "zero_count": int((valid == 0).sum()),
        "minimum": float(valid.min()),
        "p01": float(quantiles.loc[0.01]),
        "median": float(quantiles.loc[0.5]),
        "p99": float(quantiles.loc[0.99]),
        "maximum": float(valid.max()),
    }


def profile_unisolar(raw: Path) -> dict[str, Any]:
    power = pd.read_csv(raw / "unisolar" / "Solar_Energy_Generation.csv.zip", parse_dates=["Timestamp"])
    weather = pd.read_csv(raw / "unisolar" / "Weather_Data_reordered_all.csv.zip", parse_dates=["Timestamp"])

    site_rows: list[dict[str, Any]] = []
    for (campus, site), frame in power.groupby(["CampusKey", "SiteKey"], sort=True):
        target = pd.to_numeric(frame["SolarGeneration"], errors="coerce")
        site_rows.append(
            {
                "campus": int(campus),
                "site": int(site),
                "rows": int(len(frame)),
                "target_missing": int(target.isna().sum()),
                "target_completeness": float(target.notna().mean()),
                "timestamp_start": frame["Timestamp"].min().isoformat(),
                "timestamp_end": frame["Timestamp"].max().isoformat(),
                "duplicate_timestamps": int(frame.duplicated("Timestamp").sum()),
                "cadence_minutes": cadence_minutes(frame["Timestamp"]),
                "target": numeric_summary(target),
            }
        )

    weather_rows: list[dict[str, Any]] = []
    weather_variables = [column for column in weather.columns if column not in {"CampusKey", "Timestamp"}]
    for campus, frame in weather.groupby("CampusKey", sort=True):
        weather_rows.append(
            {
                "campus": int(campus),
                "rows": int(len(frame)),
                "timestamp_start": frame["Timestamp"].min().isoformat(),
                "timestamp_end": frame["Timestamp"].max().isoformat(),
                "duplicate_timestamps": int(frame.duplicated("Timestamp").sum()),
                "cadence_minutes": cadence_minutes(frame["Timestamp"]),
                "missing_fraction": {
                    column: float(frame[column].isna().mean()) for column in weather_variables
                },
            }
        )

    return {
        "power_columns": power.columns.tolist(),
        "weather_columns": weather.columns.tolist(),
        "power_rows": int(len(power)),
        "weather_rows": int(len(weather)),
        "site_profiles": site_rows,
        "weather_profiles": weather_rows,
        "timestamp_timezone": "not encoded in the supplied CSV timestamps",
    }


def profile_hkust_file(path: Path) -> dict[str, Any]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    time_column = "Time" if "Time" in header else None
    target_candidates = [
        column
        for column in header
        if "totalActivePower" in column or column.strip().lower() == "power(w)"
    ]
    result: dict[str, Any] = {
        "file": path.name,
        "columns": header,
        "time_column": time_column,
        "target_candidates": target_candidates,
    }
    if not time_column or not target_candidates:
        return result

    target_column = target_candidates[0]
    frame = pd.read_csv(path, usecols=[time_column, target_column], parse_dates=[time_column])
    target = pd.to_numeric(frame[target_column], errors="coerce")
    result.update(
        {
            "rows": int(len(frame)),
            "timestamp_start": frame[time_column].min().isoformat(),
            "timestamp_end": frame[time_column].max().isoformat(),
            "duplicate_timestamps": int(frame.duplicated(time_column).sum()),
            "cadence_minutes": cadence_minutes(frame[time_column]),
            "target": numeric_summary(target),
        }
    )
    return result


def profile_hkust(raw: Path, workers: int) -> dict[str, Any]:
    root = raw / "hkust" / "Dataset" / "Time series dataset"
    pv_root = root / "PV generation dataset"
    site_files = sorted(pv_root.rglob("Site level dataset/*.csv"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        profiles = list(executor.map(profile_hkust_file, site_files))

    weather_root = root / "Meteorological dataset"
    weather_inventory = {
        item.name: sorted(file.name for file in item.iterdir() if file.is_file())
        for item in weather_root.iterdir()
        if item.is_dir()
    }
    return {
        "site_level_file_count": len(site_files),
        "site_profiles": profiles,
        "weather_inventory": weather_inventory,
        "timestamp_timezone": "not stated in the supplied README or CSV fields",
    }


def profile_solete(raw: Path) -> dict[str, Any]:
    path = raw / "solete" / "SOLETE_Pombo_60min.h5"
    with h5py.File(path, "r") as handle:
        columns = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["DATA/axis0"][:]
        ]
        timestamps = pd.to_datetime(handle["DATA/axis1"][:])
        values = pd.DataFrame(handle["DATA/block0_values"][:], columns=columns)

    return {
        "rows": int(len(values)),
        "timestamp_start": timestamps.min().isoformat(),
        "timestamp_end": timestamps.max().isoformat(),
        "duplicate_timestamps": int(pd.Series(timestamps).duplicated().sum()),
        "cadence_minutes": cadence_minutes(pd.Series(timestamps)),
        "missing_fraction": {column: float(values[column].isna().mean()) for column in columns},
        "numerical_ranges": {column: numeric_summary(values[column]) for column in columns},
        "timestamp_timezone": "not encoded in the HDF5 timestamp index",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw = args.project_root.resolve() / "data" / "raw"
    result = {
        "profile_version": "0.1",
        "profiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "unisolar": profile_unisolar(raw),
        "hkust": profile_hkust(raw, workers=args.workers),
        "solete": profile_solete(raw),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "profile_version": result["profile_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
