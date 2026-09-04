"""Build quality-flagged hourly HKUST and SOLETE panels for Stage A."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROVISIONAL_MAX_POWER_W = 1_000_000.0


def hkust_site_id(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/")


def build_hkust_file(path: Path, pv_root: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    target = next((column for column in columns if column.strip().lower() == "power(w)"), None)
    if target is None:
        raise ValueError(f"No site-level power(W) target in {path}")

    frame = pd.read_csv(path, usecols=["Time", target], parse_dates=["Time"])
    frame = frame.rename(columns={"Time": "timestamp", target: "power_w_raw"})
    frame["power_w_raw"] = pd.to_numeric(frame["power_w_raw"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    frame["invalid_negative"] = frame["power_w_raw"] < 0
    frame["invalid_extreme"] = frame["power_w_raw"] > PROVISIONAL_MAX_POWER_W
    frame["power_w_qc"] = frame["power_w_raw"].where(
        ~(frame["invalid_negative"] | frame["invalid_extreme"])
    )

    indexed = frame.set_index("timestamp")
    hourly = indexed.resample("1h").agg(
        power_w=("power_w_qc", "mean"),
        source_observations=("power_w_raw", "size"),
        valid_observations=("power_w_qc", "count"),
        negative_observations=("invalid_negative", "sum"),
        extreme_observations=("invalid_extreme", "sum"),
    )
    hourly = hourly.reset_index()
    hourly.insert(0, "site_id", hkust_site_id(path, pv_root))
    hourly["source"] = "hkust"
    hourly["timestamp_timezone"] = "unverified_local_or_UTC"
    return hourly


def build_hkust(raw: Path, workers: int) -> pd.DataFrame:
    pv_root = raw / "hkust" / "Dataset" / "Time series dataset" / "PV generation dataset"
    files = sorted(pv_root.rglob("Site level dataset/*.csv"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        panels = list(executor.map(lambda path: build_hkust_file(path, pv_root), files))
    return pd.concat(panels, ignore_index=True).sort_values(["site_id", "timestamp"])


def build_solete(raw: Path) -> pd.DataFrame:
    path = raw / "solete" / "SOLETE_Pombo_60min.h5"
    with h5py.File(path, "r") as handle:
        columns = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["DATA/axis0"][:]
        ]
        timestamps = pd.to_datetime(handle["DATA/axis1"][:])
        values = pd.DataFrame(handle["DATA/block0_values"][:], columns=columns)

    panel = pd.DataFrame(
        {
            "site_id": "solete_pombo",
            "timestamp": timestamps,
            "power_kw": values["P_Solar[kW]"],
            "ghi_kw_m2": values["GHI[kW1m2]"],
            "poa_irr_kw_m2": values["POA Irr[kW1m2]"],
            "temperature_c": values["TEMPERATURE[degC]"],
            "humidity_pct": values["HUMIDITY[%]"],
            "wind_speed_m_s": values["WIND_SPEED[m1s]"],
            "wind_direction_deg": values["WIND_DIR[deg]"],
            "pressure_mbar": values["Pressure[mbar]"],
            "solar_azimuth_deg": values["Azimuth[deg]"],
            "solar_elevation_deg": values["Elevation[deg]"],
        }
    )
    panel["source"] = "solete"
    panel["timestamp_timezone"] = "unverified_local_or_UTC"
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    project = args.project_root.resolve()
    processed = project / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    hkust = build_hkust(project / "data" / "raw", workers=args.workers)
    solete = build_solete(project / "data" / "raw")

    hkust_path = processed / "stage_a_hkust_hourly.parquet"
    solete_path = processed / "stage_a_solete_hourly.parquet"
    hkust.to_parquet(hkust_path, index=False)
    solete.to_parquet(solete_path, index=False)

    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "provisional_hkust_max_power_w": PROVISIONAL_MAX_POWER_W,
        "hkust": {
            "rows": int(len(hkust)),
            "sites": int(hkust["site_id"].nunique()),
            "missing_hourly_power": int(hkust["power_w"].isna().sum()),
            "negative_observations_flagged": int(hkust["negative_observations"].sum()),
            "extreme_observations_flagged": int(hkust["extreme_observations"].sum()),
            "output": str(hkust_path),
        },
        "solete": {
            "rows": int(len(solete)),
            "sites": int(solete["site_id"].nunique()),
            "missing_hourly_power": int(solete["power_kw"].isna().sum()),
            "output": str(solete_path),
        },
    }
    summary_path = processed / "stage_a_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
