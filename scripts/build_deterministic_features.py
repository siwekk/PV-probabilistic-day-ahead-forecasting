"""Add solar geometry, clear-sky irradiance, and capacity QC to Stage A panels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


CAPACITY_MARGIN = 1.20


def hkust_features(project: Path) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    panel = pd.read_parquet(project / "data" / "processed" / "stage_a_hkust_hourly.parquet")
    metadata = pd.read_parquet(project / "data" / "processed" / "hkust_site_metadata.parquet")
    frame = panel.merge(metadata, on="site_id", how="left", validate="many_to_one")
    if frame[["latitude", "longitude", "rated_power_kw"]].isna().any().any():
        raise ValueError("Missing required HKUST metadata after site mapping")

    frame["timestamp_local"] = pd.to_datetime(frame["timestamp"]).dt.tz_localize("Asia/Hong_Kong")
    result: list[pd.DataFrame] = []
    for site_id, site in frame.groupby("site_id", sort=False):
        local_times = pd.DatetimeIndex(site["timestamp_local"])
        location = pvlib.location.Location(
            latitude=float(site["latitude"].iloc[0]),
            longitude=float(site["longitude"].iloc[0]),
            tz="Asia/Hong_Kong",
        )
        position = location.get_solarposition(local_times)
        clearsky = location.get_clearsky(local_times, model="ineichen")
        site = site.copy()
        site["solar_zenith_deg"] = position["apparent_zenith"].to_numpy()
        site["solar_azimuth_deg"] = position["azimuth"].to_numpy()
        site["clear_sky_ghi_w_m2"] = clearsky["ghi"].to_numpy()
        site["clear_sky_dni_w_m2"] = clearsky["dni"].to_numpy()
        site["clear_sky_dhi_w_m2"] = clearsky["dhi"].to_numpy()
        site["capacity_upper_w"] = site["rated_power_kw"] * 1000 * CAPACITY_MARGIN
        site["capacity_excess"] = site["power_w"] > site["capacity_upper_w"]
        site["power_w_capacity_qc"] = site["power_w"].where(~site["capacity_excess"])
        result.append(site)

    frame = pd.concat(result, ignore_index=True)
    local = pd.DatetimeIndex(frame["timestamp_local"])
    frame["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    frame["dayofyear_sin"] = np.sin(2 * np.pi * local.dayofyear / 366)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * local.dayofyear / 366)
    frame["feature_timezone_assumption"] = "Asia/Hong_Kong"
    summary = {
        "rows": int(len(frame)),
        "sites": int(frame["site_id"].nunique()),
        "capacity_excess_hours": int(frame["capacity_excess"].sum()),
        "timezone_assumption": "Asia/Hong_Kong",
    }
    return frame, summary


def solete_features(project: Path) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    frame = pd.read_parquet(project / "data" / "processed" / "stage_a_solete_hourly.parquet")
    times_utc = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"])).tz_localize("UTC")
    times = times_utc.tz_convert("Europe/Copenhagen")
    frame["timestamp_local"] = times
    location = pvlib.location.Location(55.6867, 12.0985, tz="Europe/Copenhagen")
    position = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")
    frame["rated_power_kw"] = 10.0
    frame["latitude"] = 55.6867
    frame["longitude"] = 12.0985
    frame["solar_zenith_deg"] = position["apparent_zenith"].to_numpy()
    frame["solar_azimuth_deg"] = position["azimuth"].to_numpy()
    frame["clear_sky_ghi_w_m2"] = clearsky["ghi"].to_numpy()
    frame["clear_sky_dni_w_m2"] = clearsky["dni"].to_numpy()
    frame["clear_sky_dhi_w_m2"] = clearsky["dhi"].to_numpy()
    frame["capacity_upper_kw"] = frame["rated_power_kw"] * CAPACITY_MARGIN
    frame["capacity_excess"] = frame["power_kw"] > frame["capacity_upper_kw"]
    frame["power_kw_capacity_qc"] = frame["power_kw"].where(~frame["capacity_excess"])
    frame["hour_sin"] = np.sin(2 * np.pi * times.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * times.hour / 24)
    frame["dayofyear_sin"] = np.sin(2 * np.pi * times.dayofyear / 366)
    frame["dayofyear_cos"] = np.cos(2 * np.pi * times.dayofyear / 366)
    frame["feature_timezone_assumption"] = "UTC timestamps converted to Europe/Copenhagen"
    summary = {
        "rows": int(len(frame)),
        "sites": int(frame["site_id"].nunique()),
        "capacity_excess_hours": int(frame["capacity_excess"].sum()),
        "timezone_assumption": "UTC timestamps converted to Europe/Copenhagen",
    }
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project_root.resolve()
    processed = project / "data" / "processed"
    hkust, hkust_summary = hkust_features(project)
    solete, solete_summary = solete_features(project)
    hkust.to_parquet(processed / "stage_a_hkust_features.parquet", index=False)
    solete.to_parquet(processed / "stage_a_solete_features.parquet", index=False)
    summary = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "capacity_margin": CAPACITY_MARGIN,
        "hkust": hkust_summary,
        "solete": solete_summary,
    }
    (processed / "stage_a_feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
