#!/usr/bin/env python3
"""Build the frozen hourly NREL panel for the Q1 study."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pvlib


FILE_NAME = re.compile(
    r"(?P<kind>Actual|DA)_(?P<lat>-?\d+(?:\.\d+)?)_"
    r"(?P<lon>-?\d+(?:\.\d+)?)_(?P<year>\d{4})_"
    r"(?P<plant_type>UPV|DPV)_(?P<capacity>\d+(?:\.\d+)?)MW_"
    r"(?P<resolution>\d+)_Min\.csv$",
    re.IGNORECASE,
)


def site_id(state: str, fields: dict[str, str]) -> str:
    return "_".join(
        [
            state,
            fields["lat"],
            fields["lon"],
            fields["plant_type"].upper(),
            fields["capacity"] + "MW",
        ]
    )


def split_sites(sites: list[str], fractions: dict[str, float]) -> dict[str, str]:
    ordered = sorted(
        sites,
        key=lambda value: hashlib.sha256(("20260828|" + value).encode("ascii")).hexdigest(),
    )
    count = len(ordered)
    train_end = max(1, int(np.floor(count * fractions["train_fraction"])))
    calibration_end = train_end + max(1, int(np.floor(count * fractions["calibration_fraction"])))
    calibration_end = min(calibration_end, count - 1)
    result = {}
    for index, value in enumerate(ordered):
        if index < train_end:
            result[value] = "train"
        elif index < calibration_end:
            result[value] = "calibration"
        else:
            result[value] = "test"
    return result


def read_member(archive: ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        content = handle.read()
    frame = pd.read_csv(io.BytesIO(content))
    if list(frame.columns) != ["LocalTime", "Power(MW)"]:
        raise ValueError(f"Unexpected columns in {member}: {list(frame.columns)}")
    frame["LocalTime"] = pd.to_datetime(frame["LocalTime"], format="%m/%d/%y %H:%M")
    frame["Power(MW)"] = pd.to_numeric(frame["Power(MW)"], errors="raise")
    if frame["LocalTime"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {member}")
    return frame


def trajectory_features(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("delivery_date", sort=False)["source_forecast_normalized"]
    frame["trajectory_mean"] = grouped.transform("mean")
    frame["trajectory_max"] = grouped.transform("max")
    frame["trajectory_energy_proxy"] = grouped.transform("sum")
    frame["trajectory_ramp"] = grouped.diff().fillna(0.0)
    frame["trajectory_abs_ramp"] = frame["trajectory_ramp"].abs()
    frame["trajectory_max_abs_ramp"] = frame.groupby("delivery_date", sort=False)[
        "trajectory_abs_ramp"
    ].transform("max")
    peak_hour = frame.loc[grouped.idxmax(), ["delivery_date", "delivery_hour"]].rename(
        columns={"delivery_hour": "trajectory_peak_hour"}
    )
    return frame.merge(peak_hour, on="delivery_date", how="left", validate="many_to_one")


def build_site(
    archive: ZipFile,
    state: str,
    fixed_offset: int,
    actual_member: str,
    da_member: str,
    fields: dict[str, str],
    daylight_threshold: float,
) -> pd.DataFrame:
    actual = read_member(archive, actual_member)
    day_ahead = read_member(archive, da_member)
    actual_hourly = (
        actual.set_index("LocalTime")["Power(MW)"]
        .resample("1h")
        .mean()
        .rename("target_power_mw")
        .reset_index()
    )
    if len(actual_hourly) != 8760 or len(day_ahead) != 8760:
        raise ValueError(f"Expected 8760 hourly rows for {actual_member}")
    day_ahead = day_ahead.rename(columns={"Power(MW)": "source_forecast_mw"})
    frame = actual_hourly.merge(day_ahead, on="LocalTime", how="inner", validate="one_to_one")
    if len(frame) != 8760:
        raise ValueError(f"Actual and DA timestamps do not align for {actual_member}")

    latitude = float(fields["lat"])
    longitude = float(fields["lon"])
    capacity = float(fields["capacity"])
    fixed_tz = timezone(timedelta(hours=fixed_offset))
    aware_local = pd.DatetimeIndex(frame["LocalTime"]).tz_localize(fixed_tz)
    valid_utc = aware_local.tz_convert("UTC")
    location = pvlib.location.Location(latitude, longitude, tz=fixed_tz)
    position = location.get_solarposition(aware_local)
    clearsky = location.get_clearsky(aware_local, model="ineichen")

    frame["source"] = "NREL_Solar_Integration_2006"
    frame["state"] = state
    frame["site_id"] = site_id(state, fields)
    frame["latitude"] = latitude
    frame["longitude"] = longitude
    frame["capacity_mw"] = capacity
    frame["plant_type"] = fields["plant_type"].upper()
    frame["fixed_utc_offset_hours"] = fixed_offset
    frame["valid_timestamp_utc"] = valid_utc
    frame["delivery_date"] = frame["LocalTime"].dt.strftime("%Y-%m-%d")
    frame["delivery_hour"] = frame["LocalTime"].dt.hour.astype("int8")
    frame["target_power_normalized"] = frame["target_power_mw"] / capacity
    frame["source_forecast_normalized"] = frame["source_forecast_mw"] / capacity
    frame["source_residual_normalized"] = (
        frame["target_power_normalized"] - frame["source_forecast_normalized"]
    )
    frame["solar_zenith_deg"] = position["apparent_zenith"].to_numpy()
    frame["solar_azimuth_deg"] = position["azimuth"].to_numpy()
    frame["clear_sky_ghi_w_m2"] = clearsky["ghi"].to_numpy()
    frame["clear_sky_envelope"] = np.clip(frame["clear_sky_ghi_w_m2"] / 1000.0, 0.0, 1.2)
    frame["daylight"] = frame["clear_sky_ghi_w_m2"] >= daylight_threshold
    hour = frame["delivery_hour"].to_numpy()
    day_of_year = frame["LocalTime"].dt.dayofyear.to_numpy()
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.0)
    frame["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.0)
    frame["issue_timestamp_utc"] = pd.NaT
    frame["forecast_origin_status"] = "not_exposed_by_source"
    return trajectory_features(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    design = json.loads((root / "configs" / "nrel_q1_study.json").read_text(encoding="utf-8"))
    raw = root / "data" / "raw" / "nrel-solar-integration"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    state_frames = []
    summaries = {}

    for state, state_config in design["states"].items():
        archive_path = raw / state_config["archive"]
        with ZipFile(archive_path) as archive:
            members = {}
            for member in archive.namelist():
                match = FILE_NAME.search(Path(member).name)
                if not match:
                    continue
                fields = match.groupdict()
                key = (
                    fields["lat"],
                    fields["lon"],
                    fields["year"],
                    fields["plant_type"].upper(),
                    fields["capacity"],
                )
                members.setdefault(key, {})[fields["kind"].upper()] = (member, fields)
            sites = []
            for key in sorted(members):
                pair = members[key]
                if not {"ACTUAL", "DA"} <= pair.keys():
                    raise ValueError(f"Incomplete NREL pair for {state}: {key}")
                actual_member, fields = pair["ACTUAL"]
                da_member, _ = pair["DA"]
                sites.append(
                    build_site(
                        archive,
                        state,
                        int(state_config["fixed_utc_offset_hours"]),
                        actual_member,
                        da_member,
                        fields,
                        float(design["daylight_clearsky_ghi_w_m2"]),
                    )
                )
        state_frame = pd.concat(sites, ignore_index=True)
        split_map = split_sites(
            sorted(state_frame["site_id"].unique()), design["complete_site_split"]
        )
        state_frame["site_split"] = state_frame["site_id"].map(split_map)
        destination = processed / f"nrel_q1_panel_{state.lower()}.parquet"
        state_frame.to_parquet(destination, index=False)
        summaries[state] = {
            "rows": int(len(state_frame)),
            "sites": int(state_frame["site_id"].nunique()),
            "delivery_days": int(state_frame["delivery_date"].nunique()),
            "daylight_rows": int(state_frame["daylight"].sum()),
            "site_split_counts": {
                key: int(value)
                for key, value in state_frame.groupby("site_split")["site_id"].nunique().items()
            },
            "target_min": float(state_frame["target_power_normalized"].min()),
            "target_max": float(state_frame["target_power_normalized"].max()),
            "forecast_min": float(state_frame["source_forecast_normalized"].min()),
            "forecast_max": float(state_frame["source_forecast_normalized"].max()),
            "output": str(destination.relative_to(root)),
        }
        state_frames.append(state_frame)

    full = pd.concat(state_frames, ignore_index=True)
    full["site_code"] = full["site_id"].astype("category")
    full["state_code"] = full["state"].astype("category")
    destination = processed / "nrel_q1_panel.parquet"
    full.to_parquet(destination, index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(full)),
        "sites": int(full["site_id"].nunique()),
        "states": summaries,
        "missing_values": {
            column: int(count)
            for column, count in full.isna().sum().items()
            if int(count) > 0
        },
        "output": str(destination.relative_to(root)),
    }
    summary_path = processed / "nrel_q1_panel_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
