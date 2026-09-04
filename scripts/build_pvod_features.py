"""Create deterministic, aligned NWP, and strictly lagged PVOD features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    panel_path = root / "data" / "processed" / "pvod_15min_panel.parquet"
    panel = pd.read_parquet(panel_path).sort_values(["station", "timestamp_utc"]).copy()
    local = panel["timestamp_local"]
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    panel["hour_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    panel["hour_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    panel["doy_sin"] = np.sin(2.0 * np.pi * local.dt.dayofyear / 365.25)
    panel["doy_cos"] = np.cos(2.0 * np.pi * local.dt.dayofyear / 365.25)
    panel["nwp_winddirection_sin"] = np.sin(np.deg2rad(panel["nwp_winddirection"]))
    panel["nwp_winddirection_cos"] = np.cos(np.deg2rad(panel["nwp_winddirection"]))
    panel["lmd_winddirection_sin"] = np.sin(np.deg2rad(panel["lmd_winddirection"]))
    panel["lmd_winddirection_cos"] = np.cos(np.deg2rad(panel["lmd_winddirection"]))

    panel["solar_zenith"] = np.nan
    panel["clearsky_ghi"] = np.nan
    for station, index in panel.groupby("station").groups.items():
        rows = panel.loc[index]
        location = pvlib.location.Location(float(rows["latitude"].iloc[0]), float(rows["longitude"].iloc[0]), tz="Asia/Shanghai")
        times = pd.DatetimeIndex(rows["timestamp_local"])
        panel.loc[index, "solar_zenith"] = location.get_solarposition(times)["apparent_zenith"].to_numpy()
        panel.loc[index, "clearsky_ghi"] = location.get_clearsky(times)["ghi"].to_numpy()

    for column in ["power_normalized", "lmd_totalirrad", "lmd_diffuseirrad", "lmd_temperature", "lmd_pressure", "lmd_windspeed", "lmd_winddirection_sin", "lmd_winddirection_cos"]:
        for lag in ([1, 4, 96] if column == "power_normalized" else [1]):
            name = ("power" if column == "power_normalized" else column) + f"_lag_{lag}"
            panel[name] = panel.groupby("station", sort=False)[column].shift(lag)

    target = root / "data" / "processed" / "pvod_15min_features.parquet"
    panel.to_parquet(target, index=False)
    summary = {"rows": int(len(panel)), "target": str(target.relative_to(root)), "features": [c for c in panel.columns if c not in {"power_mw", "power_normalized"}]}
    (target.parent / "pvod_feature_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"rows": summary["rows"], "target": summary["target"]}, indent=2))


if __name__ == "__main__":
    main()
