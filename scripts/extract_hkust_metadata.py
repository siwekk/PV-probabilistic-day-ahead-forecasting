"""Extract site metadata from the HKUST Brick Turtle file and map it to panel IDs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def local_name(value: Any) -> str:
    return str(value).rsplit("#", maxsplit=1)[-1].rsplit("/", maxsplit=1)[-1]


def normalise(value: str) -> str:
    value = value.casefold().replace("center", "centre")
    key = re.sub(r"[^a-z0-9]", "", value)
    aliases = {"indoorsportscentre": "indoorsportcentre"}
    return aliases.get(key, key)


def block_value(block: str, property_name: str) -> str | None:
    match = re.search(
        rf"ext:{re.escape(property_name)}\s*\[\s*(.*?)\s*\]",
        block,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = re.search(r"brick:value\s+(?:\"([^\"]+)\"|([-+]?\d+(?:\.\d+)?))", match.group(1))
    if not value:
        return None
    return value.group(1) or value.group(2)


def metadata_rows(ttl: Path) -> list[dict[str, Any]]:
    text = ttl.read_text(encoding="utf-8")
    systems = re.findall(
        r"(?ms)^pvsystem:([A-Za-z0-9_]+) a brick:PV_Generation_System\s*;(.*?)(?=^pvsystem:|\Z)",
        text,
    )
    rows: list[dict[str, Any]] = []
    for system_id, block in systems:
        latitude = re.search(r"brick:latitude\s+([-+]?\d+(?:\.\d+)?)", block)
        longitude = re.search(r"brick:longitude\s+([-+]?\d+(?:\.\d+)?)", block)
        rated_power = block_value(block, "ratedPowerOutput")
        rows.append(
            {
                "metadata_system_id": system_id,
                "metadata_key": normalise(system_id),
                "latitude": float(latitude.group(1)) if latitude else None,
                "longitude": float(longitude.group(1)) if longitude else None,
                "rated_power_kw": float(rated_power) if rated_power else None,
                "tilt_raw": block_value(block, "tiltAngle"),
                "azimuth_raw": block_value(block, "azimuth"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    project = args.project_root.resolve()
    ttl = project / "data" / "raw" / "hkust" / "Dataset" / "Metadata" / "PV generation system metadata.ttl"
    panel_path = project / "data" / "processed" / "stage_a_hkust_hourly.parquet"
    output = project / "data" / "processed" / "hkust_site_metadata.parquet"
    report = project / "data" / "processed" / "hkust_metadata_mapping.json"

    metadata = pd.DataFrame(metadata_rows(ttl))
    panel = pd.read_parquet(panel_path, columns=["site_id"])
    sites = pd.DataFrame({"site_id": sorted(panel["site_id"].unique())})
    sites["metadata_key"] = sites["site_id"].str.rsplit("/", n=1).str[-1].map(normalise)
    mapped = sites.merge(metadata, on="metadata_key", how="left", validate="one_to_one")
    mapped.to_parquet(output, index=False)

    result = {
        "panel_sites": int(len(sites)),
        "metadata_systems": int(len(metadata)),
        "mapped_sites": int(mapped["metadata_system_id"].notna().sum()),
        "unmatched_panel_sites": mapped.loc[mapped["metadata_system_id"].isna(), "site_id"].tolist(),
        "unmatched_metadata_systems": metadata.loc[
            ~metadata["metadata_key"].isin(set(sites["metadata_key"])), "metadata_system_id"
        ].tolist(),
        "output": str(output),
    }
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
