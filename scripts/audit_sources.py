"""Create a reproducible integrity and schema audit for frozen PV sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_csv_summary(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        if len(entries) != 1:
            raise ValueError(f"Expected one CSV entry in {path}, found {len(entries)}")
        entry = entries[0]
        with archive.open(entry) as binary_handle:
            rows = csv.reader((line.decode("utf-8-sig") for line in binary_handle))
            columns = next(rows)
            first_record: list[str] | None = None
            last_record: list[str] | None = None
            row_count = 0
            for record in rows:
                if first_record is None:
                    first_record = record
                last_record = record
                row_count += 1
    return {
        "archive": path.name,
        "archive_sha256": sha256(path),
        "csv_entry": entry.filename,
        "rows": row_count,
        "columns": columns,
        "first_record": first_record,
        "last_record": last_record,
    }


def folder_inventory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {"file_count": len(files), "total_bytes": sum(item.stat().st_size for item in files)}


def hdf5_summary(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:
        return {"status": "unavailable", "reason": f"h5py is not installed: {error}"}

    datasets: list[dict[str, Any]] = []
    legacy_table: dict[str, Any] = {}
    with h5py.File(path, "r") as handle:
        def collect(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                datasets.append(
                    {
                        "path": name,
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    }
                )

        handle.visititems(collect)

        if "DATA/axis0" in handle and "DATA/axis1" in handle:
            raw_columns = handle["DATA/axis0"][:]
            raw_index = handle["DATA/axis1"][:]
            legacy_table = {
                "columns": [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in raw_columns
                ],
                "rows": int(len(raw_index)),
            }
            try:
                import pandas as pd

                timestamps = pd.to_datetime(raw_index)
                legacy_table["index_start"] = timestamps.min().isoformat()
                legacy_table["index_end"] = timestamps.max().isoformat()
            except Exception as error:
                legacy_table["index_interpretation_error"] = str(error)

    pandas_layout: dict[str, Any] = {}
    try:
        import pandas as pd

        with pd.HDFStore(path, mode="r") as store:
            pandas_layout["keys"] = store.keys()
            pandas_layout["storer_classes"] = {
                key: type(store.get_storer(key)).__name__ for key in store.keys()
            }
    except Exception as error:  # pandas layout is optional metadata only
        pandas_layout["inspection_error"] = str(error)

    return {
        "status": "ok",
        "datasets": datasets,
        "legacy_table": legacy_table,
        "pandas_layout": pandas_layout,
    }


def audit(project_root: Path) -> dict[str, Any]:
    raw = project_root / "data" / "raw"
    unisolar = raw / "unisolar"
    hkust = raw / "hkust" / "Dataset"
    solete = raw / "solete" / "SOLETE_Pombo_60min.h5"

    sites_path = unisolar / "Solar_Site_Details.csv"
    with sites_path.open(newline="", encoding="utf-8-sig") as handle:
        sites = list(csv.DictReader(handle))

    pv_root = hkust / "Time series dataset" / "PV generation dataset"
    weather_root = hkust / "Time series dataset" / "Meteorological dataset"
    pv_files = list(pv_root.rglob("*.csv"))
    weather_files = list(weather_root.rglob("*.csv")) + list(weather_root.rglob("*.xlsx"))
    groups: dict[str, int] = {}
    for file in pv_files:
        group = str(file.parent.relative_to(pv_root)).replace("\\", "/")
        groups[group] = groups.get(group, 0) + 1

    capacity_present = sum(bool((row.get("kWp") or "").strip()) for row in sites)
    coordinates_present = sum(
        bool((row.get("lat") or "").strip()) and bool((row.get("Lon") or "").strip())
        for row in sites
    )

    return {
        "audit_version": "0.2",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "unisolar": {
            "source": "frozen local project copy",
            "generation": zip_csv_summary(unisolar / "Solar_Energy_Generation.csv.zip"),
            "weather": zip_csv_summary(unisolar / "Weather_Data_reordered_all.csv.zip"),
            "site_rows": len(sites),
            "campus_count": len({row["CampusKey"] for row in sites}),
            "coordinates_present": coordinates_present,
            "capacity_present": capacity_present,
            "capacity_missing": len(sites) - capacity_present,
        },
        "hkust": {
            "source": "frozen local project copy",
            "inventory": folder_inventory(hkust),
            "pv_file_count": len(pv_files),
            "weather_file_count": len(weather_files),
            "pv_groups": groups,
            "metadata_sha256": sha256(hkust / "Metadata" / "PV generation system metadata.ttl"),
            "weather_variable_directories": sorted(
                item.name for item in weather_root.iterdir() if item.is_dir()
            ),
        },
        "solete": {
            "source": "frozen local project copy",
            "file": solete.name,
            "bytes": solete.stat().st_size,
            "sha256": sha256(solete),
            "hdf5": hdf5_summary(solete),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="project root containing data/raw",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    result = audit(args.project_root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
