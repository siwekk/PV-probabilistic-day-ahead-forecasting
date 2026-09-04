#!/usr/bin/env python3
"""Freeze and audit the data sources used by the Q1 study.

Run this script from the repository root on the calculation server.  It is
deliberately read-only with respect to raw data.  Results are written to the
processed-data and documentation directories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"

NREL_DIR = RAW / "nrel-solar-integration"
ECMWF_DIR = RAW / "ecmwf-probabilistic-solar"

OUTPUT_JSON = PROCESSED / "q1_source_audit.json"
OUTPUT_TEX = DOCS / "q1_source_audit_findings.tex"

NREL_NAME = re.compile(
    r"(?P<kind>Actual|DA|HA4)_(?P<lat>-?\d+(?:\.\d+)?)_"
    r"(?P<lon>-?\d+(?:\.\d+)?)_(?P<year>\d{4})_"
    r"(?P<plant_type>UPV|DPV)_(?P<capacity>\d+(?:\.\d+)?)MW_"
    r"(?P<resolution>\d+)_Min\.csv$",
    re.IGNORECASE,
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iso_utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def decode_line(raw: bytes) -> str:
    return raw.decode("utf-8-sig", errors="replace").strip("\r\n")


def zip_csv_profile(archive: ZipFile, member: str) -> dict[str, Any]:
    first_data = None
    last_data = None
    row_count = 0
    with archive.open(member) as handle:
        header = decode_line(handle.readline())
        for raw_line in handle:
            line = decode_line(raw_line)
            if not line:
                continue
            if first_data is None:
                first_data = line
            last_data = line
            row_count += 1
    return {
        "member": member,
        "header": header,
        "data_rows": row_count,
        "first_row": first_data,
        "last_row": last_data,
    }


def nrel_pair_key(match: re.Match[str]) -> tuple[str, ...]:
    fields = match.groupdict()
    return (
        fields["lat"],
        fields["lon"],
        fields["year"],
        fields["plant_type"].upper(),
        fields["capacity"],
    )


def audit_nrel_archive(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        for member in members:
            match = NREL_NAME.search(Path(member).name)
            if not match:
                rejected.append(member)
                continue
            fields = match.groupdict()
            records.append(
                {
                    "member": member,
                    "kind": fields["kind"].upper(),
                    "pair_key": list(nrel_pair_key(match)),
                    "resolution_minutes": int(fields["resolution"]),
                }
            )

        pairs: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        for record in records:
            key = tuple(record["pair_key"])
            pairs.setdefault(key, {})[record["kind"]] = record

        complete = {key: value for key, value in pairs.items() if {"ACTUAL", "DA"} <= value.keys()}
        incomplete = {
            "|".join(key): sorted(value.keys())
            for key, value in pairs.items()
            if not {"ACTUAL", "DA"} <= value.keys()
        }

        profiles = []
        row_patterns: Counter[tuple[int, int]] = Counter()
        header_patterns: Counter[tuple[str, str]] = Counter()
        for key in sorted(complete):
            actual = zip_csv_profile(archive, complete[key]["ACTUAL"]["member"])
            day_ahead = zip_csv_profile(archive, complete[key]["DA"]["member"])
            row_patterns[(actual["data_rows"], day_ahead["data_rows"])] += 1
            header_patterns[(actual["header"], day_ahead["header"])] += 1
            profiles.append(
                {
                    "pair_key": list(key),
                    "actual": actual,
                    "day_ahead": day_ahead,
                }
            )

    return {
        "archive": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "csv_members": len(members),
        "actual_members": sum(record["kind"] == "ACTUAL" for record in records),
        "day_ahead_members": sum(record["kind"] == "DA" for record in records),
        "hour_ahead_members": sum(record["kind"] == "HA4" for record in records),
        "hour_ahead_decision": "Excluded because the study protocol is day-ahead forecasting",
        "complete_pairs": len(complete),
        "incomplete_pairs": incomplete,
        "unrecognised_csv_members": rejected,
        "row_count_patterns": [
            {"actual_rows": key[0], "day_ahead_rows": key[1], "pairs": count}
            for key, count in sorted(row_patterns.items())
        ],
        "header_patterns": [
            {"actual": key[0], "day_ahead": key[1], "pairs": count}
            for key, count in sorted(header_patterns.items())
        ],
        "pair_profiles": profiles,
    }


def tabular_profile(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    result: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if suffix != ".csv":
        return result

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        first = None
        last = None
        rows = 0
        widths: Counter[int] = Counter()
        for row in reader:
            if not row:
                continue
            if first is None:
                first = row
            last = row
            rows += 1
            widths[len(row)] += 1
    result.update(
        {
            "columns": header,
            "data_rows": rows,
            "first_row": first,
            "last_row": last,
            "row_widths": dict(sorted(widths.items())),
        }
    )
    return result


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_ecmwf() -> dict[str, Any]:
    files = sorted(
        path
        for path in ECMWF_DIR.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    profiles = [tabular_profile(path) for path in files]
    provenance_hits = []
    for path in files:
        if path.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if "modeled generation" in lowered or "modelled generation" in lowered or "sam_gen" in lowered:
                provenance_hits.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": line_number,
                        "text": line.strip()[:500],
                    }
                )
    return {
        "root": str(ECMWF_DIR.relative_to(ROOT)),
        "git_revision": git_revision(ECMWF_DIR),
        "target_provenance": (
            "SAM-derived modelled generation estimate used by the source repository "
            "as a proxy for real PV AC power, not verified plant telemetry"
        ),
        "provenance_code_hits": provenance_hits,
        "file_count": len(files),
        "files": profiles,
    }


def find_existing_evidence() -> list[dict[str, Any]]:
    patterns = (
        "*audit*.json",
        "*result*.json",
        "*metrics*.json",
        "*split*.json",
        "*manifest*.json",
    )
    candidates: set[Path] = set()
    for base in (PROCESSED, ROOT / "results", ROOT / "configs"):
        if not base.exists():
            continue
        for pattern in patterns:
            candidates.update(base.rglob(pattern))
    evidence = []
    for path in sorted(candidates):
        if path.resolve() == OUTPUT_JSON.resolve():
            continue
        evidence.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return evidence


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_tex(report: dict[str, Any]) -> str:
    nrel = report["sources"]["nrel_solar_integration"]
    ecmwf = report["sources"]["ecmwf_probabilistic_solar"]
    archives = nrel["archives"]
    total_pairs = sum(item["complete_pairs"] for item in archives)
    incomplete = sum(len(item["incomplete_pairs"]) for item in archives)
    names = ", ".join(Path(item["archive"]).name for item in archives)
    revision = ecmwf["git_revision"] or "not available"
    return "\n".join(
        [
            r"\section{Q1 source-freeze audit}",
            "",
            (
                "The source freeze was generated on "
                + tex_escape(report["generated_at"])
                + ". Raw source files were read without modification. The machine-readable "
                + r"record is stored in \texttt{data/processed/q1\_source\_audit.json}."
            ),
            "",
            r"\subsection{NREL Solar Integration archives}",
            "",
            (
                f"The audit found {len(archives)} state archives, {total_pairs} complete "
                "Actual and DA file pairs, and "
                f"{incomplete} incomplete pairs. The frozen archives are {tex_escape(names)}. "
                "Each site also has an HA4 series, which is excluded because it is a four-hour-ahead "
                "product rather than a day-ahead forecast. "
                "Each archive checksum, member name, header, row count, first record, and last "
                "record is retained in the machine-readable record. These series are simulated, "
                "so they support cross-site, cross-state, and scaling tests, but they do not "
                "replace validation on measured PV output."
            ),
            "",
            r"\subsection{ECMWF and Jacumba archive}",
            "",
            (
                rf"The frozen repository revision is \texttt{{{tex_escape(revision)}}}. "
                f"The audit recorded {ecmwf['file_count']} source files and their checksums. "
                "The target variable is a SAM-derived modelled generation estimate. The source "
                "code treats it as a proxy for real AC power, but it is not verified plant "
                "telemetry. This archive will therefore test operational NWP provenance, "
                "probabilistic irradiance handling, and calibration under time shift. It will "
                "not be used as evidence of accuracy on measured plant power. The deterministic "
                r"archive is documented at \url{https://doi.org/10.1016/j.solener.2021.12.011}, "
                "and the ensemble archive at "
                r"\url{https://doi.org/10.1016/j.solener.2022.10.062}."
            ),
            "",
            r"\subsection{Audit decision}",
            "",
            (
                "PVOD remains the principal external source with measured PV output. EMSx remains "
                "a supplementary issued-trajectory correction experiment. NREL supplies the large "
                "geographic stress test, while ECMWF and Jacumba supply the strongest available "
                "forecast-vintage and ensemble-weather experiment. Claims and table captions must "
                "preserve these distinctions."
            ),
            "",
        ]
    )


def main() -> None:
    missing = [path for path in (NREL_DIR, ECMWF_DIR) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required source directories: " + ", ".join(map(str, missing)))

    nrel_archives = sorted(NREL_DIR.glob("*.zip"))
    if not nrel_archives:
        raise FileNotFoundError(f"No NREL zip archives found in {NREL_DIR}")

    report = {
        "schema_version": 1,
        "generated_at": iso_utc_now(),
        "repository_root": str(ROOT),
        "sources": {
            "nrel_solar_integration": {
                "source_url": "https://www.nlr.gov/grid/solar-power-data",
                "role": "simulated cross-site and cross-state stress test",
                "archives": [audit_nrel_archive(path) for path in nrel_archives],
            },
            "ecmwf_probabilistic_solar": audit_ecmwf(),
        },
        "existing_project_evidence": find_existing_evidence(),
    }

    PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    OUTPUT_TEX.write_text(render_tex(report), encoding="ascii")
    print(json.dumps({"json": str(OUTPUT_JSON), "tex": str(OUTPUT_TEX)}, indent=2))


if __name__ == "__main__":
    main()
