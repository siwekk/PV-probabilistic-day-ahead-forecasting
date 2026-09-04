#!/usr/bin/env python3
"""Print a compact, human-reviewable summary of the Q1 source audit."""

from __future__ import annotations

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
path = root / "data" / "processed" / "q1_source_audit.json"
report = json.loads(path.read_text(encoding="ascii"))

nrel = []
for archive in report["sources"]["nrel_solar_integration"]["archives"]:
    nrel.append(
        {
            "archive": archive["archive"],
            "csv_members": archive["csv_members"],
            "actual_members": archive["actual_members"],
            "day_ahead_members": archive["day_ahead_members"],
            "hour_ahead_members": archive["hour_ahead_members"],
            "hour_ahead_decision": archive["hour_ahead_decision"],
            "complete_pairs": archive["complete_pairs"],
            "incomplete_pairs": len(archive["incomplete_pairs"]),
            "unrecognised_csv_members": len(archive["unrecognised_csv_members"]),
            "row_count_patterns": archive["row_count_patterns"],
            "first_pair_sample": archive["pair_profiles"][0],
        }
    )

ecmwf = report["sources"]["ecmwf_probabilistic_solar"]
summary = {
    "generated_at": report["generated_at"],
    "nrel": nrel,
    "ecmwf_revision": ecmwf["git_revision"],
    "ecmwf_files": ecmwf["file_count"],
    "target_provenance": ecmwf["target_provenance"],
    "provenance_code_hits": ecmwf["provenance_code_hits"],
    "existing_evidence_files": len(report["existing_project_evidence"]),
}
print(json.dumps(summary, indent=2))
