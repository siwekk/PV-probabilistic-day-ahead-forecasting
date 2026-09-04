"""Audit the fixed PVDAQ external-validation subset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "pvdaq" / "selected"
OUTPUT = ROOT / "data" / "processed" / "pvdaq_selected_audit.json"


def main() -> None:
    records = []
    for path in sorted(RAW.rglob("*.csv")):
        frame = pd.read_csv(path)
        columns = list(frame.columns)
        time_columns = [x for x in columns if "time" in x.lower() or "date" in x.lower()]
        pv_columns = [
            x
            for x in columns
            if any(token in x.lower() for token in ("ac", "dc", "power", "energy", "pv"))
        ]
        record = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(frame),
            "columns": columns,
            "time_columns": time_columns,
            "pv_related_columns": pv_columns,
        }
        if frame.empty:
            record["status"] = "empty_source_file"
        else:
            record.update(
                {
                    "status": "usable",
                    "missing_fraction_max": float(frame.isna().mean().max()),
                    "first_row": frame.iloc[0].astype(str).to_dict(),
                }
            )
        records.append(
            record
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "rows": sum(x["rows"] for x in records)}, indent=2))


if __name__ == "__main__":
    main()
