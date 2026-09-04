#!/usr/bin/env python3
"""Print compact schemas and row counts for saved prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    results = args.project_root.resolve() / "results"
    patterns = (
        "*predictions*.parquet",
        "emsx_*lead_96.parquet",
    )
    files = sorted({path for pattern in patterns for path in results.glob(pattern)})
    for path in files:
        frame = pd.read_parquet(path)
        print(f"{path.name}\trows={len(frame)}\tcolumns={list(frame.columns)}")


if __name__ == "__main__":
    main()
