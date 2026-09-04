#!/usr/bin/env python3
"""Print compact conditional-calibration results for manuscript integration."""

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
data = json.loads((root / "results/nrel_state_holdout.json").read_text())
summary = {}
for state, rotation in data["rotations"].items():
    methods = rotation["models"]["physics_residual_trajectory"]["conditional_calibration"]
    summary[state] = {
        method: {
            "overall": values["overall"]["all"],
            "season": values["season_group"],
        }
        for method, values in methods.items()
    }
print(json.dumps(summary, indent=2))
