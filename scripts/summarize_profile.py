"""Print the key indicators from a source-profile JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))

    unisolar = profile["unisolar"]
    hkust = profile["hkust"]
    solete = profile["solete"]
    unisolar_sites = unisolar["site_profiles"]
    hkust_sites = hkust["site_profiles"]
    hkust_usable = [item for item in hkust_sites if "target" in item]
    hkust_incompatible = [item["file"] for item in hkust_sites if "target" not in item]
    unisolar_low_coverage = [
        f"campus{item['campus']}_site{item['site']}"
        for item in unisolar_sites
        if item["target_completeness"] < 0.8
    ]
    hkust_negative = [
        item["file"] for item in hkust_usable if item["target"]["negative_count"] > 0
    ]
    hkust_extreme = [
        item["file"] for item in hkust_usable if item["target"]["maximum"] > 1_000_000
    ]
    weather_gaps = {
        f"campus{item['campus']}": {
            column: missing
            for column, missing in item["missing_fraction"].items()
            if missing > 0.1
        }
        for item in unisolar["weather_profiles"]
    }

    result = {
        "unisolar": {
            "power_rows": unisolar["power_rows"],
            "weather_rows": unisolar["weather_rows"],
            "sites": len(unisolar_sites),
            "target_completeness_min": min(item["target_completeness"] for item in unisolar_sites),
            "target_completeness_median": sorted(
                item["target_completeness"] for item in unisolar_sites
            )[len(unisolar_sites) // 2],
            "target_min": min(item["target"]["minimum"] for item in unisolar_sites),
            "target_max": max(item["target"]["maximum"] for item in unisolar_sites),
            "weather_missing_fraction_max": max(
                value
                for item in unisolar["weather_profiles"]
                for value in item["missing_fraction"].values()
            ),
            "sites_below_80_percent_target_completeness": unisolar_low_coverage,
            "weather_fields_over_10_percent_missing": weather_gaps,
        },
        "hkust": {
            "sites": len(hkust_sites),
            "usable_target_sites": len(hkust_usable),
            "incompatible_files": hkust_incompatible,
            "cadences_minutes": sorted({item.get("cadence_minutes") for item in hkust_usable}),
            "target_min": min(item["target"]["minimum"] for item in hkust_usable),
            "target_max": max(item["target"]["maximum"] for item in hkust_usable),
            "negative_target_total": sum(
                item["target"]["negative_count"] for item in hkust_usable
            ),
            "target_completeness_min": min(
                item["target"]["non_missing"] / item["rows"] for item in hkust_usable
            ),
            "files_with_negative_power": hkust_negative,
            "files_with_power_over_1_MW": hkust_extreme,
        },
        "solete": {
            "rows": solete["rows"],
            "cadence_minutes": solete["cadence_minutes"],
            "missing_fraction_max": max(solete["missing_fraction"].values()),
            "solar_power": solete["numerical_ranges"]["P_Solar[kW]"],
            "ghi": solete["numerical_ranges"]["GHI[kW1m2]"],
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
