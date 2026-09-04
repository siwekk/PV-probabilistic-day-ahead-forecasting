#!/usr/bin/env python3
"""Write a concise LaTeX record of the completed NREL Q1 experiments."""

from __future__ import annotations

import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
ladder = json.loads((root / "results" / "nrel_physics_ladder.json").read_text(encoding="ascii"))
holdout = json.loads((root / "results" / "nrel_state_holdout.json").read_text(encoding="ascii"))
uncertainty = json.loads((root / "results" / "nrel_q1_uncertainty.json").read_text(encoding="ascii"))
site_holdout = json.loads((root / "results" / "nrel_site_holdout.json").read_text(encoding="ascii"))
xgboost = json.loads((root / "results" / "nrel_matched_xgboost.json").read_text(encoding="ascii"))

m0 = ladder["M0_source_forecast"]
m1 = ladder["models"]["M1_nonphysical_direct"]
m2 = ladder["models"]["M2_solar_geometry_direct"]
m3 = ladder["models"]["M3_clear_sky_direct"]
m4 = ladder["models"]["M4_physical_residual"]
m5 = ladder["models"]["M5_physical_residual_trajectory"]


def pct_reduction(old: float, new: float) -> float:
    return 100.0 * (old - new) / old


known_mae = uncertainty["known_site_temporal"]["delta_mae"]
known_interval = uncertainty["known_site_temporal"]["delta_interval_score"]

lines = [
    r"\section{Preliminary NREL physics and transfer findings}",
    "",
    (
        "The frozen NREL experiment contains 335 simulated PV systems in Arizona, Colorado, "
        "and Washington. The panel contains 2,934,600 hourly records. Actual five-minute power "
        "was averaged within each hour and paired exactly with the source day-ahead value. The "
        "public files do not expose row-level issue times, so trajectories are indexed by local "
        "delivery day. The four-hour-ahead HA4 product was excluded. The official source is "
        r"\url{https://www.nlr.gov/grid/solar-power-data}."
    ),
    "",
    r"\subsection{Controlled temporal physics ladder}",
    "",
    (
        f"On the untouched November and December daylight test rows, the issued day-ahead "
        f"forecast had a normalised MAE of {m0['daylight_mae']:.5f}. The matched non-physical "
        f"quantile booster reduced this to {m1['raw']['daylight']['mae']:.5f}. Adding solar "
        f"geometry reduced it further to {m2['raw']['daylight']['mae']:.5f}. Clear-sky features "
        f"without residual reconstruction gave {m3['raw']['daylight']['mae']:.5f}, while the "
        f"clear-sky residual model gave {m4['raw']['daylight']['mae']:.5f}. The complete physical "
        f"residual and trajectory model achieved {m5['raw']['daylight']['mae']:.5f}. This is a "
        f"{pct_reduction(m1['raw']['daylight']['mae'], m5['raw']['daylight']['mae']):.1f}" + r"\% "
        "reduction relative to the matched non-physical booster and a "
        f"{pct_reduction(m0['daylight_mae'], m5['raw']['daylight']['mae']):.1f}" + r"\% reduction "
        "relative to the issued forecast."
    ),
    "",
    (
        rf"The raw 90\% interval score improved from {m1['raw']['daylight']['interval_score_90']:.5f} "
        f"to {m5['raw']['daylight']['interval_score_90']:.5f}. Daylight-only conformal calibration "
        f"raised the complete model coverage from {m5['raw']['daylight']['coverage_90']:.4f} to "
        f"{m5['conformal']['daylight']['coverage_90']:.4f}, with calibrated interval score "
        f"{m5['conformal']['daylight']['interval_score_90']:.5f}. The calibrated result is close "
        "to nominal coverage, although a remaining conditional calibration gap must still be "
        "reported by season, hour, forecast magnitude, and ramp regime."
    ),
    "",
    (
        f"The equal-site paired difference in daylight MAE was {known_mae['point_estimate']:.6f}, "
        rf"with a 95\% moving-block interval from {known_mae['confidence_interval_95'][0]:.6f} to "
        f"{known_mae['confidence_interval_95'][1]:.6f}. The corresponding interval-score "
        f"difference was {known_interval['point_estimate']:.6f}, with limits "
        f"{known_interval['confidence_interval_95'][0]:.6f} and "
        f"{known_interval['confidence_interval_95'][1]:.6f}. Differences are defined as the "
        "physics-informed score minus the non-physical score, so negative values favour the "
        "physics-informed model."
    ),
    "",
    r"\subsection{Complete-state transfer}",
    "",
    r"\begin{center}",
    r"\begin{tabular}{lrrrrr}",
    r"\hline",
    r"State & Source MAE & Non-physical MAE & Physics MAE & Non-physical IS & Physics IS \\",
    r"\hline",
]

for state in ("AZ", "CO", "WA"):
    values = holdout["rotations"][state]
    nonphysical = values["models"]["nonphysical_direct"]["raw"]["daylight"]
    physics = values["models"]["physics_residual_trajectory"]["raw"]["daylight"]
    lines.append(
        f"{state} & {values['source_forecast']['daylight_mae']:.5f} & "
        f"{nonphysical['mae']:.5f} & {physics['mae']:.5f} & "
        f"{nonphysical['interval_score_90']:.5f} & {physics['interval_score_90']:.5f} " + r"\\"
    )

lines.extend(
    [
        r"\hline",
        r"\end{tabular}",
        r"\end{center}",
        "",
        (
            "The physics-informed model improved daylight MAE and mean pinball loss in every "
            "held-out state. The paired site and seven-day block confidence intervals excluded "
            "zero for all three MAE comparisons. Raw interval score improved in Arizona and "
            "Colorado. It worsened in Washington because the interval was too narrow under the "
            "maritime climate shift. This is a substantive limitation. The final method must "
            "therefore include shift-aware conditional calibration, and the article must report "
            "both raw and calibrated scores rather than claiming universal interval transfer."
        ),
        "",
        r"\subsection{Current claim boundary}",
        "",
        (
            "These results support retaining physics-informed in the working title for the NREL "
            "simulation study. Solar geometry supplies a reproducible gain, while residual "
            "reconstruction, physical clipping, and day-ahead trajectory summaries improve the "
            "combined point and probabilistic result. They do not yet establish the claim on "
            "measured multi-site PV data. PVOD must independently pass the controlled physics "
            "comparison before the wording is fixed for submission."
        ),
        "",
    ]
)

site_nonphysical = site_holdout["models"]["nonphysical_direct"]["raw"]["daylight"]
site_physics = site_holdout["models"]["physics_residual_trajectory"]["raw"]["daylight"]
site_physics_calibrated = site_holdout["models"]["physics_residual_trajectory"]["conformal"]["daylight"]
xgb_daylight = xgboost["raw"]["daylight"]
xgb_calibrated = xgboost["conformal"]["daylight"]
lines.extend(
    [
        r"\subsection{Complete-site and future-time holdout}",
        "",
        (
            f"The strict site holdout fitted 233 sites, used 49 separate sites for calibration, "
            f"and evaluated 53 unseen sites in November and December. Site identifiers were not "
            f"available to either model. Daylight MAE improved from {site_nonphysical['mae']:.5f} "
            f"to {site_physics['mae']:.5f}, while interval score improved from "
            f"{site_nonphysical['interval_score_90']:.5f} to "
            f"{site_physics['interval_score_90']:.5f}. Daylight calibration produced coverage "
            f"{site_physics_calibrated['coverage_90']:.4f} and interval score "
            f"{site_physics_calibrated['interval_score_90']:.5f}."
        ),
        "",
        r"\subsection{Matched XGBoost comparator}",
        "",
        (
            f"Native XGBoost quantile regression used the same residual target, feature contract, "
            f"split, clipping, and calibration rule. Its daylight MAE was "
            f"{xgb_daylight['mae']:.5f}, raw interval score was "
            f"{xgb_daylight['interval_score_90']:.5f}, and calibrated interval score was "
            f"{xgb_calibrated['interval_score_90']:.5f}. The LightGBM physical model was slightly "
            "better, but the small difference shows that the physical representation matters more "
            "than the boosting library."
        ),
        "",
    ]
)

destination = root / "docs" / "nrel_q1_preliminary_findings.tex"
destination.write_text("\n".join(lines), encoding="ascii")
print(destination)
