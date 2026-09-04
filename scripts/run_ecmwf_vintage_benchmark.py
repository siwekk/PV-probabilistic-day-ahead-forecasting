#!/usr/bin/env python3
"""Run a schedule-reconstructed ECMWF ENS day-ahead benchmark at Jacumba."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pvlib

from review_metrics import QUANTILES, conditional_metrics, diagnostics, metrics, projection_audit
from run_nine_quantile_benchmarks import conformalize

SEED = 20260829


def build_panel(root: Path) -> pd.DataFrame:
    data = root / "data/raw/ecmwf-probabilistic-solar/data"
    ens = pd.read_csv(data / "Jacumba_ENS.csv")
    ens["valid_time_utc"] = pd.to_datetime(ens["Time"], utc=True)
    members = [f"EC_GHI_{index}" for index in range(1, 51)]
    values = ens[members].to_numpy(float)
    panel = pd.DataFrame({"valid_time_utc": ens["valid_time_utc"]})
    for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
        panel[f"ens_ghi_q{int(q*100):02d}"] = np.quantile(values, q, axis=1)
    panel["ens_ghi_mean"] = np.mean(values, axis=1)
    panel["ens_ghi_std"] = np.std(values, axis=1)
    panel["ens_ghi_min"] = np.min(values, axis=1)
    panel["ens_ghi_max"] = np.max(values, axis=1)

    hres = pd.read_csv(data / "ECMWF_HRES.csv").rename(columns={"Unnamed: 0": "valid_time_utc"})
    hres["valid_time_utc"] = pd.to_datetime(hres["valid_time_utc"], utc=True)
    hres["wind_speed"] = np.sqrt(hres["u10"] ** 2 + hres["v10"] ** 2)
    panel = panel.merge(hres[["valid_time_utc", "t2m", "wind_speed", "sp", "tcwv", "lcc"]], on="valid_time_utc", how="left", validate="one_to_one")

    target = pd.read_csv(data / "60947.csv").rename(columns={"Unnamed: 0": "valid_time_utc"})
    target["valid_time_utc"] = pd.to_datetime(target["valid_time_utc"], utc=True)
    panel = panel.merge(target[["valid_time_utc", "SAM_gen", "gen_clean", "gen_curtailed"]], on="valid_time_utc", how="inner", validate="one_to_one")
    panel["target_power_normalized"] = panel["SAM_gen"] / 20.0

    mcclear = pd.read_csv(data / "McClear_Jacumba.csv", sep=";")
    mcclear["period_start"] = pd.to_datetime(mcclear["Observation period"].str.split("/").str[0], utc=True)
    mcclear = mcclear.set_index("period_start")[["Clear sky GHI"]].resample("1h").sum()
    mcclear.index = mcclear.index + pd.Timedelta(hours=1)
    mcclear = mcclear.rename(columns={"Clear sky GHI": "clear_sky_ghi"}).reset_index().rename(columns={"period_start": "valid_time_utc"})
    panel = panel.merge(mcclear, on="valid_time_utc", how="left")

    solar = pvlib.solarposition.get_solarposition(panel["valid_time_utc"], 32.6193, -116.130)
    panel["solar_zenith"] = solar["zenith"].to_numpy()
    panel["solar_azimuth"] = solar["azimuth"].to_numpy()
    panel["daylight"] = panel["solar_zenith"] < 85.0
    panel["hour_sin"] = np.sin(2 * np.pi * panel["valid_time_utc"].dt.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel["valid_time_utc"].dt.hour / 24)
    panel["doy_sin"] = np.sin(2 * np.pi * panel["valid_time_utc"].dt.dayofyear / 365.25)
    panel["doy_cos"] = np.cos(2 * np.pi * panel["valid_time_utc"].dt.dayofyear / 365.25)
    panel["delivery_date"] = panel["valid_time_utc"].dt.floor("D")
    panel["issue_time_utc"] = panel["delivery_date"] - pd.Timedelta(hours=12)
    panel["lead_hours"] = (panel["valid_time_utc"] - panel["issue_time_utc"]).dt.total_seconds() / 3600
    panel["clear_sky_envelope"] = np.clip(panel["clear_sky_ghi"] / panel["clear_sky_ghi"].quantile(0.999), 0.0, 1.2)
    panel["physical_residual"] = panel["target_power_normalized"] - panel["clear_sky_envelope"]
    grouped = panel.groupby("delivery_date", observed=True)["ens_ghi_mean"]
    panel["trajectory_mean"] = grouped.transform("mean")
    panel["trajectory_std"] = grouped.transform("std")
    panel["trajectory_ramp"] = grouped.diff().fillna(0.0)
    panel["trajectory_max_abs_ramp"] = panel["trajectory_ramp"].abs().groupby(panel["delivery_date"]).transform("max")
    return panel.dropna().reset_index(drop=True)


def fit(root: Path, panel: pd.DataFrame) -> dict:
    base = ["ens_ghi_q05", "ens_ghi_q10", "ens_ghi_q25", "ens_ghi_q50", "ens_ghi_q75", "ens_ghi_q90", "ens_ghi_q95", "ens_ghi_mean", "ens_ghi_std", "ens_ghi_min", "ens_ghi_max", "t2m", "wind_speed", "sp", "tcwv", "lcc", "lead_hours", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    physical = base + ["solar_zenith", "solar_azimuth", "clear_sky_ghi", "clear_sky_envelope", "trajectory_mean", "trajectory_std", "trajectory_ramp", "trajectory_max_abs_ramp"]
    train = panel[panel["valid_time_utc"] < "2019-01-01"].copy()
    tune = panel[(panel["valid_time_utc"] >= "2019-01-01") & (panel["valid_time_utc"] < "2019-07-01")].copy()
    cal = panel[(panel["valid_time_utc"] >= "2019-07-01") & (panel["valid_time_utc"] < "2020-01-01")].copy()
    test = panel[panel["valid_time_utc"] >= "2020-01-01"].copy()
    development = pd.concat([train, tune], ignore_index=True)
    output = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": "ECMWF operational ENS archive with SAM-derived Jacumba generation", "vintage_contract": {"forecast_system": "ECMWF ENS, 50 perturbed members", "cycle": "12Z schedule reconstructed for the next UTC delivery day", "issue_time": "12:00 UTC on the day preceding delivery", "lead_range_hours": [12, 35], "spatial_interpolation": "Jacumba point series supplied by the source archive", "missing_forecasts": int(panel[["ens_ghi_mean"]].isna().any(axis=1).sum()), "timezone": "UTC", "qualification": "The public CSV preserves valid time but not the original issue identifier. The experiment is schedule-reconstructed, not row-vintage verified."}, "rows": {"train": len(train), "tune": len(tune), "calibration": len(cal), "test": len(test)}, "models": {}}
    prediction = test[["valid_time_utc", "issue_time_utc", "lead_hours", "delivery_date", "daylight", "target_power_normalized"]].copy()
    for name, features, target, residual, project in [("nonphysical_direct", base, "target_power_normalized", False, False), ("physical_residual", physical, "physical_residual", True, True)]:
        test_raw, cal_columns, test_columns, timings, sizes, best = [], [], [], {}, {}, {}
        for q in QUANTILES:
            started = time.perf_counter()
            model = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=1400, learning_rate=0.04, num_leaves=63, min_child_samples=50, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=SEED)
            model.fit(train[features], train[target], eval_set=[(tune[features], tune[target])], callbacks=[lgb.early_stopping(80, verbose=False)])
            iterations = int(model.best_iteration_ or model.n_estimators)
            final = lgb.LGBMRegressor(objective="quantile", alpha=float(q), n_estimators=iterations, learning_rate=0.04, num_leaves=63, min_child_samples=50, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=32, verbosity=-1, random_state=SEED)
            final.fit(development[features], development[target])
            cal_pred, test_pred = final.predict(cal[features]), final.predict(test[features])
            if residual:
                cal_pred += cal["clear_sky_envelope"].to_numpy(float)
                test_pred += test["clear_sky_envelope"].to_numpy(float)
            test_raw.append(test_pred.copy())
            if project:
                cal_pred, test_pred = np.clip(cal_pred, 0, 1.2), np.clip(test_pred, 0, 1.2)
            cal_columns.append(cal_pred); test_columns.append(test_pred)
            timings[str(q)] = time.perf_counter() - started; sizes[str(q)] = len(final.booster_.model_to_string().encode()); best[str(q)] = iterations
        raw = np.column_stack(test_raw)
        cal_q, test_q = np.sort(np.column_stack(cal_columns), axis=1), np.sort(np.column_stack(test_columns), axis=1)
        if project:
            cal_q[~cal["daylight"].to_numpy(bool)] = 0.0; test_q[~test["daylight"].to_numpy(bool)] = 0.0
        conformed, radii = conformalize(cal["target_power_normalized"].to_numpy(float), cal_q, test_q, cal["daylight"].to_numpy(bool), test["daylight"].to_numpy(bool))
        day = test["daylight"].to_numpy(bool)
        groups = {
            "lead_band": pd.cut(test["lead_hours"], [11.9, 17, 23, 29, 35.1], labels=["12-17h", "18-23h", "24-29h", "30-35h"]),
            "solar_elevation_regime": pd.cut(90 - test["solar_zenith"], [-90, 0, 15, 35, 90], labels=["night", "low", "medium", "high"]),
            "season": np.select([test["valid_time_utc"].dt.month.isin([12,1,2]), test["valid_time_utc"].dt.month.isin([3,4,5]), test["valid_time_utc"].dt.month.isin([6,7,8])], ["winter", "spring", "summer"], default="autumn"),
            "clear_sky_variability": pd.qcut(test["trajectory_max_abs_ramp"].rank(method="first"), 3, labels=["low", "medium", "high"]),
            "production_regime": pd.cut(test["target_power_normalized"], [-np.inf, .02, .2, .5, np.inf], labels=["near_zero", "low", "medium", "high"]),
        }
        output["models"][name] = {"raw_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), test_q[day]), "conformal_daylight": metrics(test.loc[day, "target_power_normalized"].to_numpy(float), conformed[day]), "physical_diagnostics": diagnostics(test, raw, test_q), "projection_audit": projection_audit(test.reset_index(drop=True), raw), "conditional": conditional_metrics(test, conformed, groups), "conformal_radii": radii, "compute": {"training_seconds_total": sum(timings.values()), "training_seconds_by_quantile": timings, "serialized_bytes_total": sum(sizes.values()), "best_iterations": best}}
        for column, q in enumerate(QUANTILES): prediction[f"{name}_q{int(q*100):02d}"] = conformed[:, column]
    (root / "results/ecmwf_vintage_review.json").write_text(json.dumps(output, indent=2) + "\n")
    prediction.to_parquet(root / "results/ecmwf_vintage_review_predictions.parquet", index=False)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    panel = build_panel(root)
    panel.to_parquet(root / "data/processed/ecmwf_jacumba_vintage_panel.parquet", index=False)
    print(json.dumps(fit(root, panel), indent=2))


if __name__ == "__main__":
    main()
