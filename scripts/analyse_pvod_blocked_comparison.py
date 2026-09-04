"""Compare rolling-origin QGBM and MLP errors with daily issue-block bootstrap."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.project_root.resolve();x=pd.read_parquet(r/"results/pvod_rolling_origin_predictions.parquet");x["daylight"]=x.clearsky_ghi>=20
 out={}
 for label,part in {"all":x,"daylight":x[x.daylight]}.items():
  daily=part.assign(qgbm_error=abs(part.target_power_normalized-part.qgbm_median),mlp_error=abs(part.target_power_normalized-part.mlp_median)).groupby(["fold","issue_timestamp_utc"])[["qgbm_error","mlp_error"]].mean();d=(daily.mlp_error-daily.qgbm_error).to_numpy();rng=np.random.default_rng(20260826);boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(20000)]);out[label]={"n_issue_days":int(len(d)),"qgbm_mae":float(daily.qgbm_error.mean()),"mlp_mae":float(daily.mlp_error.mean()),"mean_mlp_minus_qgbm":float(d.mean()),"relative_qgbm_improvement":float(d.mean()/daily.mlp_error.mean()),"bootstrap_95_ci":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"p_qgbm_not_better":float(np.mean(boot<=0))}
 (r/"results/pvod_blocked_comparison.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
