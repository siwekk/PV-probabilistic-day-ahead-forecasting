"""Diagnose raw rolling-origin QGBM reliability by station and solar regime."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
def report(x):
 y=x.target_power_normalized.to_numpy();return {"n":int(len(x)),"mae":float(np.mean(abs(y-x.qgbm_median))),"coverage_90_raw":float(np.mean((y>=x.qgbm_low)&(y<=x.qgbm_high))),"width_90_raw":float(np.mean(x.qgbm_high-x.qgbm_low))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.project_root.resolve();x=pd.read_parquet(r/"results/pvod_rolling_origin_predictions.parquet");x["regime"]=np.where(x.clearsky_ghi>=20,"daylight","night");out={"by_station":{k:report(v) for k,v in x.groupby("station")},"by_regime":{k:report(v) for k,v in x.groupby("regime")},"by_station_daylight":{k:report(v) for k,v in x[x.regime=="daylight"].groupby("station")}}
 (r/"results/pvod_interval_strata.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
