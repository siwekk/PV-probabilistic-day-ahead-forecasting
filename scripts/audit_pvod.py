"""Audit the frozen PVOD station records before NWP modelling."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.project_root.resolve(); d=root/"data"/"raw"/"pvod"/"records"
    meta=pd.read_csv(d/"metadata.csv",encoding="utf-8-sig"); rows=[]
    for path in sorted(d.glob("station*.csv")):
        x=pd.read_csv(path,parse_dates=["date_time"]); station=path.stem; cap=float(meta.loc[meta.Station_ID==station,"Capacity"].iloc[0]); power=pd.to_numeric(x.power,errors="coerce")
        midday=x.groupby(x.date_time.dt.hour).power.median(); nwp=pd.to_numeric(x.nwp_globalirrad,errors="coerce"); diff=nwp.diff()
        rows.append({"station":station,"rows":int(len(x)),"start":x.date_time.min().isoformat(),"end":x.date_time.max().isoformat(),"cadence_minutes":float(x.date_time.sort_values().diff().dropna().dt.total_seconds().div(60).mode().iloc[0]),"duplicate_timestamps":int(x.duplicated("date_time").sum()),"max_missing_fraction":float(x.isna().mean().max()),"power_min":float(power.min()),"power_max":float(power.max()),"power_p99":float(power.quantile(.99)),"capacity_metadata":cap,"capacity_if_mw":cap/1000,"power_over_capacity_if_mw_fraction":float((power>cap/1000*1.2).mean()),"peak_median_power_hour":int(midday.idxmax()),"nwp_globalirrad_negative_differences":int((diff<0).sum()),"nwp_globalirrad_positive_differences":int((diff>0).sum())})
    out={"source":"PVOD v1.0 frozen station records","rows":rows,"notes":{"nwp_radiation":"Both positive and negative within-day differences indicate instantaneous or interval values, not daily accumulations.","timestamp_timezone":"Raw timestamps are treated as UTC. This agrees with the source implementation, which explicitly converts UTC to UTC+8, and with the 04:00 raw median power peak.","nwp_availability":"The files contain no NWP issue time or lead time. NWP fields must be described as aligned covariates until a source-confirmed operational availability convention is found."}}
    target=root/"data"/"processed"/"pvod_audit.json"; target.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
