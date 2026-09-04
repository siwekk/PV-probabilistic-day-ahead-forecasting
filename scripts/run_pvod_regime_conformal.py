"""Evaluate daylight and night conditional conformal calibration over rolling PVOD folds."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
Q=(.05,.5,.95)
def fit(tr,ca,te,features):
 tr=tr.dropna(subset=features+["target_power_normalized"]);ca=ca.dropna(subset=features+["target_power_normalized"]);te=te.dropna(subset=features+["target_power_normalized"]);p=[]
 for q in Q:
  m=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=600,learning_rate=.05,num_leaves=63,min_child_samples=100,colsample_bytree=.9,reg_lambda=1,n_jobs=32,verbosity=-1,random_state=20260826);m.fit(tr[features],tr.target_power_normalized,categorical_feature=["station_code"]);p.append((np.clip(m.predict(ca[features]),0,1.2),np.clip(m.predict(te[features]),0,1.2)))
 return ca,te,np.column_stack([z[0] for z in p]),np.column_stack([z[1] for z in p])
def stat(x,lo,hi):
 y=x.target_power_normalized.to_numpy();return {"n":int(len(x)),"coverage_90":float(np.mean((y>=lo)&(y<=hi))),"width_90":float(np.mean(hi-lo))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.project_root.resolve();d=json.loads((r/"configs/pvod_day_ahead_study.json").read_text());folds=json.loads((r/"configs/pvod_rolling_origins.json").read_text())["folds"];x=pd.read_parquet(r/"data/processed/pvod_day_ahead_panel.parquet");x.target_timestamp_utc=pd.to_datetime(x.target_timestamp_utc,utc=True);features=["station_code"]+d["future_features"]+d["origin_features"];out={"folds":{},"pooled":{}};allrows=[]
 for c in folds:
  t=x.target_timestamp_utc;tr=x[t<=pd.Timestamp(c["train_end"])];ca=x[(t>=pd.Timestamp(c["calibration_start"]))&(t<=pd.Timestamp(c["calibration_end"]))];te=x[(t>=pd.Timestamp(c["test_start"]))&(t<=pd.Timestamp(c["test_end"]))];ca,te,pc,pt=fit(tr,ca,te,features);dayc=ca.clearsky_ghi.to_numpy()>=20;dayt=te.clearsky_ghi.to_numpy()>=20;yc=ca.target_power_normalized.to_numpy();s=np.maximum.reduce([pc[:,0]-yc,yc-pc[:,2],np.zeros(len(ca))]);radii={};
  for station in sorted(ca.station.unique()):
   station_mask=ca.station.to_numpy()==station;radii[station]={"daylight":float(np.quantile(s[station_mask&dayc],.9,method="higher")),"night":float(np.quantile(s[station_mask&(~dayc)],.9,method="higher"))}
  rad=np.array([radii[station]["daylight" if daytime else "night"] for station,daytime in zip(te.station.to_numpy(),dayt)]);lo=np.clip(pt[:,0]-rad,0,None);hi=pt[:,2]+rad;out["folds"][c["name"]]={"radii":radii,"all":stat(te,lo,hi),"daylight":stat(te[dayt],lo[dayt],hi[dayt]),"night":stat(te[~dayt],lo[~dayt],hi[~dayt])};z=te[["station","clearsky_ghi","target_power_normalized"]].copy();z["fold"]=c["name"];z["low"]=lo;z["high"]=hi;allrows.append(z)
 z=pd.concat(allrows);day=z.clearsky_ghi>=20;out["pooled"]={"all":stat(z,z.low.to_numpy(),z.high.to_numpy()),"daylight":stat(z[day],z.low.to_numpy()[day],z.high.to_numpy()[day]),"night":stat(z[~day],z.low.to_numpy()[~day],z.high.to_numpy()[~day]),"station_daylight":{k:stat(v,v.low.to_numpy(),v.high.to_numpy()) for k,v in z[day].groupby("station")}}
 (r/"results/pvod_station_regime_conformal.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
