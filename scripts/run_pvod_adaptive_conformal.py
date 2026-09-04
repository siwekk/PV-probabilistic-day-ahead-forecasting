"""Compare recent-window daylight conformal calibration across rolling PVOD folds."""
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
 return ca,te,np.column_stack([v[0] for v in p]),np.column_stack([v[1] for v in p])
def stat(y,lo,hi):return {"coverage_90":float(np.mean((y>=lo)&(y<=hi))),"width_90":float(np.mean(hi-lo)),"n":int(len(y))}
def main():
 p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.project_root.resolve();d=json.loads((r/"configs/pvod_day_ahead_study.json").read_text());folds=json.loads((r/"configs/pvod_rolling_origins.json").read_text())["folds"];x=pd.read_parquet(r/"data/processed/pvod_day_ahead_panel.parquet");x.target_timestamp_utc=pd.to_datetime(x.target_timestamp_utc,utc=True);features=["station_code"]+d["future_features"]+d["origin_features"];wins=[7,14,28];out={"windows_days":wins,"folds":{},"pooled":{str(w):[] for w in wins}}
 for c in folds:
  t=x.target_timestamp_utc;tr=x[t<=pd.Timestamp(c["train_end"])];ca=x[(t>=pd.Timestamp(c["calibration_start"]))&(t<=pd.Timestamp(c["calibration_end"]))];te=x[(t>=pd.Timestamp(c["test_start"]))&(t<=pd.Timestamp(c["test_end"]))];ca,te,pc,pt=fit(tr,ca,te,features);dayc=ca.clearsky_ghi.to_numpy()>=20;dayt=te.clearsky_ghi.to_numpy()>=20;score=np.maximum.reduce([pc[:,0]-ca.target_power_normalized.to_numpy(),ca.target_power_normalized.to_numpy()-pc[:,2],np.zeros(len(ca))]);out["folds"][c["name"]]={}
  cutoff=ca.target_timestamp_utc.max()
  for w in wins:
   recent=(ca.target_timestamp_utc>=cutoff-pd.Timedelta(days=w)).to_numpy()&dayc;radius=float(np.quantile(score[recent],.9,method="higher"));lo=np.clip(pt[:,0]-np.where(dayt,radius,0),0,None);hi=pt[:,2]+np.where(dayt,radius,0);record={"radius_daylight":radius,"all":stat(te.target_power_normalized.to_numpy(),lo,hi),"daylight":stat(te.target_power_normalized.to_numpy()[dayt],lo[dayt],hi[dayt]),"night":stat(te.target_power_normalized.to_numpy()[~dayt],lo[~dayt],hi[~dayt]),"n_recent_daylight":int(recent.sum())};out["folds"][c["name"]][str(w)]=record;out["pooled"][str(w)].append((te.target_power_normalized.to_numpy()[dayt],lo[dayt],hi[dayt]))
 for w,parts in out["pooled"].items():
  y=np.concatenate([z[0] for z in parts]);lo=np.concatenate([z[1] for z in parts]);hi=np.concatenate([z[2] for z in parts]);out["pooled"][w]=stat(y,lo,hi)
 (r/"results/pvod_adaptive_conformal.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
