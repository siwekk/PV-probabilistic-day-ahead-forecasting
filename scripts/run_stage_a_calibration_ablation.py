"""Evaluate conformal calibration and the clear-sky irradiance ablation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

QS=(0.05,0.5,0.95)
BASE=["lag1","lag24","hour_sin","hour_cos","dayofyear_sin","dayofyear_cos","solar_zenith_deg"]

def prepare(x,power):
    x=x.copy().sort_values(["site_id","timestamp"]); x["y"]=x[power]/x["rated_power_kw"]
    x["lag1"]=x.groupby("site_id")["y"].shift(1); x["lag24"]=x.groupby("site_id")["y"].shift(24)
    return x

def fit(train, features):
    train=train.dropna(subset=features+["y"]); models={}
    for q in QS:
        m=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=400,learning_rate=.05,num_leaves=31,min_child_samples=100,subsample=.8,colsample_bytree=.9,n_jobs=32,verbosity=-1)
        m.fit(train[features],train.y); models[q]=m
    return models

def metric(y,p):
    return {"mae_median":float(np.mean(abs(y-p[.5]))),"coverage_90":float(np.mean((y>=p[.05])&(y<=p[.95]))),"width_90":float(np.mean(p[.95]-p[.05]))}

def run(train,calib,tests,features):
    models=fit(train,features); calib=calib.dropna(subset=features+["y"]); cp={q:models[q].predict(calib[features]) for q in QS}
    conformity=np.maximum(cp[.05]-calib.y.to_numpy(),calib.y.to_numpy()-cp[.95]); widen=float(np.quantile(conformity,.9,method="higher"))
    out={"conformal_widening":widen,"evaluations":{}}
    for name,test in tests.items():
        test=test.dropna(subset=features+["y"]); pred={q:models[q].predict(test[features]) for q in QS}; calibrated={.05:pred[.05]-widen,.5:pred[.5],.95:pred[.95]+widen}
        out["evaluations"][name]={"raw":metric(test.y.to_numpy(),pred),"conformal":metric(test.y.to_numpy(),calibrated),"n":int(len(test))}
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.project_root.resolve(); d=root/"data"/"processed"
    m=json.loads((d/"stage_a_split_manifest.json").read_text()); t=m["config"]["temporal"]; seen=m["seen_training_sites"]; unseen=m["unseen_test_sites"]
    hk=pd.read_parquet(d/"stage_a_hkust_features.parquet"); hk.timestamp=pd.to_datetime(hk.timestamp); hk["power_kw"]=hk["power_w_capacity_qc"]/1000; hk=prepare(hk,"power_kw"); so=pd.read_parquet(d/"stage_a_solete_features.parquet"); so.timestamp=pd.to_datetime(so.timestamp); so=prepare(so,"power_kw")
    train=hk[(hk.site_id.isin(seen))&(hk.timestamp<=t["train_end"])]; calib=hk[(hk.site_id.isin(seen))&(hk.timestamp>=t["calibration_start"])&(hk.timestamp<=t["calibration_end"])]
    tests={"temporal":hk[(hk.site_id.isin(seen))&(hk.timestamp>=t["test_start"])],"unseen_site":hk[(hk.site_id.isin(unseen))&(hk.timestamp>=t["test_start"])],"external_solete":so}
    result={"physics_features":run(train,calib,tests,BASE+["clear_sky_ghi_w_m2"]),"no_clearsky_irradiance":run(train,calib,tests,BASE)}
    out=root/"results"/"stage_a_calibration_ablation.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
