"""Run matched QGBM and MLP PVOD day-ahead rolling-origin evaluations."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Q=(.05,.5,.95)
class MLP(nn.Module):
    def __init__(self,n): super().__init__(); self.net=nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x): return torch.sort(self.net(x),dim=1).values
def loss(p,y):
    q=torch.tensor([.05,.5,.95],device=p.device); e=y[:,None]-p
    return torch.maximum(q*e,(q-1)*e).mean()
def metric(y,p):
    return {"mae":float(np.mean(abs(y-p[:,1]))),"coverage":float(np.mean((y>=p[:,0])&(y<=p[:,2]))),"width":float(np.mean(p[:,2]-p[:,0]))}
def split(f,c):
    t=f.target_timestamp_utc
    return f[t<=pd.Timestamp(c['train_end'])],f[(t>=pd.Timestamp(c['calibration_start']))&(t<=pd.Timestamp(c['calibration_end']))],f[(t>=pd.Timestamp(c['test_start']))&(t<=pd.Timestamp(c['test_end']))]
def fit_qgbm(train,cal,test,features):
    train=train.dropna(subset=features+["target_power_normalized"]); cal=cal.dropna(subset=features+["target_power_normalized"]); test=test.dropna(subset=features+["target_power_normalized"])
    out={"cal":[],"test":[]}
    for q in Q:
        m=lgb.LGBMRegressor(objective="quantile",alpha=q,n_estimators=600,learning_rate=.05,num_leaves=63,min_child_samples=100,colsample_bytree=.9,reg_lambda=1,n_jobs=32,verbosity=-1,random_state=20260826)
        m.fit(train[features],train.power_normalized if "power_normalized" in train else train.target_power_normalized,categorical_feature=["station_code"])
        out["cal"].append(np.clip(m.predict(cal[features]),0,1.2)); out["test"].append(np.clip(m.predict(test[features]),0,1.2))
    return train,cal,test,np.column_stack(out["cal"]),np.column_stack(out["test"])
def fit_mlp(train,cal,test,features):
    train=train.dropna(subset=features+["target_power_normalized"]); cal=cal.dropna(subset=features+["target_power_normalized"]); test=test.dropna(subset=features+["target_power_normalized"])
    sc=StandardScaler().fit(train[features].astype(float)); xt=sc.transform(train[features].astype(float)).astype("float32"); xc=sc.transform(cal[features].astype(float)).astype("float32"); xe=sc.transform(test[features].astype(float)).astype("float32")
    torch.manual_seed(20260826);np.random.seed(20260826);random.seed(20260826);d=torch.device("cuda" if torch.cuda.is_available() else "cpu");m=MLP(xt.shape[1]).to(d);o=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4);dl=DataLoader(TensorDataset(torch.from_numpy(xt),torch.from_numpy(train.target_power_normalized.to_numpy("float32"))),batch_size=4096,shuffle=True)
    best=None;bestloss=float("inf");wait=0;ct=torch.from_numpy(xc).to(d);cy=torch.from_numpy(cal.target_power_normalized.to_numpy("float32")).to(d)
    for epoch in range(70):
        m.train()
        for x,y in dl:o.zero_grad();z=loss(m(x.to(d)),y.to(d));z.backward();o.step()
        m.eval()
        with torch.no_grad():v=float(loss(m(ct),cy))
        if v<bestloss-1e-6:bestloss=v;best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};wait=0
        else:
            wait+=1
            if wait==10:break
    m.load_state_dict(best);m.eval()
    with torch.no_grad(): pc=np.clip(m(ct).cpu().numpy(),0,1.2);pe=np.clip(m(torch.from_numpy(xe).to(d)).cpu().numpy(),0,1.2)
    return train,cal,test,pc,pe,epoch+1
def main():
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();r=a.project_root.resolve();d=json.loads((r/"configs/pvod_day_ahead_study.json").read_text());folds=json.loads((r/"configs/pvod_rolling_origins.json").read_text())["folds"]
    f=pd.read_parquet(r/"data/processed/pvod_day_ahead_panel.parquet");f.target_timestamp_utc=pd.to_datetime(f.target_timestamp_utc,utc=True);hist=["station_code","hour_sin","hour_cos","doy_sin","doy_cos","solar_zenith","clearsky_ghi"]+d["origin_features"]; full=["station_code"]+d["future_features"]+d["origin_features"];summary={"folds":{}};pred=[]
    for c in folds:
        tr,ca,te=split(f,c);_,ch,th,_,ph=fit_qgbm(tr,ca,te,hist);_,cq,tq,pcq,ptq=fit_qgbm(tr,ca,te,full);_,cm,tm,pcm,ptm,epochs=fit_mlp(tr,ca,te,full)
        ycal=cq.target_power_normalized.to_numpy();radq=float(np.quantile(np.maximum.reduce([pcq[:,0]-ycal,ycal-pcq[:,2],np.zeros(len(cq))]),.9,method="higher"));ym=cm.target_power_normalized.to_numpy();radm=float(np.quantile(np.maximum.reduce([pcm[:,0]-ym,ym-pcm[:,2],np.zeros(len(cm))]),.9,method="higher"));
        daylight=tq.clearsky_ghi.to_numpy()>=20;summary["folds"][c["name"]]={"qgbm_history":metric(th.target_power_normalized.to_numpy(),ph),"qgbm_nwp_raw":metric(tq.target_power_normalized.to_numpy(),ptq),"qgbm_nwp_calibrated":metric(tq.target_power_normalized.to_numpy(),np.c_[np.clip(ptq[:,0]-radq,0,None),ptq[:,1],ptq[:,2]+radq]),"mlp_raw":metric(tm.target_power_normalized.to_numpy(),ptm),"mlp_calibrated":metric(tm.target_power_normalized.to_numpy(),np.c_[np.clip(ptm[:,0]-radm,0,None),ptm[:,1],ptm[:,2]+radm]),"qgbm_daylight_mae":float(np.mean(abs(tq.target_power_normalized.to_numpy()[daylight]-ptq[daylight,1]))),"mlp_daylight_mae":float(np.mean(abs(tm.target_power_normalized.to_numpy()[daylight]-ptm[daylight,1]))),"mlp_epochs":epochs}
        z=tq[["station","issue_timestamp_utc","target_timestamp_utc","lead_15min","clearsky_ghi","target_power_normalized"]].copy();z["fold"]=c["name"];z["qgbm_median"]=ptq[:,1];z["mlp_median"]=ptm[:,1];z["qgbm_low"]=ptq[:,0];z["qgbm_high"]=ptq[:,2];pred.append(z)
    out=r/"results";out.mkdir(exist_ok=True);(out/"pvod_rolling_origin_summary.json").write_text(json.dumps(summary,indent=2)+"\n");pd.concat(pred).to_parquet(out/"pvod_rolling_origin_predictions.parquet",index=False);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
