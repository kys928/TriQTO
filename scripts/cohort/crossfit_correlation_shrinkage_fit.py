#!/usr/bin/env python3
"""FIT-only family CV of 6D covariance correlation shrinkage (no QPU/TEST)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from crossfit_nonlinear_covariance_fit import records,FOLDS,BUDGETS,LOW,HIGH,trace
BETAS=(0.,.25,.50,.75,1.)
CUT3=7.814727903
CUT6=12.591587244

def train_scale(data):
    scales={}
    for group in ("coordinates","matched"):
        byfam={}
        for r in data:
            byfam.setdefault(r["fid"],[]).append(
                trace(r["emp"],group)/max(trace(r["estimated"],group),1e-30))
        scales[group]=np.clip(np.median([np.median(x) for x in byfam.values()]),LOW,HIGH)
    return np.diag([np.sqrt(scales["coordinates"])]*3+[np.sqrt(scales["matched"])]*3)

def sigma(r,d,beta):
    x=d@r["estimated"]@d
    x=(1-beta)*x+beta*np.diag(np.diag(x))
    x=(1+1/15)*x
    ridge=max(1e-12,1e-8*float(np.trace(x))/6)
    return x+ridge*np.eye(6)

def score(z,c):
    eigen,vec=np.linalg.eigh(c)
    if not np.isfinite(eigen).all() or eigen[0]<=0:raise RuntimeError("Covariance not PD after ridge")
    coordinates=vec.T@z
    quadratic=float(np.sum((coordinates**2)/eigen))
    pseudo_nll=.5*(float(np.log(eigen).sum())+quadratic+6*np.log(2*np.pi))
    return pseudo_nll,quadratic

def stats(a):
    x=np.asarray(a,dtype=float)
    return {"n":len(x),"mean":float(x.mean()),"median":float(np.median(x)),
            "p10":float(np.quantile(x,.1)),"p90":float(np.quantile(x,.9))}

def evaluate(corpus):
    data=records(corpus)
    result={"schema":"triqto.fit_correlation_shrinkage_cv.v1",
      "data":"120 FIT families only; first root candidate and observation; 16 calibration draws",
      "observed_only":True,"shot_cost_increase":0,
      "frozen_beta_grid":list(BETAS),"selection_metric":"training-fold Gaussian pseudo-NLL",
      "family_disjoint_cv_folds":FOLDS,
      "SELECTION_TEST_used":False,"model_evaluated":False,
      "by_budget":{}}
    for shots in BUDGETS:
        observed=[r for r in data if r["shots"]==shots]
        results=[]
        choices=[]
        for held in range(FOLDS):
            train=[r for r in observed if r["fold"]!=held]
            valid=[r for r in observed if r["fold"]==held]
            d=train_scale(train)
            costs={}
            for beta in BETAS:
                byfam={}
                for r in train:
                    z=r["z_ref"]-r["z_other_mean"]
                    nll,_=score(z,sigma(r,d,beta))
                    byfam.setdefault(r["fid"],[]).append(nll)
                costs[beta]=float(np.mean([np.mean(x) for x in byfam.values()]))
            chosen=min(BETAS,key=lambda k:(costs[k],k))
            choices.append({"held_fold":held,"beta":chosen,
                "train_families":len({r["fid"] for r in train}),
                "held_families":len({r["fid"] for r in valid}),
                "training_mean_nll":{str(k):costs[k] for k in BETAS}})
            for r in valid:
                z=r["z_ref"]-r["z_other_mean"]
                for name,beta in (("beta_zero",0.),("trained_beta",chosen)):
                    cov=sigma(r,d,beta)
                    nll,quad=score(z,cov)
                    out={"fid":r["fid"],"model":name,"nll":nll,
                         "covered_6d":int(quad<=CUT6)}
                    empirical=(1+1/15)*r["emp"]
                    estimate=cov
                    out["relative_frobenius"]=float(np.linalg.norm(estimate-empirical,"fro")/
                                                     max(np.linalg.norm(empirical,"fro"),1e-12))
                    for grp,sl in (("coordinates",slice(0,3)),("matched",slice(3,6))):
                        z3=z[sl]
                        c3=cov[sl,sl]
                        eig,vec=np.linalg.eigh(c3)
                        if eig.min()<=0:raise RuntimeError("Invalid 3D covariance")
                        sq=float(np.sum(((vec.T@z3)**2)/eig))
                        out["covered_"+grp]=int(sq<=CUT3)
                    results.append(out)
        aggregates={}
        for name in ("beta_zero","trained_beta"):
            rows=[r for r in results if r["model"]==name]
            aggregates[name]={
               "NLL":stats([r["nll"] for r in rows]),
               "full_6d_coverage":stats([r["covered_6d"] for r in rows]),
               "coordinate_3d_coverage":stats([r["covered_coordinates"] for r in rows]),
               "matched_3d_coverage":stats([r["covered_matched"] for r in rows]),
               "relative_frobenius":stats([r["relative_frobenius"] for r in rows])}
        result["by_budget"][str(shots)]={"choices":choices,"out_of_fold":aggregates}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=evaluate(a.corpus)
    a.out.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps({budget:{"selected_betas":[x["beta"] for x in v["choices"]],
       "NLL":{k:round(z["NLL"]["mean"],4) for k,z in v["out_of_fold"].items()},
       "full_cov":{k:round(z["full_6d_coverage"]["mean"],4) for k,z in v["out_of_fold"].items()}}
          for budget,v in r["by_budget"].items()},indent=2))
