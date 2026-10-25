#!/usr/bin/env python3
"""FIT-only candidate estimator feasibility study.

Compare SAME-SHOT plug-in resampling with Bayesian posterior-predictive
Dirichlet-multinomial resampling. Method inputs are one observed count bundle;
16 other archived acquisitions are evaluation targets only. True simulator
probabilities, TEST and SELECTION are never opened by this FIT analysis.
No R0/R1/R2 scoring or estimator parameter selection is performed here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

BUDGETS=(4096,8192,16384)
REFS=(0,7,15)
REPS=192
ALPHA=0.5
EPS=.12
RIDGE=.001
METHODS=("plugin_multinomial","jeffreys_posterior_predictive")

def seed(*parts):
    raw=json.dumps(["triqto-observed-posterior-predictive-fit-20261025",*parts],
                   separators=(",",":"),sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16],"big")

def jacobian(counts,table):
    c=np.asarray(counts,dtype=np.float64)
    stats=np.einsum("...o,ok->...k",c/np.maximum(c.sum(axis=-1,keepdims=True),1),table)
    difference=(stats[:,:,0]-stats[:,:,1])/(2*EPS)
    return np.moveaxis(difference,1,-1).reshape(len(c),3*table.shape[1],3)

def coordinates_and_matches(jac,delta,w):
    jw=jac*w[None,:,None]
    norms=np.linalg.norm(jw,axis=1,keepdims=True)
    u=jw/np.maximum(norms,1e-12)
    dw=delta*w
    rhs=np.einsum("ndi,d->ni",u,dw)
    gram=np.einsum("ndi,ndj->nij",u,u)
    coord=np.linalg.solve(gram+RIDGE*np.eye(3),rhs[...,None])[...,0]
    matched=rhs/max(np.linalg.norm(dw),1e-12)
    return np.column_stack([coord,matched])

def predict_cov(counts,table,delta,weights,shots,method,rand):
    observed=np.asarray(counts,dtype=np.int64)
    if observed.shape[:3]!=(3,2,3) or not np.all(observed.sum(-1)==shots):
        raise ValueError("Invalid observed counts/shot contract")
    p=observed.reshape(18,observed.shape[-1]).astype(float)/shots
    if method=="plugin_multinomial":
        sampled=np.stack([rand.multinomial(shots,row,size=REPS) for row in p],axis=1)
    elif method=="jeffreys_posterior_predictive":
        alpha=observed.reshape(18,observed.shape[-1]).astype(float)+ALPHA
        g=rand.gamma(shape=alpha[None,:,:],scale=1.,size=(REPS,*alpha.shape))
        q=g/g.sum(axis=-1,keepdims=True)
        sampled=rand.multinomial(shots,q)
    else:raise ValueError("Unknown candidate estimator")
    frame=jacobian(sampled.reshape(REPS,3,2,3,observed.shape[-1]),table)
    x=coordinates_and_matches(frame,delta,weights)
    cov=np.cov(x,rowvar=False,ddof=1)
    assert np.all(np.isfinite(cov)) and np.linalg.eigvalsh(cov).min()>-1e-7
    return cov

def report(v):
    a=np.asarray(v,dtype=float)
    a=a[np.isfinite(a)]
    return {"n":len(a),"median":float(np.median(a)),
            "p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9))}

def run(corpus):
    roots=json.loads((corpus/"roots.json").read_text())
    chosen={}
    for r in roots:
        if r["partition"]=="FIT":chosen.setdefault(r["family_id"],r)
    if len(chosen)!=120:raise RuntimeError("Expected exactly 120 FIT families")
    rows=[]
    for i,r in enumerate(chosen.values()):
        rid=r["root_id"]
        with np.load(corpus/r["raw_path"],allow_pickle=False) as raw:
            counts=np.asarray(raw["calibration_counts"])[:,:,0]
            table=np.asarray(raw["outcome_features"],float)
        with np.load(corpus/r["derived_path"],allow_pickle=False) as derived:
            delta=np.asarray(derived["delta"][0],float)
            w=np.asarray(derived["weights"][0],float)
            saved=derived["sample_features"]
        for b,shots in enumerate(BUDGETS):
            features=coordinates_and_matches(jacobian(counts[b],table),delta,w)
            actual=np.r_[saved[b,0,0,0,:3],saved[b,0,0,0,9:12]]
            if max(abs(features[0]-actual))>1e-8:raise RuntimeError("Reference reconstruction drift")
            observed_cov=np.cov(features,rowvar=False,ddof=1)
            for method in METHODS:
                covariances=[predict_cov(counts[b,ref],table,delta,w,shots,method,
                                        np.random.default_rng(seed(rid,shots,ref,method)))
                             for ref in REFS]
                p=np.mean(covariances,axis=0)
                for name,sl in (("coords",slice(0,3)),("matched",slice(3,6))):
                    observed_trace=float(np.trace(observed_cov[sl,sl]))
                    predicted_trace=float(np.trace(p[sl,sl]))
                    ratio=observed_trace/max(predicted_trace,1e-30)
                    rows.append({"family":r["family_id"],"budget":shots,"method":method,
                                 "feature":name,"empirical_to_estimate":ratio,
                                 "abs_log_ratio":abs(float(np.log(max(ratio,1e-30)))),
                                 "source_empirical_trace":observed_trace,
                                 "predicted_trace":predicted_trace})
        if (i+1)%30==0:print({"fit_families_processed":i+1},flush=True)
    result={"schema":"triqto.nonlinear_uncertainty_fit_feasibility.v1",
            "data":"FIT only; no SELECTION, TEST, sealed labels, oracle probabilities, model predictions",
            "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
            "families":120,"per_root_candidate":0,"per_root_observation":0,
            "repeat_acquisitions_for_reference_target_only":16,
            "estimator_input":"single observed candidate calibration count table, 18 calls",
            "reference_indices":list(REFS),"resamples":REPS,
            "methods":{"plugin_multinomial":"resample counts from single observed empirical frequencies",
                       "jeffreys_posterior_predictive":"Dirichlet(observed counts + 0.5) posterior probability, then next-shot multinomial"},
            "note":"Posterior predictive includes parameter uncertainty: may overestimate unconditional fixed-p sampling variance, particularly for linear Jacobians. Not a selected stabilization remedy.",
            "by_budget":{}}
    for shots in BUDGETS:
        result["by_budget"][str(shots)]={}
        for method in METHODS:
            result["by_budget"][str(shots)][method]={}
            for feature in ("coords","matched"):
                part=[x for x in rows if x["budget"]==shots and x["method"]==method and x["feature"]==feature]
                result["by_budget"][str(shots)][method][feature]={
                  "empirical_over_estimated":report([x["empirical_to_estimate"] for x in part]),
                  "absolute_log_ratio":report([x["abs_log_ratio"] for x in part])}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=run(a.corpus)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({b:{m:{f:[round(v["empirical_over_estimated"]["median"],3),
            round(v["absolute_log_ratio"]["median"],3)] for f,v in d.items()}
            for m,d in z.items()} for b,z in result["by_budget"].items()},indent=2),flush=True)
