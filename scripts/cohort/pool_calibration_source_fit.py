#!/usr/bin/env python3
"""FIT only: pooled observed-calibration source centering K=1/2/4/8.

Comparative diagnosis, not same-shot deployable remedy for K>1. No TEST,
SELECTION, labels, simulator ideal probabilities or model inference.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from compare_nonlinear_estimators_fit import BUDGETS,jacobian,coordinates_and_matches

KS=(1,2,4,8)
REPS=512

def seed(root_id,shots,k):
    raw=json.dumps(["pooled-observed-center-20261025",root_id,shots,k],
                   sort_keys=True,separators=(",",":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16],"big")

def summarize(a):
    a=np.asarray(a)
    return {"n":len(a),"median":float(np.median(a)),
            "p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9))}

def run(corpus):
    roots=json.loads((corpus/"roots.json").read_text())
    first={}
    for root in roots:
        if root["partition"]=="FIT":
            first.setdefault(root["family_id"],root)
    if len(first)!=120:raise RuntimeError("Unexpected FIT family count")
    rows=[]
    for number,root in enumerate(first.values()):
        with np.load(corpus/root["raw_path"],allow_pickle=False) as raw:
            cc=np.asarray(raw["calibration_counts"])[:,:,0]
            table=np.asarray(raw["outcome_features"],dtype=np.float64)
        with np.load(corpus/root["derived_path"],allow_pickle=False) as d:
            delta=np.asarray(d["delta"][0],dtype=np.float64)
            weights=np.asarray(d["weights"][0],dtype=np.float64)
        for bi,shots in enumerate(BUDGETS):
            counts=cc[bi]
            if len(counts)!=16:raise RuntimeError("Wrong repeat budget")
            # Hold-out reference: last 8 independent acquisitions, never used to estimate source p.
            reference=coordinates_and_matches(jacobian(counts[8:16],table),delta,weights)
            empirical=np.cov(reference,rowvar=False,ddof=1)
            for k in KS:
                source=np.sum(counts[:k].astype(np.float64),axis=0)
                probs=source.reshape(18,source.shape[-1])/(shots*k)
                rng=np.random.Generator(np.random.PCG64(seed(root["root_id"],shots,k)))
                simulated=np.stack([rng.multinomial(shots,p,size=REPS) for p in probs],axis=1)
                frame=jacobian(simulated.reshape(REPS,3,2,3,source.shape[-1]),table)
                sampled=coordinates_and_matches(frame,delta,weights)
                predicted=np.cov(sampled,rowvar=False,ddof=1)
                for feature,sl in (("coords",slice(0,3)),("matched",slice(3,6))):
                    e=float(np.trace(empirical[sl,sl]))
                    p=float(np.trace(predicted[sl,sl]))
                    ratio=e/max(p,1e-30)
                    rows.append({"budget":shots,"k":k,"feature":feature,"ratio":ratio,
                                 "abs_log_ratio":abs(np.log(max(ratio,1e-30)))})
        if (number+1)%30==0:print(json.dumps({"processed_fit_families":number+1}),flush=True)
    results={"schema":"triqto.fit_pooled_count_center_diagnostic.v1",
             "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
             "population":"FIT first root/candidate/observation; 120 independent families",
             "protected_test_or_selection_accessed":False,
             "oracle_probabilities_used":False,
             "method":"K independent calibration count sets pooled as empirical source center",
             "development_fit_source_draws_K":list(KS),
             "holdout_reference_draw_indices":list(range(8,16)),
             "pseudo_future_shots_per_call":"original N/2N/4N, 512 simulated resamples",
             "additional_calibration_cost":"K-fold source-estimation shots; not same-shot comparator for K>1",
             "by_budget":{}}
    for b in BUDGETS:
        results["by_budget"][str(b)]={}
        for k in KS:
            results["by_budget"][str(b)][str(k)]={}
            for feat in ("coords","matched"):
                selected=[v for v in rows if v["budget"]==b and v["k"]==k and v["feature"]==feat]
                results["by_budget"][str(b)][str(k)][feat]={
                    "empirical_to_estimate":summarize([v["ratio"] for v in selected]),
                    "abs_log_ratio":summarize([v["abs_log_ratio"] for v in selected])}
    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    results=run(a.corpus)
    a.output.write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps({b:{k:{
        "coords":round(z["coords"]["empirical_to_estimate"]["median"],4),
        "matched":round(z["matched"]["empirical_to_estimate"]["median"],4)}
        for k,z in out.items()} for b,out in results["by_budget"].items()},indent=2),flush=True)
