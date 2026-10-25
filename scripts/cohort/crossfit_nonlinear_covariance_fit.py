#!/usr/bin/env python3
"""Exploratory 5-fold FAMILY cross-fit calibration of observed-only 6x6 covariance.

No SELECTION/TEST, no oracle probabilities, no model inference. Fits two global
multipliers (coordinate block, matched block) using FIT repeated-acquisition
targets; predicts a PSD covariance from a SINGLE observed calibration draw.
This is statistical training of uncertainty calibration, NOT training TriQTO.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from compare_nonlinear_estimators_fit import BUDGETS,jacobian,coordinates_and_matches

REFERENCES=(0,7,15)
REPS=192
FOLDS=5
LOW=0.25
HIGH=8.0

def fixed_seed(*args):
    body=json.dumps(["triqto-fit-crossfit-covariance-v1",*args],
                    sort_keys=True,separators=(",",":")).encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:16],"big")

def fold(family_id):
    return fixed_seed("fold",family_id)%FOLDS

def boot_cov(observed,table,delta,weight,shot,seed):
    rng=np.random.Generator(np.random.PCG64(seed))
    probs=observed.reshape(18,-1)/float(shot)
    samples=np.stack([rng.multinomial(shot,p,size=REPS) for p in probs],axis=1)
    j=jacobian(samples.reshape(REPS,3,2,3,-1),table)
    x=coordinates_and_matches(j,delta,weight)
    out=np.cov(x,rowvar=False,ddof=1)
    if not np.isfinite(out).all() or np.linalg.eigvalsh(out).min()<-1.e-7:
        raise ValueError("Nonfinite or non-PSD covariance")
    return out

def trace(c,group):
    s=slice(0,3) if group=="coordinates" else slice(3,6)
    return float(np.trace(c[s,s]))

def median_stats(a):
    a=np.asarray(a,dtype=float)
    return {"n":len(a),"median":float(np.median(a)),
            "p10":float(np.quantile(a,.1)),"p90":float(np.quantile(a,.9))}

def records(corpus):
    roots=json.loads((corpus/"roots.json").read_text())
    selected={}
    for r in roots:
        if r["partition"]=="FIT":selected.setdefault(r["family_id"],r)
    if len(selected)!=120:raise RuntimeError("Expected 120 FIT families")
    all_rows=[]
    for i,(fid,root) in enumerate(selected.items()):
        with np.load(corpus/root["raw_path"],allow_pickle=False) as z:
            raw=z["calibration_counts"][:,:,0]
            table=z["outcome_features"].astype(float)
        with np.load(corpus/root["derived_path"],allow_pickle=False) as d:
            delta=d["delta"][0].astype(float)
            weight=d["weights"][0].astype(float)
        for bi,shots in enumerate(BUDGETS):
            fx=coordinates_and_matches(jacobian(raw[bi],table),delta,weight)
            for ref in REFERENCES:
                emp=np.cov(np.delete(fx,ref,axis=0),rowvar=False,ddof=1)
                estimated=boot_cov(raw[bi,ref],table,delta,weight,shots,fixed_seed(fid,shots,ref))
                if np.min(np.linalg.eigvalsh(estimated))< -1.e-6:
                    raise RuntimeError("Estimated covariance not PSD")
                all_rows.append({"fid":fid,"fold":fold(fid),"shots":shots,"ref":ref,
                                 "emp":emp,"estimated":estimated})
        if (i+1)%30==0:print(json.dumps({"families_processed":i+1}),flush=True)
    return all_rows

def evaluate(all_rows):
    out={"schema":"triqto.observed_only_covariance_family_crossfit.v1",
         "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
         "data":"FIT only: first root/candidate/observation, 120 families, no SELECTION, TEST, oracle probabilities or model inference",
         "sampling":"one 192-replicate plug-in covariance for each individual reference 0,7,15",
         "target":"empirical covariance of 15 independent non-reference acquisition draws",
         "model":"two global PSD-preserving covariance block multipliers; factors trained only on other families",
         "folds":FOLDS,"factor_bounds":[LOW,HIGH],
         "limitations":["Post-hoc FIT crossfit, not an untouched validation set","16 draw covariance estimates are noisy",
                        "Only first candidate/observation, not accuracy or full 24D geometry"],
         "by_budget":{}}
    for shots in BUDGETS:
        rows=[r for r in all_rows if r["shots"]==shots]
        evaluated=[]
        coeffs=[]
        for held in range(FOLDS):
            train=[r for r in rows if r["fold"]!=held]
            test=[r for r in rows if r["fold"]==held]
            if not train or not test:raise RuntimeError("Empty fold")
            ratios={}
            for group in ("coordinates","matched"):
                # Median over distinct families; aggregate three reference ratios within each family.
                byfamily={}
                for r in train:
                    ratio=trace(r["emp"],group)/max(trace(r["estimated"],group),1e-30)
                    byfamily.setdefault(r["fid"],[]).append(ratio)
                raw_factor=float(np.median([np.median(v) for v in byfamily.values()]))
                ratios[group]=max(LOW,min(HIGH,raw_factor))
            coeffs.append({"fold":held,**ratios,"train_families":len(set(r["fid"] for r in train)),
                           "eval_families":len(set(r["fid"] for r in test))})
            diag=np.diag([np.sqrt(ratios["coordinates"])]*3+[np.sqrt(ratios["matched"])]*3)
            for r in test:
                corrected=diag@r["estimated"]@diag
                if np.linalg.eigvalsh(corrected).min()<-1.e-7:
                    raise RuntimeError("PSD broken by block congruence correction")
                for group in ("coordinates","matched"):
                    ref=trace(r["emp"],group)
                    base=trace(r["estimated"],group)
                    new=trace(corrected,group)
                    evaluated.append({"family":r["fid"],"fold":held,"ref":r["ref"],"group":group,
                                      "baseline_ratio":ref/max(base,1e-30),
                                      "calibrated_ratio":ref/max(new,1e-30)})
        aggregates={}
        for group in ("coordinates","matched"):
            use=[r for r in evaluated if r["group"]==group]
            aggregates[group]={
              "uncorrected_ratio":median_stats([r["baseline_ratio"] for r in use]),
              "crossfit_ratio":median_stats([r["calibrated_ratio"] for r in use]),
              "uncorrected_abs_log_error":median_stats([abs(np.log(max(r["baseline_ratio"],1e-30))) for r in use]),
              "crossfit_abs_log_error":median_stats([abs(np.log(max(r["calibrated_ratio"],1e-30))) for r in use])}
        out["by_budget"][str(shots)]={"fold_coefficients":coeffs,"out_of_fold":aggregates}
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=evaluate(records(a.corpus))
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    short={shots:{grp:{k:round(z[k]["median"],4)
       for k in ("uncorrected_ratio","crossfit_ratio","uncorrected_abs_log_error","crossfit_abs_log_error")}
       for grp,z in budget["out_of_fold"].items()} for shots,budget in result["by_budget"].items()}
    print(json.dumps(short,indent=2))
if __name__=="__main__":main()
