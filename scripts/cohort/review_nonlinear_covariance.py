#!/usr/bin/env python3
"""Exploratory nonlinear covariance audit on FIT/SELECTION, never TEST.

Independent algebra; counts-only plug-in estimator, with ideal probabilities
used STRICTLY for post-hoc explanation of near-null true Jacobian directions.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

BUDGETS = (4096,8192,16384)
REFERENCES = (0,7,15)
REPLICATES = 128
EPS = 0.12
RIDGE = 0.001

def jacs(counts,table):
    """[draw,axis,sign,basis,outcome] -> [draw,stat,basis,axis]."""
    c=np.asarray(counts,dtype=np.float64)
    means=np.einsum("...o,ok->...k",c / c.sum(axis=-1,keepdims=True),table)
    diff=(means[:,:,0]-means[:,:,1])/(2*EPS)
    return np.moveaxis(diff,1,-1).reshape(len(c),3*table.shape[1],3)

def cm_features(j,delta,weights):
    jw=np.asarray(j)*weights[None,:,None]
    norms=np.linalg.norm(jw,axis=1,keepdims=True)
    u=jw/np.maximum(norms,1e-12)
    dw=delta*weights
    rhs=np.einsum("ndi,d->ni",u,dw)
    gram=np.einsum("ndi,ndj->nij",u,u)
    coord=np.linalg.solve(gram+RIDGE*np.eye(3),rhs[...,None])[...,0]
    matched=rhs/max(np.linalg.norm(dw),1e-12)
    return np.column_stack([coord,matched])

def root_seed(root_id,budget,reference):
    raw=json.dumps(["nonlinear-variance-independent-20261025",root_id,budget,reference],
                    separators=(",",":"),sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16],"big")

def bootstrap(counts,table,delta,weights,shots,seed):
    c=np.asarray(counts,dtype=np.int64)
    assert c.shape[:3]==(3,2,3)
    assert np.all(c.sum(axis=-1)==shots)
    r=np.random.Generator(np.random.PCG64(seed))
    v=c.reshape(18,c.shape[-1]).astype(float)/shots
    samples=np.stack([r.multinomial(shots,p,size=REPLICATES) for p in v],axis=1)
    frames=jacs(samples.reshape(REPLICATES,3,2,3,c.shape[-1]),table)
    f=cm_features(frames,delta,weights)
    return np.cov(f,rowvar=False,ddof=1)

def q(v):
    a=np.asarray(v,dtype=float)
    a=a[np.isfinite(a)]
    if not len(a):return None
    return {"n":len(a),"median":float(np.median(a)),
            "p10":float(np.quantile(a,0.1)),"p90":float(np.quantile(a,0.9)),
            "above_1p5":float(np.mean(a>1.5)),"above_2":float(np.mean(a>2))}

def run(root_dir):
    roots=json.loads((root_dir/"roots.json").read_text())
    chosen={}
    for r in roots:
        if r["partition"] not in ("FIT","SELECTION"):continue
        chosen.setdefault(r["family_id"],r)
    if len(chosen)!=180:raise RuntimeError("Expected exactly 180 development families")
    rows=[]
    for i,root in enumerate(chosen.values()):
        rid=root["root_id"]
        path=root_dir/root["raw_path"]
        with np.load(path,allow_pickle=False) as raw:
            cc=np.asarray(raw["calibration_counts"])[:,:,0] # [budget,draw,axis,sign,basis,outcome]
            table=np.asarray(raw["outcome_features"],dtype=float)
        with np.load(root_dir/root["derived_path"],allow_pickle=False) as derived:
            delta=np.asarray(derived["delta"][0],dtype=float)
            weights=np.asarray(derived["weights"][0],dtype=float)
            stored=np.asarray(derived["sample_features"])
        with np.load(root_dir/"acquisition_audit"/root["partition"]/(rid[7:]+".npz"),
                     allow_pickle=False) as audit:
            target=np.asarray(audit["first_candidate_probabilities"],dtype=float)
        means=np.einsum("asbo,ok->asbk",target,table)
        true_j=np.moveaxis((means[:,0]-means[:,1])/(2*EPS),0,-1).reshape(3*table.shape[1],3)
        minimum=float(np.linalg.norm(true_j,axis=0).min())
        null_class="near_null" if minimum<0.1 else "weak" if minimum<0.3 else "nonweak"
        for b,shots in enumerate(BUDGETS):
            frames=jacs(cc[b],table)
            features=cm_features(frames,delta,weights)
            saved=np.r_[stored[b,0,0,0,:3],stored[b,0,0,0,9:12]]
            if np.max(abs(features[0]-saved))>1e-8:
                raise RuntimeError("Independent coordinate/matched reconstruction mismatch")
            empirical=np.cov(features,rowvar=False,ddof=1)
            ec=float(np.trace(empirical[:3,:3]))
            em=float(np.trace(empirical[3:,3:]))
            ref_ratios=[]
            for reference in REFERENCES:
                boot=bootstrap(cc[b,reference],table,delta,weights,shots,
                               root_seed(rid,shots,reference))
                bc=float(np.trace(boot[:3,:3]));bm=float(np.trace(boot[3:,3:]))
                ref_ratios.append((ec/max(bc,1e-30),em/max(bm,1e-30),bc,bm))
            rows.append({
                "family_id":root["family_id"],"budget":shots,"null_class":null_class,
                "true_min_axis_norm":minimum,
                "empirical_coordinate_trace":ec,"empirical_matched_trace":em,
                "coordinate_ratios":[x[0] for x in ref_ratios],
                "matched_ratios":[x[1] for x in ref_ratios],
                "reference_coordinate_trace":[x[2] for x in ref_ratios],
                "reference_matched_trace":[x[3] for x in ref_ratios],
                "coordinate_ratio_three_ref_average":ec/max(float(np.mean([x[2] for x in ref_ratios])),1e-30),
                "matched_ratio_three_ref_average":em/max(float(np.mean([x[3] for x in ref_ratios])),1e-30),
                "coordinate_reference_trace_cv":float(np.std([x[2] for x in ref_ratios],ddof=1)/max(np.mean([x[2] for x in ref_ratios]),1e-30))
            })
        if (i+1)%30==0:print(json.dumps({"completed_development_families":i+1}),flush=True)
    # Report aggregate only: avoid new model scores or selection statistics.
    result={"schema":"triqto.nonlinear_covariance_exploratory.v1",
            "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
            "population":"180 FIT/SELECTION families; first root, first candidate, first observation",
            "independent_calibration_draws":16,"bootstrap_replicates":REPLICATES,
            "predeclared_reference_indices":list(REFERENCES),
            "test_records_opened":0,"sealed_labels_opened":False,
            "note":"Analysis-only exact Jacobian norm for mechanism investigation; never bootstrap estimator input",
            "by_budget":{}}
    for b in BUDGETS:
        sub=[r for r in rows if r["budget"]==b]
        def summary(items):
            return {"coordinate_ratio_ref0":q([r["coordinate_ratios"][0] for r in items]),
              "coordinate_ratio_all_three_reference_values":q([x for r in items for x in r["coordinate_ratios"]]),
              "coordinate_ratio_three_reference_average":q([r["coordinate_ratio_three_ref_average"] for r in items]),
              "matched_ratio_ref0":q([r["matched_ratios"][0] for r in items]),
              "matched_ratio_three_reference_average":q([r["matched_ratio_three_ref_average"] for r in items]),
              "reference_coordinate_trace_cv":q([r["coordinate_reference_trace_cv"] for r in items]),
              "empirical_coordinate_trace":q([r["empirical_coordinate_trace"] for r in items]),
              "empirical_matched_trace":q([r["empirical_matched_trace"] for r in items])}
        result["by_budget"][str(b)]={
            "all":summary(sub),
            "true_axis_norm_strata":{s:summary([r for r in sub if r["null_class"]==s])
                                     for s in ("near_null","weak","nonweak")}
        }
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    out=run(args.corpus)
    args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"families":180,"by_budget":{k:{
         "coordinate_ref0":v["all"]["coordinate_ratio_ref0"]["median"],
         "coordinate_three_ref":v["all"]["coordinate_ratio_three_reference_average"]["median"],
         "matched_three_ref":v["all"]["matched_ratio_three_reference_average"]["median"],
         "null_strata":{s:None if x["coordinate_ratio_three_reference_average"] is None
                         else [x["coordinate_ratio_three_reference_average"]["n"],
                               x["coordinate_ratio_three_reference_average"]["median"]]
                        for s,x in v["true_axis_norm_strata"].items()}}
         for k,v in out["by_budget"].items()}},indent=2),flush=True)
