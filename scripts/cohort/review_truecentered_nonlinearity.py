#!/usr/bin/env python3
"""Exploratory oracle-centered control for nonlinear covariance discrepancy.

Read only FIT/SELECTION data, acquisition_audit first-candidate probabilities.
Oracle probabilities are used ONLY to diagnose sampling variation, never as
uncertainty-estimator inputs or model features. No TEST arrays or truth.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from review_nonlinear_covariance import jacs, cm_features, bootstrap, q, BUDGETS, REFERENCES, EPS

REPS=256

def seed(rid,shots):
    d=json.dumps(["oracle-centered-nonlinear-control-20261025",rid,shots],
                 separators=(",",":"),sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(d).digest()[:16],"big")

def oracle_sample_frame(p,table,delta,weights,shots,rid):
    rand=np.random.Generator(np.random.PCG64(seed(rid,shots)))
    ordered=p.reshape(18,p.shape[-1])
    c=np.stack([rand.multinomial(shots,pp,size=REPS) for pp in ordered],axis=1)
    return cm_features(jacs(c.reshape(REPS,3,2,3,p.shape[-1]),table),delta,weights)

def min_axis_snr(p,table,shots):
    m=np.einsum("asbo,ok->asbk",p,table)
    expected=np.moveaxis((m[:,0]-m[:,1])/(2*EPS),0,-1).reshape(-1,3)
    signal=np.linalg.norm(expected,axis=0)
    variance=np.zeros(3)
    for axis in range(3):
        for sign in range(2):
            for basis in range(3):
                probabilities=p[axis,sign,basis]
                moment=probabilities @ (table*table)
                mu=probabilities @ table
                variance[axis]+=float(np.sum(moment-mu*mu))/shots/(2*EPS)**2
    return float(np.min(signal / np.maximum(np.sqrt(variance),1.e-12)))

def trace(a,axis):
    return float(np.trace(a[:3,:3] if axis=="coordinates" else a[3:,3:]))

def run(corpus):
    roots=json.loads((corpus/"roots.json").read_text())
    chosen={}
    for r in roots:
        if r["partition"] in ("FIT","SELECTION"):
            chosen.setdefault(r["family_id"],r)
    if len(chosen)!=180:raise RuntimeError("Unexpected development family count")
    rows=[]
    for index,root in enumerate(chosen.values()):
        rid=root["root_id"]
        with np.load(corpus/root["raw_path"],allow_pickle=False) as raw:
            raw_counts=np.asarray(raw["calibration_counts"])[:,:,0]
            table=np.asarray(raw["outcome_features"],dtype=float)
        with np.load(corpus/root["derived_path"],allow_pickle=False) as d:
            delta=np.asarray(d["delta"][0],dtype=float)
            weights=np.asarray(d["weights"][0],dtype=float)
        with np.load(corpus/"acquisition_audit"/root["partition"]/(rid[7:]+".npz"),
                     allow_pickle=False) as a:
            true_p=np.asarray(a["first_candidate_probabilities"],dtype=float)
        for budget_idx,shots in enumerate(BUDGETS):
            observed=cm_features(jacs(raw_counts[budget_idx],table),delta,weights)
            empirical=np.cov(observed,rowvar=False,ddof=1)
            oracle=np.cov(oracle_sample_frame(true_p,table,delta,weights,shots,rid),rowvar=False,ddof=1)
            plugin=[]
            for ref in REFERENCES:
                plugin.append(bootstrap(raw_counts[budget_idx,ref],table,delta,weights,shots,
                                        seed(rid,shots+ref+300000)))
            plugin_mean=np.mean(plugin,axis=0)
            snr=min_axis_snr(true_p,table,shots)
            cls="lt1" if snr<1 else "1to3" if snr<3 else "ge3"
            row={"budget":shots,"class":cls,"min_snr":snr}
            for axis in ("coordinates","matched"):
                e=trace(empirical,axis)
                o=trace(oracle,axis)
                p=trace(plugin_mean,axis)
                row.update({axis+"_emp_over_oracle":e/max(o,1e-30),
                            axis+"_plugin_over_oracle":p/max(o,1e-30),
                            axis+"_emp_over_plugin":e/max(p,1e-30)})
            rows.append(row)
        if (index+1)%45==0:print(json.dumps({"completed_families":index+1}),flush=True)
    cols=["coordinates_emp_over_oracle","coordinates_plugin_over_oracle",
          "coordinates_emp_over_plugin","matched_emp_over_oracle",
          "matched_plugin_over_oracle","matched_emp_over_plugin"]
    def aggregated(group):
        return {col:q([r[col] for r in group]) for col in cols}
    result={"schema":"triqto.oracle_centered_nonlinear_covariance_exploratory.v1",
            "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
            "test_used":False,"oracle_inputs_entered_deployable_estimator":False,
            "oracle_usage":"analysis-only finite-shot Monte Carlo target, never inference",
            "families":180,"observed_draws":16,"oracle_centered_draws":REPS,
            "plugin_draws_per_reference":128,"plugin_reference_indices":list(REFERENCES),
            "by_budget":{}}
    for b in BUDGETS:
        sub=[x for x in rows if x["budget"]==b]
        result["by_budget"][str(b)]={"all":aggregated(sub),
          "min_axis_snr_strata":{k:{"roots":sum(x["class"]==k for x in sub),
                                     "stats":aggregated([x for x in sub if x["class"]==k])}
                                 for k in ("lt1","1to3","ge3")},
          "min_snr_distribution":q([x["min_snr"] for x in sub])}
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=run(a.corpus)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"root_count":180,
      "summary":{b:{k:round(v["all"][k]["median"],5) for k in (
          "coordinates_emp_over_oracle","coordinates_plugin_over_oracle","coordinates_emp_over_plugin",
          "matched_emp_over_oracle","matched_plugin_over_oracle")}
          for b,v in result["by_budget"].items()}},indent=2),flush=True)
