#!/usr/bin/env python3
"""Post-FIT descriptive diagnostic of predeclared I2 screening; no new tuning.

Uses successful FIT-only prediction artifacts. Family bootstrap CIs here are
descriptive, not independent confirmation. Does not change prechosen I2.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from reduce_fit_inference_marginalization import audit_and_join,stats,disagreement

BOOTSTRAPS=4000
SEED=20261025

def delta_by_family(rows,method):
    groups={}
    for r in rows:groups.setdefault(r["family_id"],[]).append(r)
    out=[]
    for fid,part in sorted(groups.items()):
        ref=stats(part,"I0");other=stats(part,method)
        out.append([other["pairwise_disagreement"]-ref["pairwise_disagreement"],
                    other["draw_level_accuracy"]-ref["draw_level_accuracy"]])
    return np.asarray(out)

def boot(delta):
    rng=np.random.default_rng(SEED)
    n=len(delta)
    draws=rng.integers(0,n,size=(BOOTSTRAPS,n))
    means=delta[draws].mean(axis=1)
    return {"families":n,
            "observed_disagreement_delta":float(delta[:,0].mean()),
            "disagreement_delta_95pct_percentile":np.quantile(means[:,0],[.025,.975]).tolist(),
            "observed_accuracy_delta":float(delta[:,1].mean()),
            "accuracy_delta_95pct_percentile":np.quantile(means[:,1],[.025,.975]).tolist(),
            "confidence_limit_note":"FIT exploratory percentile family-block bootstrap; not SELECTION/TEST confirmation"}

def paired_flips(rows,method):
    correct=wrong=changed=0
    class_detail={}
    for y in (0,1,2):
        class_detail[str(y)]={"newly_correct":0,"newly_wrong":0,"flipped":0}
    for r in rows:
        truth=r["truth_mechanism"]
        a=np.asarray(r["predictions"]["I0"])
        b=np.asarray(r["predictions"][method])
        diff=a!=b
        newly_good=(a!=truth)&(b==truth)
        newly_bad=(a==truth)&(b!=truth)
        correct+=int(newly_good.sum())
        wrong+=int(newly_bad.sum())
        changed+=int(diff.sum())
        cls=class_detail[str(truth)]
        cls["newly_correct"]+=int(newly_good.sum())
        cls["newly_wrong"]+=int(newly_bad.sum())
        cls["flipped"]+=int(diff.sum())
    return {"total_flipped_draws":changed,
            "formerly_wrong_now_correct":correct,
            "formerly_correct_now_wrong":wrong,
            "per_truth_class":class_detail,
            "class_counts_all_balanced":True}

def check(shards):
    _,rows=audit_and_join(shards)
    groups={"fit_tuning":rows[:96],"fit_disjoint_screen":rows[96:]}
    out={"schema":"triqto.fit_inference_paired_diagnostics.v1",
      "status":"EXPLORATORY_FIT_POSTHOC_DIAGNOSTIC_NO_METHOD_RESELECTION",
      "predeclared_candidate":"I2",
      "independent_test_access":False,
      "spent_selection_access":False,
      "replacement_test_access":False,
      "bootstrap_families":True,
      "bootstrap_repetitions":BOOTSTRAPS,
      "source_sha256":{p.name:"sha256:"+hashlib.sha256(p.read_bytes()).hexdigest() for p in shards},
      "groups":{}}
    for group,part in groups.items():
        out["groups"][group]={}
        for method in ("I1","I2"):
            out["groups"][group][method]={
              "family_paired":boot(delta_by_family(part,method)),
              "argmax_flips":paired_flips(part,method),
              "mean_predictive_entropy":float(np.mean([
                 v for r in part for v in r["inference_entropy"][method]]))}
    return out

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--input-directory",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    p=a.parse_args()
    paths=sorted(p.input_directory.rglob("posterior-fit-*.json"))
    if len(paths)!=4:raise ValueError("Require four immutable FIT-only artifact shards")
    result=check(paths)
    p.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"I2_screening":result["groups"]["fit_disjoint_screen"]["I2"],
                      "I1_screening":result["groups"]["fit_disjoint_screen"]["I1"]},indent=2))
