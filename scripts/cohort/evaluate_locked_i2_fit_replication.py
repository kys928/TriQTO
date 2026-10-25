#!/usr/bin/env python3
"""Fixed I2 vs R0 48-family FIT replication and decision audit; no retuning."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

EXPECTED=[(72,84),(84,96),(96,108),(108,120)]
REPS=4000
SEED=20261025
def disagreement(labels):
    counts=np.bincount(np.asarray(labels,dtype=int),minlength=3)
    return 1.-float(np.sum(counts*(counts-1)//2))/(len(labels)*(len(labels)-1)//2)

def read_all(files):
    inputs=[json.loads(x.read_text()) for x in files]
    inputs.sort(key=lambda v:int(v["family_slice"][0]))
    if [tuple(x["family_slice"]) for x in inputs]!=EXPECTED:
        raise RuntimeError("Wrong locked FIT family slices")
    rows=[]
    for item in inputs:
        if (item.get("schema")!="triqto.locked_fit_i2_replication_shard.v1" or
            item.get("partition")!="FIT" or item.get("test_or_selection_access") or
            item.get("methods")!=["I0","I2"] or item.get("locked_method")!="I2" or
            item.get("families")!=12 or item.get("observations")!=72 or
            item.get("mc_draws_per_observation_and_acquisition")!=64):
            raise RuntimeError("FIT-only frozen I2 replication schema/coverage failure")
        rows.extend(item["rows"])
    by={}
    for r in rows:
        if set(r["predictions"])!={"I0","I2"} or r["truth_mechanism"] not in (0,1,2):
            raise RuntimeError("Unknown method or truth")
        if len(r["predictions"]["I0"])!=16 or len(r["predictions"]["I2"])!=16:
            raise RuntimeError("Expected 16 paired acquisitions per example")
        by.setdefault(r["family_id"],[]).append(r)
    if len(by)!=48 or len(rows)!=288 or any(len(x)!=6 for x in by.values()):
        raise RuntimeError("Expected complete 48 family x 6 injected examples")
    return by

def compute(by):
    paired=[]
    changed_good=changed_bad=changed_any=0
    per_class={str(i):{"wrong_to_right":0,"right_to_wrong":0,"changed":0} for i in range(3)}
    base_dis=[];new_dis=[];base_acc=[];new_acc=[]
    for fid,items in sorted(by.items()):
        bD=float(np.mean([disagreement(x["predictions"]["I0"]) for x in items]))
        aD=float(np.mean([disagreement(x["predictions"]["I2"]) for x in items]))
        bA=float(np.mean([np.mean(np.asarray(x["predictions"]["I0"])==x["truth_mechanism"]) for x in items]))
        aA=float(np.mean([np.mean(np.asarray(x["predictions"]["I2"])==x["truth_mechanism"]) for x in items]))
        base_dis.append(bD);new_dis.append(aD);base_acc.append(bA);new_acc.append(aA)
        paired.append([aD-bD,aA-bA])
        for x in items:
            yy=x["truth_mechanism"]
            a=np.asarray(x["predictions"]["I0"]);b=np.asarray(x["predictions"]["I2"])
            z=str(yy)
            to_right=int(np.sum((a!=yy)&(b==yy)))
            to_wrong=int(np.sum((a==yy)&(b!=yy)))
            diff=int(np.sum(a!=b))
            changed_good+=to_right;changed_bad+=to_wrong;changed_any+=diff
            per_class[z]["wrong_to_right"]+=to_right
            per_class[z]["right_to_wrong"]+=to_wrong
            per_class[z]["changed"]+=diff
    values=np.asarray(paired)
    rng=np.random.Generator(np.random.PCG64(SEED))
    samples=rng.integers(0,len(values),size=(REPS,len(values)))
    bootstrap=values[samples].mean(axis=1)
    limits=np.quantile(bootstrap,[.025,.975],axis=0)
    improve=bool(limits[1,0]<0)
    noninferiority=bool(limits[0,1]>-.02)
    return {"schema":"triqto.fit_locked_i2_replication_verdict.v1",
      "status":"FIT_ONLY_INTERNAL_REPLICATION_NOT_INDEPENDENT_CONFIRMATION",
      "locked_method":"I2","families":len(values),"mechanism_examples":288,
      "draws_per_example":16,"posterior_replicates_per_acquisition":64,
      "source":"FIT 72:120 only (previously untouched); no original/spent SELECTION, old or new TEST",
      "accuracy":{"R0":float(np.mean(base_acc)),"I2":float(np.mean(new_acc)),
                  "I2_minus_R0":float(values[:,1].mean()),
                  "percentile_95pct_ci":limits[:,1].tolist(),
                  "noninferiority_2pp_pass_descriptive":noninferiority},
      "disagreement":{"R0":float(np.mean(base_dis)),"I2":float(np.mean(new_dis)),
                  "I2_minus_R0":float(values[:,0].mean()),
                  "percentile_95pct_ci":limits[:,0].tolist(),
                  "upper_below_zero_descriptive":improve},
      "changed_predictions":{"flips":changed_any,"wrong_to_right":changed_good,
                             "right_to_wrong":changed_bad,"by_true_class":per_class},
      "go_to_fresh_independent_selection_design":bool(improve and noninferiority),
      "decision_note":("FIT replication meets prospective exploratory screen; fresh source-separated development-SELECTION still mandatory" if improve and noninferiority else
        "FIT replication does not meet predeclared screening gates: do not spend a new selection cohort for I2 as-is"),
      "bootstrap":"4000 deterministic family resamples, percentile descriptive; not a confirmatory p-value",
      "protected_sources_accessed":False,"original_spent_selection_accessed":False,
      "replacement_test_accessed":False,"qpu_shots_added":0}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--shards",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    files=sorted(a.shards.rglob("locked-i2-fit-*.json"))
    if len(files)!=4:raise RuntimeError("Expected four immutable FIT-only I2 replication shards")
    rows=read_all(files);result=compute(rows)
    result["shard_sha256"]={x.name:"sha256:"+hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in ("status","families","accuracy","disagreement",
         "changed_predictions","go_to_fresh_independent_selection_design")},indent=2))
