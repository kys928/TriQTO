#!/usr/bin/env python3
"""Reduce predetermined FIT alpha-grid shards: no SELECTION or TEST input."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
# Standalone predeclared finite grid; no Qiskit or PyTorch import needed.
ALPHAS={"R1":(.05,.1,.25,.5,1.),"R2":(.01,.03,.1,.25,.5,1.)}
DRAWS=16
def key(remedy,alpha):return f"{remedy}@{alpha:g}"
def disagreement(labels):
    counts=np.bincount(np.asarray(labels,dtype=np.int64),minlength=3)
    return 1.-float(np.sum(counts*(counts-1)//2))/(len(labels)*(len(labels)-1)//2)

def score(rows,method):
    acc=np.mean([np.mean(np.asarray(r["predictions"][method])==r["truth_mechanism"]) for r in rows])
    d=np.mean([disagreement(r["predictions"][method]) for r in rows])
    return {"accuracy":float(acc),"disagreement":float(d),"families":len({r["family_id"] for r in rows}),
            "observations":len(rows)}

def collect(paths):
    shards=sorted((json.loads(p.read_text()) for p in paths), key=lambda x:int(x["family_slice"][0]))
    bounds=[tuple(x["family_slice"]) for x in shards]
    if bounds!=[(8,16),(16,24),(24,32),(32,40)]:
        raise RuntimeError("Expected exact prospective FIT shard ranges")
    all_methods=["R0"]+[key(k,v) for k,alphas in ALPHAS.items() for v in alphas]
    rows=[]
    for shard in shards:
        if shard.get("schema")!="triqto.fit_alpha_grid_shard.v1" or shard.get("scope")!="FIT_ONLY":
            raise ValueError("Unexpected source dataset")
        if shard["methods"]!=all_methods or shard["n_observations"]!=24 or shard["family_count"]!=8:
            raise ValueError("Incomplete shard")
        rows+=shard["rows"]
    if len(rows)!=96 or len({r["family_id"] for r in rows})!=32:
        raise ValueError("Must have exactly 32 different FIT families, 3 observations each")
    return rows

def reduce(paths):
    rows=collect(paths)
    # deterministic index by exactly ordered shard, no per-example alpha adaptation
    tuning=rows[:48]
    screen=rows[48:]
    if set(x["family_id"] for x in tuning)&set(x["family_id"] for x in screen):
        raise RuntimeError("FIT tuning vs screening family collision")
    baseline=score(tuning,"R0")
    methods=["R0"]+[key(k,v) for k,alphas in ALPHAS.items() for v in alphas]
    mt={m:score(tuning,m) for m in methods}
    mv={m:score(screen,m) for m in methods}
    proposal={}
    for name,alpha_values in ALPHAS.items():
        accepted=[]
        for a in alpha_values:
            m=key(name,a)
            if mt[m]["accuracy"]>=mt["R0"]["accuracy"]-.01 and mt[m]["disagreement"]<mt["R0"]["disagreement"]:
                accepted.append((mt[m]["disagreement"],a))
        proposed=(min(accepted)[1] if accepted else min(alpha_values,
                  key=lambda a:(mt[key(name,a)]["disagreement"],a)))
        selected=key(name,proposed)
        proposal[name]={
          "candidate":selected,"alpha":proposed,
          "fit_tuning_met_accuracy_and_stability_heuristic":bool(accepted),
          "fit_screening_nonworse_disagreement":mv[selected]["disagreement"]<mv["R0"]["disagreement"],
          "fit_screening_accuracy_delta":mv[selected]["accuracy"]-mv["R0"]["accuracy"],
          "fit_screening_mechanism_agreement_delta":mv[selected]["disagreement"]-mv["R0"]["disagreement"]}
    return {"schema":"triqto.fit_alpha_grid_analysis.v1",
      "status":"EXPLORATORY_FIT_ONLY_NO_METHOD_SELECTION",
      "families_tuning":16,"families_screening":16,
      "comparison_scope":"Frozen candidate mechanism network and score path only",
      "observations_tuning":len(tuning),"observations_screening":len(screen),
      "candidate_grid":{"R1":list(ALPHAS["R1"]),"R2":list(ALPHAS["R2"])},
      "tuning_performance":mt,"screening_performance":mv,
      "prespecified_proposals":proposal,
      "disjoint_fit_family_proof":True,
      "selection_used":False,"test_used":False,
      "models_changed":False,
      "freeze_status":"PRE_SELECTION_CANDIDATES_ONLY_NOT_FINAL_WINNER",
      "source_shard_sha256":{p.name:"sha256:"+hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-directory",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    shards=sorted(a.input_directory.rglob("fit-grid-*.json"))
    if len(shards)!=4:raise RuntimeError("Expected exactly four FIT shards")
    result=reduce(shards)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"proposal":result["prespecified_proposals"],
      "R0_tuning":result["tuning_performance"]["R0"],
      "R0_screening":result["screening_performance"]["R0"]},indent=2))
