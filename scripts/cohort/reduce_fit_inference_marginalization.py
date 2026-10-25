#!/usr/bin/env python3
"""Offline FIT inference marginalized score analysis; no repository or S3 input."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

METHODS=("I0","I1","I2")
SLICES=((40,48),(48,56),(56,64),(64,72))

def disagreement(labels):
    x=np.bincount(np.asarray(labels,dtype=np.int64),minlength=3)
    return 1.-float(np.sum(x*(x-1)//2))/(len(labels)*(len(labels)-1)//2)

def stats(rows,method):
    a=np.mean([np.mean(np.asarray(r["predictions"][method])==r["truth_mechanism"]) for r in rows])
    d=np.mean([disagreement(r["predictions"][method]) for r in rows])
    return {"draw_level_accuracy":float(a),"pairwise_disagreement":float(d),
            "families":len({r["family_id"] for r in rows}),"observations":len(rows)}

def audit_and_join(paths):
    files=[json.loads(x.read_text()) for x in paths]
    files.sort(key=lambda x:int(x["family_slice"][0]))
    if [tuple(x["family_slice"]) for x in files]!=list(SLICES):
        raise ValueError("Expected exactly four predeclared FIT family shards")
    rows=[]
    for f in files:
        if (f.get("schema")!="triqto.fit_inference_marginalization_shard.v1" or f.get("partition")!="FIT"
            or f.get("test_or_selection_access") or f.get("methods")!=list(METHODS)
            or f.get("observations")!=48 or f.get("families")!=8
            or f.get("calibration_draws_per_observation")!=16
            or f.get("mc_draws_per_observation_and_acquisition")!=64):
            raise ValueError("Invalid FIT shard provenance or incomplete inference")
        rows.extend(f["rows"])
    if len(rows)!=192 or len({r["family_id"] for r in rows})!=32:
        raise ValueError("Expected 32 disjoint FIT families and 192 mechanism examples")
    ids={(r["family_id"],r["root_id"],r["observation_id"]) for r in rows}
    if len(ids)!=len(rows):raise ValueError("Duplicate FIT observation")
    for r in rows:
        if sorted(r["predictions"])!=sorted(METHODS) or r["truth_mechanism"] not in (0,1,2):
            raise ValueError("Unexpected method or truth")
        if any(len(r["predictions"][m])!=16 for m in METHODS):
            raise ValueError("Incomplete paired predictions")
    return files,rows

def summarize(paths):
    files,rows=audit_and_join(paths)
    training=rows[:96]
    screen=rows[96:]
    if set(r["family_id"] for r in training)&set(r["family_id"] for r in screen):
        raise ValueError("Leakage across FIT train/screen families")
    train={m:stats(training,m) for m in METHODS}
    eval_={m:stats(screen,m) for m in METHODS}
    candidates=[]
    for method in ("I1","I2"):
        better=train["I0"]["pairwise_disagreement"]-train[method]["pairwise_disagreement"]
        accdiff=train[method]["draw_level_accuracy"]-train["I0"]["draw_level_accuracy"]
        if better>=.001 and accdiff>=-.01:candidates.append((better,method))
    proposed=(max(candidates,key=lambda z:(z[0],1 if z[1]=="I1" else 0))[1] if candidates else None)
    def diff(d,method):
        return {"accuracy_delta":d[method]["draw_level_accuracy"]-d["I0"]["draw_level_accuracy"],
                "disagreement_delta":d[method]["pairwise_disagreement"]-d["I0"]["pairwise_disagreement"]}
    deltas={m:{"fit_tuning":diff(train,m),"fit_screen":diff(eval_,m)} for m in ("I1","I2")}
    followup=bool(proposed and deltas[proposed]["fit_screen"]["accuracy_delta"]>=-.01
                   and deltas[proposed]["fit_screen"]["disagreement_delta"]<0)
    return {"schema":"triqto.fit_inference_marginalization_analysis.v1",
            "status":"FIT_ONLY_EXPLORATORY_NOT_A_FROZEN_METHOD",
            "method_candidates":list(METHODS),"posterior_replicates":64,
            "trained_candidate":proposed,
            "supports_fresh_selection_design":followup,
            "reason":("Candidate clears descriptive disjoint FIT screening; independent new selection still required" if followup
                      else "No posterior method demonstrated accuracy-safe stability gain on disjoint FIT screening"),
            "fit_tuning":train,"fit_screen":eval_,"paired_deltas":deltas,
            "original_spent_selection_used":False,"original_test_used":False,
            "replacement_test_used":False,"no_extra_qpu_shots":True,
            "source_shard_sha256":{x.name:"sha256:"+hashlib.sha256(x.read_bytes()).hexdigest() for x in paths}}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input-directory",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    paths=sorted(a.input_directory.rglob("posterior-fit-*.json"))
    if len(paths)!=4:raise ValueError("Require four FIT shard files")
    result=summarize(paths)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"trained_candidate":result["trained_candidate"],
      "fit_tuning":result["fit_tuning"],"fit_screen":result["fit_screen"],
      "paired_deltas":result["paired_deltas"],
      "supports_fresh_selection_design":result["supports_fresh_selection_design"]},indent=2))
