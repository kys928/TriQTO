#!/usr/bin/env python3
"""One-shot family-clustered SELECTION decision gate for frozen R0/R1/R2.

Requires prediction artifact with same family, root, observation & acquisition
draws in all methods. Never consumes original/replace TEST, S3 or model weights.
Outputs only aggregate metrics and a selection decision. No automatic TEST run.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

METHODS=("R0","R1","R2")
DRAWS=16
REPEATS=2000
NI_MARGIN=-0.020
BOOTSTRAP_SEED=20261025

def pair_disagreement(p):
    n=np.bincount(np.asarray(p,dtype=np.int64),minlength=3)
    return 1.-float(np.sum(n*(n-1)//2))/(len(p)*(len(p)-1)//2)

def validate(rows):
    if not isinstance(rows,list) or not rows:raise ValueError("Empty selection predictions")
    all_ids=set();family=set();roots={};mechanisms={}
    for r in rows:
        if set(r)!=set(["family_id","root_id","observation_id","truth_mechanism","predictions"]):
            raise ValueError("Unexpected field in selection record")
        key=(r["family_id"],r["root_id"],r["observation_id"])
        if key in all_ids:raise ValueError("Repeated observation in selection")
        all_ids.add(key);family.add(r["family_id"])
        roots.setdefault(r["family_id"],set()).add(r["root_id"])
        mechanisms.setdefault((r["family_id"],r["root_id"]),[]).append(int(r["truth_mechanism"]))
        if int(r["truth_mechanism"]) not in (0,1,2):raise ValueError("Only 3 injected mechanisms allowed")
        if set(r["predictions"])!=set(METHODS):raise ValueError("Method mismatch")
        for method in METHODS:
            labels=r["predictions"][method]
            if len(labels)!=DRAWS or any(v not in (0,1,2) for v in labels):
                raise ValueError("Each method must score exactly 16 finite-shot repeats")
    if len(family)!=60 or len(rows)!=60*4*12:
        raise ValueError("Require exactly 60 full 4-root x 12-injected selection families")
    if any(len(x)!=4 for x in roots.values()):
        raise ValueError("Every selection family must contain four variants")
    if any(len(v)!=12 or [v.count(k) for k in (0,1,2)]!=[4,4,4] for v in mechanisms.values()):
        raise ValueError("Every root must include exactly 4 of each mechanism")
    return sorted(family)

def metrics(rows,method):
    grouped={}
    for r in rows:
        grouped.setdefault(int(r["truth_mechanism"]),[]).extend(r["predictions"][method])
    ba=float(np.mean([np.mean(np.asarray(grouped[c])==c) if grouped.get(c) else 0.
                      for c in range(3)]))
    diss=float(np.mean([pair_disagreement(r["predictions"][method]) for r in rows]))
    return {"balanced_accuracy":ba,"pairwise_disagreement":diss}

def boot_once(rows):
    groups={}
    for r in rows:groups.setdefault(r["family_id"],[]).append(r)
    names=sorted(groups)
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    base={m:metrics(rows,m) for m in METHODS}
    # Pre-aggregate each complete family once. Drawing the same family in a
    # bootstrap replicate repeats its entire confusion matrix and 16-draw
    # stability events. Identical mathematically to row expansion, but fast.
    family_conf={}
    family_dis={}
    family_weight=[]
    for method in METHODS:
        conf=[];davg=[]
        for name in names:
            items=groups[name]
            matrix=np.zeros((3,3),dtype=np.int64)
            for item in items:
                y=item["truth_mechanism"]
                pred=np.asarray(item["predictions"][method],dtype=np.int64)
                matrix[y]+=np.bincount(pred,minlength=3)
            conf.append(matrix)
            davg.append(float(np.mean([pair_disagreement(x["predictions"][method]) for x in items])))
        family_conf[method]=np.stack(conf)
        family_dis[method]=np.asarray(davg)
    # All 60 cohort families contribute equally (4 variants * 12 injected
    # observations each, 16 draws), so per-family mean disagreement is correct.
    sample=rng.integers(0,len(names),size=(REPEATS,len(names)))
    diffs={m:{"BA":[],"D":[]} for m in ("R1","R2")}
    boot_ba={};boot_dis={}
    for method in METHODS:
        cm=family_conf[method][sample].sum(axis=1)
        denominators=np.maximum(cm.sum(axis=2),1)
        boot_ba[method]=np.mean(np.diagonal(cm,axis1=1,axis2=2)/denominators,axis=1)
        boot_dis[method]=family_dis[method][sample].mean(axis=1)
    for remedy in diffs:
        diffs[remedy]["BA"]=(boot_ba[remedy]-boot_ba["R0"]).tolist()
        diffs[remedy]["D"]=(boot_dis[remedy]-boot_dis["R0"]).tolist()
    details={}
    for remedy in diffs:
        ba=np.asarray(diffs[remedy]["BA"])
        dis=np.asarray(diffs[remedy]["D"])
        # Two possible remediations: 97.5% two-sided Bonferroni simultaneous
        # descriptive intervals -> 1.25% / 98.75% quantile endpoints.
        ba_lower=float(np.quantile(ba,.0125))
        dis_upper=float(np.quantile(dis,.9875))
        qualifies=bool(ba_lower>NI_MARGIN and dis_upper<0)
        details[remedy]={
          "delta_balanced_accuracy":base[remedy]["balanced_accuracy"]-base["R0"]["balanced_accuracy"],
          "delta_pairwise_disagreement":base[remedy]["pairwise_disagreement"]-base["R0"]["pairwise_disagreement"],
          "BA_delta_lower_bonferroni97p5":ba_lower,
          "disagreement_delta_upper_bonferroni97p5":dis_upper,
          "noninferiority_pass":bool(ba_lower>NI_MARGIN),
          "stability_benefit_pass":bool(dis_upper<0),
          "joint_pass":qualifies}
    qualified=[k for k,v in details.items() if v["joint_pass"]]
    winner=("R0" if not qualified else sorted(qualified,key=lambda k:(
      details[k]["delta_pairwise_disagreement"],
      -base[k]["balanced_accuracy"],
      METHODS.index(k)))[0])
    return {"schema":"triqto.stabilization_method_selection_decision.v1",
      "status":"COMPLETE_SELECTION_ONLY_NO_TEST_ACCESS",
      "families":len(names),"records":len(rows),"draws":DRAWS,
      "baseline":base,"contrasts":details,
      "bootstrap_replicates":REPEATS,"bootstrap_unit":"family_id",
      "multiplicity_adjustment":"two comparisons, 97.5% Bonferroni intervals",
      "NI_margin_absolute":NI_MARGIN,"selected_remedy":winner,
      "any_remedy_supported":winner!="R0",
      "no_confirmatory_TEST_access":True,
      "interpretation":"On permitted SELECTION only; new TEST cannot run without subsequent separate signed method freeze."}

def run(inp,out):
    v=json.loads(inp.read_text())
    if v.get("schema")!="triqto.stabilization_selection_scored_repeats.v1":
        raise ValueError("Unknown selection prediction source")
    if v.get("partition")!="SELECTION" or v.get("test_read",True):
        raise ValueError("Refuse any data that are not designated SELECTION-only")
    rows=v["rows"];validate(rows)
    report=boot_once(rows)
    report["predictions_artifact_sha256"]="sha256:"+hashlib.sha256(inp.read_bytes()).hexdigest()
    out.write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--predictions",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    v=a.parse_args()
    out=run(v.predictions,v.output)
    print(json.dumps({k:out[k] for k in ("selected_remedy","any_remedy_supported","families","records")}))
