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
    all_ids=set();family=set()
    for r in rows:
        if set(r)!=set(["family_id","root_id","observation_id","truth_mechanism","predictions"]):
            raise ValueError("Unexpected field in selection record")
        key=(r["family_id"],r["root_id"],r["observation_id"])
        if key in all_ids:raise ValueError("Repeated observation in selection")
        all_ids.add(key);family.add(r["family_id"])
        if int(r["truth_mechanism"]) not in (0,1,2):raise ValueError("Only 3 injected mechanisms allowed")
        if set(r["predictions"])!=set(METHODS):raise ValueError("Method mismatch")
        for method in METHODS:
            labels=r["predictions"][method]
            if len(labels)!=DRAWS or any(v not in (0,1,2) for v in labels):
                raise ValueError("Each method must score exactly 16 finite-shot repeats")
    if len(family)!=60:
        raise ValueError("Require all 60 selection families for final selection")
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
    diffs={m:{"BA":[],"D":[] } for m in ("R1","R2")}
    for _ in range(REPEATS):
        sampled=rng.integers(0,len(names),size=len(names))
        sub=[item for i in sampled for item in groups[names[int(i)]]]
        m={k:metrics(sub,k) for k in METHODS}
        for remedy in diffs:
            diffs[remedy]["BA"].append(m[remedy]["balanced_accuracy"]-m["R0"]["balanced_accuracy"])
            diffs[remedy]["D"].append(m[remedy]["pairwise_disagreement"]-m["R0"]["pairwise_disagreement"])
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
