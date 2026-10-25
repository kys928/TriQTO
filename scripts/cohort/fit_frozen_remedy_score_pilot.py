#!/usr/bin/env python3
"""Bounded FIT-only real frozen candidate-network score-path remedy pilot.

Never reads SELECTION/TEST, reserves, or QPU; model checkpoint bytes must match
the final Step-14 equivalence-aware freeze before trusted deserialization.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np
import torch
from botocore.config import Config
import boto3
import core as c
from crossfit_nonlinear_covariance_fit import boot_cov,fixed_seed
from remedy_feature_transforms import calibrated_covariance,apply

# Extract only frozen pooling function AST, avoiding legacy training imports.
# core.py already SHA256 verifies the complete frozen source module.
c.source_functions("step14_equivalence_aware_common.py",
                   ["_logsumexp","_softmax_logweights","equivalence_aware_pool_one"],c.NS)
pool_one=c.NS["equivalence_aware_pool_one"]

SEEDS=(6201,6202,6203)
METHOD="equiv_fit_4df8aea718950bdc2650a03b"
CHECKSUM={
  6201:"sha256:861f9cd1d20b2b3d53d5a04d969e06bb0d539e59084bc3e47ff34c9a0ed8d416",
  6202:"sha256:2469148e71d8fa7b200edac65adc4c0d0c745dd0ab9ffc37cf915216985c56c8",
  6203:"sha256:4bd586b015b0a9aaa863d856108da2552eee3e62259574e1adb6f60ee1e3c40f"}
FACTORS=(1.9043089473549872,1.390918830738678)

def checkpoints():
    env=os.environ
    s3=boto3.client("s3",endpoint_url=env["RUNPOD_S3_ENDPOINT"],region_name=env["RUNPOD_DATACENTER_ID"].lower(),
      aws_access_key_id=env["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=env["RUNPOD_S3_SECRET_ACCESS_KEY"],
      config=Config(signature_version="s3v4",connect_timeout=15,read_timeout=30))
    models=[]
    for seed in SEEDS:
        key=f"triqto-data/step14_equivalence_aware_development/{METHOD}/candidate_network_seed{seed}.pt"
        from io import BytesIO
        raw=s3.get_object(Bucket=env["RUNPOD_NETWORK_VOLUME_ID"],Key=key)["Body"].read()
        actual="sha256:"+hashlib.sha256(raw).hexdigest()
        if actual!=CHECKSUM[seed]:raise RuntimeError("Frozen checkpoint SHA256 mismatch")
        payload=torch.load(BytesIO(raw),map_location="cpu",weights_only=False)
        if payload["method_id"]!=METHOD or payload["seed"]!=seed or not payload.get("fit_only"):
            raise RuntimeError("Checkpoint metadata mismatch")
        sd=payload["state_dict"]
        weights=[np.asarray(sd[k],np.float32) for k in ("net.0.weight","net.0.bias",
                                                             "net.3.weight","net.3.bias")]
        mean=np.asarray(payload["normalization_mean"],np.float32)
        std=np.asarray(payload["normalization_std"],np.float32)
        if mean.shape!=(24,) or std.shape!=(24,) or np.any(std<=0):
            raise RuntimeError("Frozen input normalizer drift")
        models.append({"weights":weights,"mean":mean,"std":std,"seed":seed})
    return models

def model_logits(x,models):
    x=np.asarray(x,np.float32)
    out=[]
    for model in models:
        w1,b1,w2,b2=model["weights"]
        y=(x-model["mean"])/model["std"]
        h=np.maximum(y@w1.T+b1,0)
        z=h@w2.T+b2
        out.append(z.astype(np.float64))
    return np.mean(np.stack(out),axis=0)

def disagreement(labels):
    labels=np.asarray(labels,dtype=np.int64)
    if len(labels)<2:return 0.
    counts=np.bincount(labels,minlength=3)
    agree=sum(int(n)*(int(n)-1)//2 for n in counts)
    pairs=len(labels)*(len(labels)-1)//2
    return 1.-agree/pairs

def run(corpus):
    roots=json.loads((corpus/"roots.json").read_text())
    fit={}
    for r in roots:
        if r["partition"]=="FIT":fit.setdefault(r["family_id"],r)
    if len(fit)!=120:raise RuntimeError("FIT family size drift")
    models=checkpoints()
    results=[]
    for pos,r in enumerate(list(fit.values())[:8]):
        with np.load(corpus/r["raw_path"],allow_pickle=False) as z:
            calibration=z["calibration_counts"][0]
            diagnostics=z["diagnostic_counts"]
            table=z["outcome_features"].astype(float)
            ids=z["observation_ids"].tolist()
        with (corpus/"development_truth"/"FIT"/(r["root_id"][7:]+".json")).open() as f:
            truth=json.load(f)["truth"]
        if [str(v) for v in ids] != [v["observation_id"] for v in truth]:
            raise RuntimeError("FIT truth ID mismatch")
        selection=[i for i,t in enumerate(truth) if t["mechanism"] is not None][:3]
        # FIT labels select 3 non-clean examples; these are *not* a method
        # hyperparameter selection or input to the inference algorithm.
        delta,weights=c.diagnostic_delta(diagnostics,table)
        jac=c.counts_to_jacobian(calibration)
        fx=c.canonical_batch(delta,jac[None],weights)[0]
        pr=c.profile_batch(delta,jac[None],weights)[0]
        mechanisms={v:i for i,v in enumerate(c.MECHANISMS)}
        preds={method:[] for method in ("R0","R1","R2")}
        for obs in selection:
            values={method:[] for method in preds}
            for draw in range(c.DRAWS):
                pair_dist,pair_gate=c.finite_frame_pair_geometry(jac[draw])
                original=fx[draw,:,obs,:]
                profiles=pr[draw,:,obs,:]
                for method in preds:
                    if method=="R0":
                        xx=original
                    else:
                        transformed=[]
                        for k in range(len(original)):
                            cov=boot_cov(calibration[draw,k],table,delta[obs],weights[obs],
                                       c.BUDGETS[0],fixed_seed("fit-score",r["root_id"],draw,obs,k))
                            corrected=calibrated_covariance(cov,*FACTORS)
                            transformed.append(apply(original[k],corrected,method))
                        xx=np.stack(transformed)
                    logits=model_logits(xx,models)
                    pooled=pool_one(
                        logits,profiles,pair_dist,pair_gate,tau=.02,frame_temperature=2.)
                    values[method].append(int(np.argmax(pooled)))
            for method in preds:
                preds[method].append({"truth":mechanisms[truth[obs]["mechanism"]],
                   "disagreement":disagreement(values[method]),
                   "accuracy":np.mean(np.asarray(values[method])==mechanisms[truth[obs]["mechanism"]]),
                   "argmax_histogram":np.bincount(values[method],minlength=3).tolist()})
        results.append({"family_id":r["family_id"],"mechanism_obs_evaluated":len(selection),
                        "methods":preds})
        print(json.dumps({"fit_families_processed":pos+1,"frozen_network_seeds":len(models)}),flush=True)
    report={"schema":"triqto.fit_frozen_model_remedy_score_pilot.v1",
       "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
       "method":METHOD,"checkpoint_hashes":CHECKSUM,
       "partitions_read":["FIT"],"families":len(results),
       "example_count":sum(r["mechanism_obs_evaluated"] for r in results),
       "draws_per_example":c.DRAWS,"shot_budget":c.BUDGETS[0],
       "variance_scales":{"coordinates":FACTORS[0],"matched":FACTORS[1]},
       "selection_test_or_protected_access":False,"model_weights_changed":False,
       "scope":"candidate-network score path only; frozen equivalence pooling, not Step-15 rank-adapter path",
       "performance":{}}
    for method in ("R0","R1","R2"):
        examples=[row for item in results for row in item["methods"][method]]
        report["performance"][method]={
          "mean_disagreement":float(np.mean([r["disagreement"] for r in examples])),
          "mean_draw_level_accuracy":float(np.mean([r["accuracy"] for r in examples])),
          "n_examples":len(examples)}
    report["FIT_ONLY_EXPLORATORY_NOT_FOR_METHOD_SELECTION"]=True
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    out=run(a.corpus)
    a.out.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out["performance"],indent=2))
