#!/usr/bin/env python3
"""One-time, entirely SELECTION-partition frozen-head N-shot scoring shard.

Only after an immutable FIT-derived candidate lock is present. No original
TEST, no replacement TEST, no sealed reserves. Produces one N-only prediction
shard of four complete SELECTION families (16 variants * 12 mechanisms).
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import core as c
from fit_frozen_remedy_score_pilot import checkpoints,model_logits,pool_one
from crossfit_nonlinear_covariance_fit import boot_cov,fixed_seed
from remedy_feature_transforms import calibrated_covariance,apply

PARTITION="SELECTION"
N_SHOTS=4096
FAM_PER_SHARD=4
FACTORS=(1.9043089473549872,1.390918830738678)

def load_lock(path):
    lock=json.loads(path.read_text())
    if lock.get("schema")!="triqto.stabilization_pre_selection_lock.v1" or lock.get("status")!="FROZEN_CANDIDATES_BEFORE_SELECTION":
        raise RuntimeError("Missing prospective FIT-only remedy lock")
    alphas=lock.get("candidate_alpha",{})
    if set(alphas)!={"R1","R2"} or not all(0.<float(a)<=1 for a in alphas.values()):
        raise RuntimeError("Missing valid alpha pair")
    if lock.get("selection_inspected_before_lock",True):
        raise RuntimeError("No SELECTION may influence alpha")
    if lock.get("checkpoint_method_id")!="equiv_fit_4df8aea718950bdc2650a03b":
        raise RuntimeError("Frozen mechanism checkpoint lineage mismatch")
    if not lock.get("first_selection_run_intent_once",False):
        raise RuntimeError("Single use SELECTION not declared")
    return lock

def families(corpus,start,stop):
    roots=json.loads((corpus/"roots.json").read_text())
    families={}
    for r in roots:
        if r["partition"]==PARTITION:
            families.setdefault(r["family_id"],[]).append(r)
    if len(families)!=60 or any(len(v)!=4 for v in families.values()):
        raise RuntimeError("Incomplete 60 family x four variant SELECTION manifest")
    if stop-start!=FAM_PER_SHARD or start<0 or stop>60 or start%4:
        raise RuntimeError("Unapproved selection shard")
    return list(families.items())[start:stop]

def process_root(corpus,root,models,alphas):
    with np.load(corpus/root["raw_path"],allow_pickle=False) as z:
        cal=np.asarray(z["calibration_counts"][0])
        diag=np.asarray(z["diagnostic_counts"])
        outcome=z["outcome_features"].astype(float)
        ids=z["observation_ids"].tolist()
    # SELECTION truth is permitted only for method choice, not network input.
    truth=json.loads((corpus/"development_truth"/"SELECTION"/(root["root_id"][7:]+".json")).read_text())["truth"]
    if ids!=[r["observation_id"] for r in truth]:raise RuntimeError("Selection label/public-ID misalignment")
    indexes=[i for i,r in enumerate(truth) if r["mechanism"] is not None]
    if len(indexes)!=12 or [truth[i]["mechanism"] for i in indexes].count(None):
        raise RuntimeError("Bad injected-mechanism coverage")
    names=[x["mechanism"] for x in truth if x["mechanism"] is not None]
    if any(names.count(m)!=4 for m in c.MECHANISMS):
        raise RuntimeError("Missing class-balanced selection root")
    delta,w=c.diagnostic_delta(diag,outcome)
    jac=c.counts_to_jacobian(cal,outcome)
    features=c.canonical_batch(delta,jac[None],w)[0]
    profiles=c.profile_batch(delta,jac[None],w)[0]
    methods=("R0","R1","R2")
    result=[{"family_id":root["family_id"],"root_id":root["root_id"],
             "observation_id":ids[i],
             "truth_mechanism":c.MECHANISMS.index(truth[i]["mechanism"]),
             "predictions":{m:[] for m in methods}} for i in indexes]
    for d in range(16):
        distance,gate=c.finite_frame_pair_geometry(jac[d])
        for ii,oi in enumerate(indexes):
            xx=features[d,:,oi,:]
            pp=profiles[d,:,oi,:]
            covs=[]
            for ci in range(len(xx)):
                raw=boot_cov(cal[d,ci],outcome,delta[oi],w[oi],N_SHOTS,
                     fixed_seed("selection-frozen",root["root_id"],d,oi,ci))
                covs.append(calibrated_covariance(raw,*FACTORS))
            for method in methods:
                transformed=(xx if method=="R0" else np.stack([
                    apply(x,cov,method,alpha=float(alphas[method]))
                    for x,cov in zip(xx,covs)]))
                scores=model_logits(transformed,models)
                pooled=pool_one(scores,pp,distance,gate,tau=.02,frame_temperature=2.)
                result[ii]["predictions"][method].append(int(np.argmax(pooled)))
    if any(len(r["predictions"]["R0"])!=16 for r in result):
        raise RuntimeError("Incomplete frozen method selection calibration")
    return result

def run(corpus,lockfile,start,stop):
    lock=load_lock(lockfile)
    models=checkpoints()
    records=[]
    for fi,(family,variants) in enumerate(families(corpus,start,stop)):
        for root in variants:
            records.extend(process_root(corpus,root,models,lock["candidate_alpha"]))
        print(json.dumps({"selection_families_scored":start+fi+1,"rows_this_shard":len(records)}),flush=True)
    if len(records)!=(stop-start)*4*12:
        raise RuntimeError("Incomplete selection prediction shard")
    return {"schema":"triqto.stabilization_selection_scored_repeats.v1",
       "partition":PARTITION,"test_read":False,
       "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
       "original_test_opened":False,"replacement_test_opened":False,
       "method_selection_lock_sha256":"sha256:"+hashlib.sha256(lockfile.read_bytes()).hexdigest(),
       "family_slice":[start,stop],"rows":records}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--lock",type=Path,required=True)
    p.add_argument("--start",type=int,required=True)
    p.add_argument("--stop",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    obj=run(a.corpus,a.lock,a.start,a.stop)
    a.output.write_text(json.dumps(obj,separators=(",",":"),sort_keys=True)+"\n")
    print(json.dumps({"status":"SELECTION_ONLY","families":a.stop-a.start,
                      "injected_observations":len(obj["rows"]),"test_access":False}))
