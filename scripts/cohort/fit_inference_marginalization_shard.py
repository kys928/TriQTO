#!/usr/bin/env python3
"""Run inference-stage posterior predictive integration with frozen mechanism heads.

Only FIT [40,72), first root, first two examples of each injected class,
16 historical N-shot acquisitions. Prior SELECTION spent; never opened.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import core as c
from fit_frozen_remedy_score_pilot import checkpoints,model_logits,pool_one
from inference_uncertainty_marginalization import (
    DRAWS,METHODS,stable_seed,posterior_calibration_frequencies,
    jacobians_from_frequencies,marginalized_prediction)

OBS_PER_CLASS=2
ROOT_START=40
ROOT_STOP=72

def select_roots(corpus,start,stop):
    if start%8 or stop-start!=8 or not(ROOT_START<=start<stop<=ROOT_STOP):
        raise ValueError("Only frozen 8-FIT-family shards between 40 and 72")
    roots=json.loads((corpus/"roots.json").read_text())
    fit={}
    for root in roots:
        if root["partition"]=="FIT":fit.setdefault(root["family_id"],root)
    if len(fit)!=120:raise RuntimeError("Frozen cohort FIT manifest changed")
    return list(fit.values())[start:stop]

def selected_observations(truth):
    chosen=[]
    for mechanism in c.MECHANISMS:
        cands=[index for index,row in enumerate(truth)
               if row["mechanism"]==mechanism][:OBS_PER_CLASS]
        if len(cands)!=OBS_PER_CLASS:raise RuntimeError("Missing class-balanced FIT observations")
        chosen+=cands
    return sorted(chosen)

def score_marginalized(latent_jac,delta,weights,obs,profile,distance,gate,models):
    # Unlike feature shrinkage R1/R2, original measured 24D feature vector is
    # NEVER replaced. Only posterior hypothetical vectors enter the marginal
    # predictive integral; original evidence profile and kernel remain frozen.
    features=c.canonical_batch(delta[obs:obs+1],latent_jac[None],
                               weights[obs:obs+1])[0][:,:,0,:]
    reps,candidates,dims=features.shape
    if dims!=24:raise RuntimeError("Frozen canonical 24D feature contract")
    scores=model_logits(features.reshape(reps*candidates,24),models).reshape(reps,candidates,3)
    pooled=np.stack([pool_one(s,profile,distance,gate,
                              tau=.02,frame_temperature=2.) for s in scores])
    return marginalized_prediction(pooled)

def run(corpus,start,stop):
    models=checkpoints()
    rows=[]
    for index,root in enumerate(select_roots(corpus,start,stop),start=start):
        with np.load(corpus/root["raw_path"],allow_pickle=False) as raw:
            calibration=np.asarray(raw["calibration_counts"][0])
            diagnostics=np.asarray(raw["diagnostic_counts"])
            table=np.asarray(raw["outcome_features"],float)
            observation_ids=raw["observation_ids"].tolist()
        truth=json.loads((corpus/"development_truth"/"FIT"/
            (root["root_id"][7:]+".json")).read_text())["truth"]
        if observation_ids!=[x["observation_id"] for x in truth]:
            raise RuntimeError("FIT metadata/truth mismatch")
        chosen=selected_observations(truth)
        delta,weights=c.diagnostic_delta(diagnostics,table)
        observed_jac=c.counts_to_jacobian(calibration,table)
        raw_features=c.canonical_batch(delta,observed_jac[None],weights)[0]
        profiles=c.profile_batch(delta,observed_jac[None],weights)[0]
        out={o:{"family_id":root["family_id"],"root_id":root["root_id"],
            "observation_id":observation_ids[o],
            "truth_mechanism":c.MECHANISMS.index(truth[o]["mechanism"]),
            "predictions":{method:[] for method in METHODS},
            "inference_entropy":{method:[] for method in ("I1","I2")}} for o in chosen}
        for acquisition in range(c.DRAWS):
            distance,gate=c.finite_frame_pair_geometry(observed_jac[acquisition])
            latent_jac={}
            for method in ("I1","I2"):
                p=posterior_calibration_frequencies(calibration[acquisition],method,
                    replicates=DRAWS,seed=stable_seed(root["root_id"],acquisition,method))
                latent_jac[method]=jacobians_from_frequencies(p,table,c.EPSILON)
            for obs in chosen:
                original=raw_features[acquisition,:,obs,:]
                original_scores=model_logits(original,models)
                initial=pool_one(original_scores,profiles[acquisition,:,obs,:],
                                 distance,gate,tau=.02,frame_temperature=2.)
                out[obs]["predictions"]["I0"].append(int(np.argmax(initial)))
                for method in ("I1","I2"):
                    result=score_marginalized(latent_jac[method],delta,weights,obs,
                         profiles[acquisition,:,obs,:],distance,gate,models)
                    out[obs]["predictions"][method].append(result["label"])
                    out[obs]["inference_entropy"][method].append(result["predictive_entropy"])
        for record in out.values():
            if any(len(record["predictions"][m])!=c.DRAWS for m in METHODS):
                raise RuntimeError("Method prediction count incomplete")
            if any(len(record["inference_entropy"][m])!=c.DRAWS for m in ("I1","I2")):
                raise RuntimeError("Missing predictive entropy")
        rows.extend(out.values())
        print(json.dumps({"fit_family_index_done":index,"root_observations":len(out),
                          "total_rows":len(rows),"posterior_draws":DRAWS}),flush=True)
    if len(rows)!=8*3*OBS_PER_CLASS:raise RuntimeError("Incomplete FIT shard")
    return {"schema":"triqto.fit_inference_marginalization_shard.v1",
      "partition":"FIT","family_slice":[start,stop],
      "families":8,"observations":len(rows),"root_variants_per_family":1,
      "calibration_draws_per_observation":c.DRAWS,
      "mc_draws_per_observation_and_acquisition":DRAWS,
      "frozen_checkpoint_seeds":[6201,6202,6203],
      "shot_budget":4096,"no_extra_qpu_shots":True,
      "test_or_selection_access":False,
      "methods":list(METHODS),"rows":rows}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--corpus",required=True,type=Path)
    a.add_argument("--start",required=True,type=int)
    a.add_argument("--stop",required=True,type=int)
    a.add_argument("--output",required=True,type=Path)
    p=a.parse_args()
    result=run(p.corpus,p.start,p.stop)
    p.output.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({"FIT_only":True,"families":result["families"],
        "observations":result["observations"],
        "methods":result["methods"],"test_used":False}))
