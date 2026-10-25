#!/usr/bin/env python3
"""Four-shard FIT-only alpha-screen over frozen mechanism-head candidate network.

No SELECTION/TEST, no reserves. Shared one-shot bootstrapped six-dimensional
covariance per candidate/obs/draw across all alpha options. Frozen network
weights/normalizers are SHA-verified from read-only non-protected S3.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import core as c
from fit_frozen_remedy_score_pilot import checkpoints,model_logits,pool_one,disagreement
from crossfit_nonlinear_covariance_fit import boot_cov,fixed_seed
from remedy_feature_transforms import calibrated_covariance,apply

ALPHAS={"R1":(.05,.1,.25,.5,1.),"R2":(.01,.03,.1,.25,.5,1.)}
FACTORS=(1.9043089473549872,1.390918830738678)
OBS=3
DRAWS=16

def key(remedy,alpha):return f"{remedy}@{alpha:g}"

def subset(corpus,start,stop):
    roots=json.loads((corpus/"roots.json").read_text())
    by={}
    for r in roots:
        if r["partition"]=="FIT":by.setdefault(r["family_id"],r)
    if len(by)!=120 or start<8 or stop>40 or start>=stop:
        raise RuntimeError("Bad declared FIT cohort slice")
    return list(by.values())[start:stop]

def run(corpus,start,stop):
    models=checkpoints()
    rows=[]
    for i,root in enumerate(subset(corpus,start,stop)):
        with np.load(corpus/root["raw_path"],allow_pickle=False) as raw:
            calibration=np.asarray(raw["calibration_counts"][0])
            diagnostics=np.asarray(raw["diagnostic_counts"])
            table=np.asarray(raw["outcome_features"],np.float64)
            ids=raw["observation_ids"].tolist()
        truth_path=corpus/"development_truth"/"FIT"/(root["root_id"][7:]+".json")
        truth=json.loads(truth_path.read_text())["truth"]
        if ids!=[v["observation_id"] for v in truth]:raise RuntimeError("Truth ordering mismatch")
        chosen=[j for j,record in enumerate(truth) if record["mechanism"] is not None][:OBS]
        if len(chosen)!=OBS:raise RuntimeError("Missing FIT mechanism examples")
        delta,weights=c.diagnostic_delta(diagnostics,table)
        jac=c.counts_to_jacobian(calibration,table)
        fx=c.canonical_batch(delta,jac[None],weights)[0]
        profiles=c.profile_batch(delta,jac[None],weights)[0]
        by_method=["R0"]+[key(k,v) for k,alphas in ALPHAS.items() for v in alphas]
        predictions={p:[[] for _ in chosen] for p in by_method}
        for draw in range(DRAWS):
            pairdist,gate=c.finite_frame_pair_geometry(jac[draw])
            for oi,obs in enumerate(chosen):
                features=fx[draw,:,obs,:]
                score_profiles=profiles[draw,:,obs,:]
                c_list=[]
                for k in range(len(features)):
                    one=boot_cov(calibration[draw,k],table,delta[obs],weights[obs],
                        c.BUDGETS[0],fixed_seed("fit-alpha-grid",root["root_id"],draw,obs,k))
                    c_list.append(calibrated_covariance(one,*FACTORS))
                for method in by_method:
                    if method=="R0":xx=features
                    else:
                        remedy,alpha=method.split("@")
                        xx=np.stack([apply(feat,sigma,remedy,alpha=float(alpha))
                            for feat,sigma in zip(features,c_list)])
                    scores=model_logits(xx,models)
                    pooled=pool_one(scores,score_profiles,pairdist,gate,
                                    tau=.02,frame_temperature=2.)
                    predictions[method][oi].append(int(np.argmax(pooled)))
        for oi,obs in enumerate(chosen):
            rows.append({"family_id":root["family_id"],"root_id":root["root_id"],
                         "observation_id":ids[obs],
                         "truth_mechanism":c.MECHANISMS.index(truth[obs]["mechanism"]),
                         "predictions":{method:predictions[method][oi] for method in by_method}})
        print(json.dumps({"fit_grid_family":start+i+1,"fit_grid_obs_total":len(rows)}),flush=True)
    return {"schema":"triqto.fit_alpha_grid_shard.v1","scope":"FIT_ONLY",
            "family_slice":[start,stop],"family_count":stop-start,
            "n_observations":len(rows),"checkpoint_hash_verified":True,
            "methods":["R0"]+[key(k,v) for k,alphas in ALPHAS.items() for v in alphas],
            "rows":rows}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--start",type=int,required=True)
    p.add_argument("--stop",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    out=run(a.corpus,a.start,a.stop)
    a.output.write_text(json.dumps(out,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({"FIT_ONLY":True,"family_count":out["family_count"],
                      "n_observations":out["n_observations"]}))
