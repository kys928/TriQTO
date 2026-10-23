#!/usr/bin/env python3
"""Exploratory fixed-pooling 24-feature block-swap replay; no truth access."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from calibration_replay import PortableReplay
from run_calibration_resampling_pilot import pairwise_disagreement

GROUPS={"A":list(range(9)),"B":list(range(9,15)),"C":list(range(15,24)),
        "AB":list(range(15)),"AC":list(range(9))+list(range(15,24)),
        "BC":list(range(9,24)),"ABC":list(range(24))}

def load(root,budget):
    paths=sorted((root/f"shots_{budget}x").glob("draw_*.npz"))
    assert len(paths)==32, (budget,len(paths))
    keys=("features","mask","profiles","distance","gate","pooled_logits","candidate_logits")
    data=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as a:
            data.append({k:a[k] for k in keys})
    assert all(np.array_equal(d["mask"],data[0]["mask"]) for d in data)
    return data,np.stack([d["features"] for d in data])

def infer(replay,ref,features):
    n=len(features)
    ex=len(ref["features"])
    table={"x":features.reshape((n*ex,)+ref["features"].shape[1:]),
           "mask":np.tile(ref["mask"],(n,1)),
           "profiles":np.tile(ref["profiles"],(n,1,1)),
           "distance":np.tile(ref["distance"],(n,1,1)),
           "gate":np.tile(ref["gate"],(n,1,1))}
    y,c=replay.infer(table)
    return y.reshape((n,ex,3)),c.reshape((n,)+ref["candidate_logits"].shape)

def run(a):
    torch.set_num_threads(1)
    replay=PortableReplay(a.evidence,a.manifest)
    report={"status":"data-informed-exploratory",
            "groups":GROUPS,"budget":{},"selected_examples":36}
    for budget in (1,2,4):
        data,x=load(a.pilot,budget)
        assert x.shape[1]==36 and x.shape[-1]==24
        labels=np.stack([np.argmax(d["pooled_logits"],axis=1) for d in data])
        blocks={k:np.empty((32,32,36),dtype=np.int8) for k in GROUPS}
        max_error=0.
        for j,d in enumerate(data):
            base=d["features"]
            if j==0:
                yp,cp=infer(replay,d,base[None])
                assert np.max(np.abs(yp[0]-d["pooled_logits"]))<1e-4
            for name,cols in GROUPS.items():
                changed=np.repeat(base[None],32,axis=0)
                changed[...,cols]=x[...,cols]
                yp,_=infer(replay,d,changed)
                blocks[name][j]=np.argmax(yp,axis=-1)
                assert np.array_equal(blocks[name][j,j],labels[j])
            if j%8==7:
                print(json.dumps({"budget":budget,"refs_complete":j+1}),flush=True)
        effects={}
        for k,v in blocks.items():
            per_ref=np.stack([pairwise_disagreement(v[j]) for j in range(32)])
            effects[k]={"disagreement":float(per_ref.mean()),
                        "min_ref":float(per_ref.mean(1).min()),
                        "max_ref":float(per_ref.mean(1).max())}
        ref_labels=labels[:,None,:]
        synergy=(blocks["ABC"]!=ref_labels)&(blocks["A"]==ref_labels)&(blocks["B"]==ref_labels)&(blocks["C"]==ref_labels)
        effects["interactions"]={
            "ABC_only_flip_fraction":float(synergy.mean()),
            "ABC_minus_sum_singletons":float(effects["ABC"]["disagreement"]-effects["A"]["disagreement"]-effects["B"]["disagreement"]-effects["C"]["disagreement"])}
        report["budget"][str(budget)]={"original_disagreement":float(pairwise_disagreement(labels).mean()),"blocks":effects}
        print(json.dumps({"completed_budget":budget,"result":report["budget"][str(budget)]}),flush=True)
    replay.verify_unchanged()
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
