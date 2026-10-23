#!/usr/bin/env python3
"""Frozen-network synthetic interpolation probes; not physically sampled evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from calibration_replay import PortableReplay
from feature_block_exploratory_replay import load,infer

TS=[0.0,0.001,0.01,0.05,0.1,0.25,1.0]
def main(a):
    torch.set_num_threads(1)
    r=PortableReplay(a.evidence,a.manifest)
    out={"interpretation":"exploratory artificial inputs, not finite-shot changes","arms":{}}
    for b in [1,2,4]:
        ds,x=load(a.pilot,b)
        results={str(t):[] for t in TS}
        for j,d in enumerate(ds):
            base=d["features"]
            delta=x[(j+1)%32]-base
            mixed=np.stack([base+t*delta for t in TS])
            logits,cand=infer(r,d,mixed)
            if np.max(np.abs(logits[0]-d["pooled_logits"]))>1.e-4:
                raise RuntimeError("Synthetic zero-step reproducibility failed")
            for idx,t in enumerate(TS):
                feature_delta=(mixed[idx]-base)[d["mask"]]
                candidate_delta=(cand[idx]-d["candidate_logits"])[d["mask"]]
                pooled_delta=logits[idx]-d["pooled_logits"]
                original=np.argmax(d["pooled_logits"],axis=1)
                changed=np.argmax(logits[idx],axis=1)
                margin=np.diff(np.sort(d["pooled_logits"],axis=1)[:,-2:],axis=1).ravel()
                results[str(t)].append({
                    "feature_norm":float(np.linalg.norm(feature_delta)),
                    "candidate_norm":float(np.linalg.norm(candidate_delta)),
                    "pooled_norm":float(np.linalg.norm(pooled_delta)),
                    "flip_fraction":float(np.mean(original!=changed)),
                    "margin_if_flip":float(np.mean(margin[original!=changed])) if np.any(original!=changed) else None})
        out["arms"][str(b)]={str(t):{
            "flip_fraction":float(np.mean([a["flip_fraction"] for a in results[str(t)]])),
            "mean_feature_norm":float(np.mean([a["feature_norm"] for a in results[str(t)]])),
            "mean_candidate_change":float(np.mean([a["candidate_norm"] for a in results[str(t)]])),
            "mean_pooled_change":float(np.mean([a["pooled_norm"] for a in results[str(t)]])),
            "mean_margin_on_flips":float(np.mean([a["margin_if_flip"] for a in results[str(t)] if a["margin_if_flip"] is not None])) if any(a["margin_if_flip"] is not None for a in results[str(t)]) else None} for t in TS}
        print(json.dumps({"finished":b,"flip_by_step":{t:round(x["flip_fraction"],5) for t,x in out["arms"][str(b)].items()}}),flush=True)
    r.verify_unchanged()
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--manifest",required=True)
    p.add_argument("--pilot",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
