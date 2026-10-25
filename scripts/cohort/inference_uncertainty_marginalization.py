#!/usr/bin/env python3
"""Pure numerical posterior uncertainty over calibration inference, not data labels.

All inputs are a *single* finite-shot observed candidate calibration bundle.
This does not alter observed real features; each replicate only recomputes the
frozen candidate score from a plausible latent calibration probability table.
"""
from __future__ import annotations
import hashlib,json
import numpy as np

JEFFREYS=0.5
DRAWS=64
N_SHOTS=4096
METHODS=("I0","I1","I2")

def stable_seed(*args):
    payload=json.dumps(["triqto-fit-inference-marginalization-v1",*args],sort_keys=True,
                       separators=(",",":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16],"big")

def posterior_calibration_frequencies(observed,method,*,replicates=DRAWS,seed):
    """Shape (replicate, candidate, mechanism axis, sign, basis, outcome)."""
    c=np.asarray(observed)
    if c.ndim!=5 or c.shape[1:4]!=(3,2,3):
        raise ValueError("Expected [candidate,3,2,3,outcome] counts")
    if np.any(c<0) or np.any(c!=np.floor(c)):
        raise ValueError("Calibration counts must be nonnegative integers")
    shots=c.sum(axis=-1)
    if np.any(shots!=shots.flat[0]) or int(shots.flat[0])<100:
        raise ValueError("Calibration requires constant positive N shots")
    if method not in ("I1","I2") or not 2<=replicates<=512:
        raise ValueError("Invalid inference method or Monte Carlo sample count")
    rand=np.random.Generator(np.random.PCG64(seed))
    posterior=c.astype(np.float64)+JEFFREYS
    drawn=rand.gamma(shape=posterior[None,...],scale=1.,
                     size=(replicates,*posterior.shape))
    latent=drawn/drawn.sum(axis=-1,keepdims=True)
    if method=="I1":
        p=latent
    else:
        p=rand.multinomial(int(shots.flat[0]),latent).astype(float)/float(shots.flat[0])
    if not np.isfinite(p).all() or np.max(np.abs(p.sum(-1)-1.))>1e-12:
        raise RuntimeError("Posterior sample mass conservation failure")
    return p

def jacobians_from_frequencies(probabilities,outcome_table,epsilon):
    p=np.asarray(probabilities,dtype=float)
    t=np.asarray(outcome_table,dtype=float)
    if p.ndim!=6 or p.shape[2:5]!=(3,2,3) or p.shape[-1]!=t.shape[0]:
        raise ValueError("Invalid posterior count frequency dimensions")
    stats=np.einsum("...o,ok->...k",p,t)
    j=(stats[...,0,:,:]-stats[...,1,:,:])/(2*float(epsilon))
    return np.moveaxis(j,-3,-1).reshape(len(p),p.shape[1],3*t.shape[1],3)

def softmax3(scores):
    a=np.asarray(scores,dtype=float)
    if a.shape[-1]!=3 or not np.isfinite(a).all():
        raise ValueError("Expected finite three-class logits")
    v=a-np.max(a,axis=-1,keepdims=True)
    e=np.exp(v)
    return e/np.sum(e,axis=-1,keepdims=True)

def marginalized_prediction(pooled_logits):
    scores=np.asarray(pooled_logits,dtype=float)
    if scores.ndim!=2 or scores.shape[1]!=3 or len(scores)<2:
        raise ValueError("Each posterior inference needs at least two three-class replicas")
    p=softmax3(scores)
    average=p.mean(axis=0)
    if not np.all(np.isfinite(average)) or abs(average.sum()-1.)>1e-12:
        raise RuntimeError("Posterior predictive probabilities must sum to one")
    p_safe=np.clip(average,1e-30,1)
    return {"label":int(np.argmax(average)),
            "posterior_class_probabilities":average.tolist(),
            "predictive_entropy":float(-np.sum(p_safe*np.log(p_safe))),
            "max_probability":float(average.max()),
            "replica_vote_disagreement":float(1.-np.max(np.bincount(
                np.argmax(p,axis=1),minlength=3))/len(p))}

if __name__=="__main__":
    raise SystemExit("Library module: import into explicitly FIT-only frozen network pilot.")
