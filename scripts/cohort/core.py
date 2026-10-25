"""Counts-only cohort mechanics, with source-bound frozen-function reuse.

The source loader evaluates selected function definitions only. It avoids
importing training modules (and torch), and records the exact source hashes.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

ROOT = Path(__file__).resolve().parents[2]
AXES = ('Z','X','Y')
MECHANISMS = ('rz_drift','rx_overrotation','ry_overrotation')
EPSILON = 0.12
RIDGE = 1e-3
DRAWS = 16
BUDGETS = (4096,8192,16384)
NAMESPACE = 'triqto-independent-stability-cohort-v1-20261025'


def digest(raw): return 'sha256:'+hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def identity(value): return digest(canonical(value))
def seed(*parts): return int.from_bytes(hashlib.sha256(canonical([NAMESPACE,*parts])).digest()[:16],'big')
def rng(*parts): return np.random.Generator(np.random.PCG64(seed(*parts)))
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+bytes([10]))

def source_functions(filename, names, namespace):
    path=ROOT/'scripts/v0_2'/filename
    parsed=ast.parse(path.read_text())
    selected=[n for n in parsed.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in selected} != set(names): raise RuntimeError('Missing frozen source definition')
    code=ast.Module(body=selected,type_ignores=[])
    exec(compile(code,str(path),'exec'),namespace)
    return digest(path.read_bytes())

spec=importlib.util.spec_from_file_location('cohort_frozen_base',ROOT/'scripts/v0_2/generate_step5_matched_diagnostic_training_dataset.py')
BASE=importlib.util.module_from_spec(spec);spec.loader.exec_module(BASE)
NS={'np':np,'math':math,'itertools':itertools,'hashlib':hashlib,'json':json,
    'Any':Any,'Mapping':Mapping,'Sequence':Sequence,'QuantumCircuit':QuantumCircuit,
    'Statevector':Statevector,'Pauli':Pauli,'BASE':BASE,'PROBE_SPEC':{'ridge':RIDGE}}
SOURCES={}
SOURCES['generate_step14_cross_motif_dataset.py']=source_functions('generate_step14_cross_motif_dataset.py',
 ['rng_for','topology_edges','family_blueprint','build_variant','op_signature','expectation','ideal_vector','identifiability','strengths_and_shots'],NS)
SOURCES['analyze_step14_latent_frame_inference.py']=source_functions('analyze_step14_latent_frame_inference.py',
 ['operation_qubits','plausible_candidates','profile_log_likelihoods'],NS)
SOURCES['analyze_step14_local_frame_canonicalization.py']=source_functions('analyze_step14_local_frame_canonicalization.py',
 ['canonicalize_evidence'],NS)
SOURCES['step14_equivalence_aware_common.py']=source_functions('step14_equivalence_aware_common.py',
 ['_normalized_columns','_best_nonidentity_assignment_mean','finite_frame_pair_geometry'],NS)
SOURCES['generate_step5_matched_diagnostic_training_dataset.py']=digest((ROOT/'scripts/v0_2/generate_step5_matched_diagnostic_training_dataset.py').read_bytes())
EXPECTED_SOURCES={
 'generate_step14_cross_motif_dataset.py':'sha256:40ac06d1254f94fbb37c3e19a39d18a46a47c899650311faa9a53e7ca50c7950',
 'analyze_step14_latent_frame_inference.py':'sha256:830e4f27606bd8274847c73cac538e2067b176ba0b47502f2264d9b6030397b2',
 'analyze_step14_local_frame_canonicalization.py':'sha256:1b84d9fd206cd0f9079a211fd8b250778daa6a69660a45cf0514057f68524b15',
 'step14_equivalence_aware_common.py':'sha256:1adc47f43f6454dda06cfbaf7b5ac0cf13fa442d1a6710336cc465464696b44a',
 'generate_step5_matched_diagnostic_training_dataset.py':'sha256:0a38742c554c176d68af8fce4631206ad7076bc008b8621b1988df39ea2e3e18'}
if SOURCES!=EXPECTED_SOURCES:raise RuntimeError('Frozen source drift; cohort acquisition refused')
for _name in ['family_blueprint','build_variant','identifiability','strengths_and_shots','plausible_candidates','op_signature','canonicalize_evidence','profile_log_likelihoods','finite_frame_pair_geometry']:
    globals()[_name]=NS[_name]
# Reuse observed-count covariance without importing a training module.
unc_spec=importlib.util.spec_from_file_location('cohort_observed_uncertainty',ROOT/'scripts/v0_2/finite_shot_uncertainty.py')
UNC=importlib.util.module_from_spec(unc_spec);unc_spec.loader.exec_module(UNC)


def circuit_record(qc):
    return {'n_qubits':qc.num_qubits,'operations':[
        [x.operation.name,[qc.find_bit(q).index for q in x.qubits],[float(p) for p in x.operation.params]] for x in qc.data]}

def circuit_from_record(rec):
    qc=QuantumCircuit(rec['n_qubits'])
    for name,qubits,params in rec['operations']:
        getattr(qc,name)(*params,*qubits)
    return qc

def operation_key(qc): return json.dumps(op_signature(qc),separators=(',',':'))
def wire_canonical_key(ops,n):
    # Order-preserving, parameter-free circuit identity up to wire relabeling.
    candidates=[]
    for perm in itertools.permutations(range(n)):
        candidates.append(tuple((name,tuple(perm[q] for q in qs)) for name,qs in ops))
    return identity(min(candidates))

def wire_key_from_signature(value):
    ops=[];mx=0
    for item in json.loads(value):
        name,tail=item.split(':',1)
        qs=tuple(int(q[1:]) for q in tail.split('-'));mx=max(mx,*qs)
        ops.append((name,qs))
    return wire_canonical_key(ops,mx+1)

def outcome_table(n):
    e=BASE.eigenvalue_table(n);pairs=BASE.pair_indices(n)
    return np.column_stack([e,*[e[:,i]*e[:,j] for i,j in pairs],np.prod(e,axis=1)])

def probabilities(qc):
    state=BASE.normalized_state(qc)
    return np.stack([BASE.basis_probabilities(state,qc.num_qubits,b) for b in AXES])

def candidate_probabilities(qc,candidates):
    return np.stack([np.stack([np.stack([
        probabilities(BASE.inject_hidden_rotation(qc,b,q,m,s*EPSILON)) for s in (1,-1)
    ]) for m in MECHANISMS]) for q,b in candidates])

def counts_to_jacobian(counts,table):
    c=np.asarray(counts);shots=c.sum(axis=-1,keepdims=True)
    if np.any(shots<=1) or np.any(c<0) or np.any(c!=np.floor(c)): raise ValueError('Invalid calibration counts')
    stats=np.einsum('...o,ok->...k',c/shots,table)
    # prefix,C,axis,sign,basis,stat -> prefix,C,basis,stat,axis
    j=(stats[...,0,:,:]-stats[...,1,:,:])/(2*EPSILON)
    return np.moveaxis(j,-3,-1).reshape(*j.shape[:-3],3*table.shape[1],3)

def acquire_calibration(p,root_id,draws=DRAWS):
    arrays=[]
    for shots in BUDGETS:
        arrays.append(np.stack([rng('calibration',root_id,shots,d).multinomial(shots,p) for d in range(draws)]))
    return np.asarray(arrays,dtype=np.uint16)

def acquire_diagnostics(qc,bp,root_index,cfg,root_id):
    strengths,shots,clean_shots=strengths_and_shots(root_index,0,cfg)
    specs=[(None,0.,int(clean_shots),None)]
    for k,t in enumerate(strengths):
        specs.extend((m,t,int(shots[k]),k) for m in MECHANISMS)
    order=rng('observation-order',root_id).permutation(13)
    clean_p=probabilities(qc)
    counts=[];metadata=[];truth=[]
    for ordinal,k in enumerate(order):
        mech,strength,sh,bin_id=specs[k]
        oid=identity(['observation',root_id,ordinal])
        obs=qc if mech is None else BASE.inject_hidden_rotation(qc,bp['boundary'],bp['affected_qubit'],mech,strength)
        p=probabilities(obs)
        # RNG streams reference/observed independent, even for clean controls.
        ref=rng('diagnostic-reference',oid).multinomial(sh,clean_p)
        seen=rng('diagnostic-observed',oid).multinomial(sh,p)
        counts.append(np.stack([ref,seen]))
        metadata.append({'observation_id':oid,'shots':sh})
        truth.append({'observation_id':oid,'mechanism':mech,'strength':strength,'strength_bin':bin_id,
                      'affected_qubit':bp['affected_qubit'],'boundary':bp['boundary'],'clean_control':mech is None})
    return np.asarray(counts,dtype=np.uint16),metadata,truth

def diagnostic_delta(counts,table):
    totals=counts.sum(axis=-1)
    means=np.einsum('...o,ok->...k',counts/totals[...,None],table)
    delta=(means[:,1]-means[:,0]).reshape(len(counts),-1)
    weights=np.repeat((1/totals[:,0]+1/totals[:,1])**-.5,table.shape[1],axis=1)
    return delta,weights

def canonical_batch(delta,jac,weights):
    # All observables within a diagnostic observation have the same shot weight.
    if not np.all(weights==weights[:,:1]): raise ValueError('This batch path requires equal per-basis shots')
    norms=np.linalg.norm(jac,axis=-2)
    u=jac/np.maximum(norms,1e-12)[...,None,:]
    gram=np.einsum('...di,...dj->...ij',u,u)
    rhs=np.einsum('...di,ed->...ei',u,delta*weights)
    coords=np.linalg.solve((gram+RIDGE*np.eye(3))[...,None,:,:],rhs[...,None])[...,0]
    cn=np.linalg.norm(coords,axis=-1,keepdims=True)
    unit=coords/np.maximum(cn,1e-12)
    dw=delta*weights;dn=np.linalg.norm(dw,axis=-1)
    matched=rhs/np.maximum(dn,1e-12)[None,None,None,:,None]
    singular=np.linalg.svd(u,compute_uv=False)
    condition=singular[...,0]/np.maximum(singular[...,-1],1e-12)
    residual=np.linalg.norm(dw[None,None,None,:,:]-np.einsum('...di,...ei->...ed',u,coords),axis=-1)/np.maximum(dn,1e-12)
    shape=coords.shape[:-1]
    out=np.concatenate([coords,np.abs(coords),unit,matched,np.abs(matched),
         np.broadcast_to(singular[...,None,:],shape+(3,)),
         np.log1p(norms[...,None,:]*weights[:,0][None,None,None,:,None]),
         np.broadcast_to(np.minimum(50,np.log1p(condition))[...,None,None],shape+(1,)),
         residual[...,None],np.broadcast_to(np.log1p(dn)[None,None,None,:,None],shape+(1,))],axis=-1)
    # returned [budget,draw,candidate,observation,24]
    return out

def profile_batch(delta,jac,weights):
    # Same algebra as frozen signed-amplitude profile, with its epsilon retained.
    jw=jac[...,None,:,:]*weights[:,0][None,None,None,:,None,None]
    dw=(delta*weights)[None,None,None,:,:]
    dots=np.einsum('...dk,...d->...k',jw,dw)
    den=np.sum(jw*jw,axis=-2)+1e-12
    amplitude=dots/den
    resid=dw[...,None]-jw*amplitude[...,None,:]
    return -.5*np.sum(resid*resid,axis=-2)

def bootstrap_joint(counts,table,delta,weights,replicates,bootstrap_seed):
    counts=np.asarray(counts)
    sh=counts.sum(axis=-1)
    if not np.all(sh==sh.flat[0]): raise ValueError('Uniform calibration shot contract violated')
    g=np.random.Generator(np.random.PCG64(bootstrap_seed))
    p=counts/sh[...,None]
    drawn=g.multinomial(int(sh.flat[0]),p,size=(replicates,*p.shape[:-1]))
    j=counts_to_jacobian(drawn,table)
    samples=[]
    for a in j:
        x=canonicalize_evidence(delta,a,weights)[0]
        samples.append(np.r_[x[:3],x[9:12]])
    samples=np.asarray(samples)
    return {'mean':samples.mean(axis=0),'covariance':np.cov(samples,rowvar=False,ddof=1),'samples':samples}
