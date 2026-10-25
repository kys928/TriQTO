"""Independent blind-corpus audit: no generator/core import and no TEST truth.

This is an independently implemented verification path, not a second human
review. It evaluates arrays and the written scientific specification directly.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

NAMESPACE='triqto-independent-stability-cohort-v1-20261025'
ALLOWED={'calibration_counts','diagnostic_counts','outcome_features','candidates','observation_ids','diagnostic_shots','budgets'}


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return 'sha256:'+hashlib.sha256(raw).hexdigest()
def audit_seed(*v):return int.from_bytes(hashlib.sha256(canonical([NAMESPACE,*v])).digest()[:16],'big')
def read(path):return json.loads(Path(path).read_text())

def validate_raw(z,root,draws):
    if set(z.files)!=ALLOWED:raise ValueError('Unapproved field in blind artifact')
    n=root['n_qubits'];cc=z['calibration_counts'];dc=z['diagnostic_counts']
    if cc.shape!=(3,draws,root['candidate_count'],3,2,3,2**n):raise ValueError('Calibration shape mismatch')
    if dc.shape!=(13,2,3,2**n):raise ValueError('Diagnostic shape mismatch')
    if cc.dtype.kind not in 'iu' or dc.dtype.kind not in 'iu':raise ValueError('Noninteger counts')
    if np.any(cc<0) or np.any(dc<0):raise ValueError('Negative counts')
    budgets=np.array([4096,8192,16384])
    if not np.array_equal(z['budgets'],budgets):raise ValueError('Budget mismatch')
    if not np.all(cc.sum(axis=-1)==budgets[:,None,None,None,None,None]):raise ValueError('Calibration shot mass mismatch')
    if not np.all(dc.sum(axis=-1)==z['diagnostic_shots'][:,None,None]):raise ValueError('Diagnostic shot mass mismatch')
    if not np.isin(z['diagnostic_shots'],[512,1024,2048,4096]).all():raise ValueError('Diagnostic shots outside grammar')
    bits=(np.arange(2**n)[:,None]>>np.arange(n))&1;e=1-2*bits
    table=np.column_stack([e,*[e[:,i]*e[:,j] for i in range(n) for j in range(i+1,n)],e.prod(axis=1)])
    if not np.array_equal(z['outcome_features'],table):raise ValueError('Outcome/wire convention mismatch')
    if not np.array_equal(z['candidates'],root['candidates']):raise ValueError('Candidate ordering mismatch')
    if len(set(z['observation_ids'].tolist()))!=13:raise ValueError('Duplicate observations')
    for field in ['outcome_features','diagnostic_shots']:
        if not np.isfinite(z[field]).all():raise ValueError('Nonfinite raw array')
    return table


def independent_jacobian(counts,table):
    dims=3*table.shape[1];out=np.zeros((dims,3))
    for axis in range(3):
        for basis in range(3):
            a=counts[axis,0,basis];b=counts[axis,1,basis]
            out[basis*table.shape[1]:(basis+1)*table.shape[1],axis]=((a/a.sum())@table-(b/b.sum())@table)/.24
    return out


def independent_canonical(d,j,w):
    a=j*w[:,None];norm=np.sqrt((a*a).sum(axis=0));u=a/np.maximum(norm,1e-12)
    dw=d*w;t=np.linalg.solve(u.T@u+.001*np.eye(3),u.T@dw)
    m=u.T@dw/max(np.linalg.norm(dw),1e-12);s=np.linalg.svd(u,compute_uv=False)
    residual=np.linalg.norm(dw-u@t)/max(np.linalg.norm(dw),1e-12)
    return np.r_[t,abs(t),t/max(np.linalg.norm(t),1e-12),m,abs(m),s,np.log1p(norm),
                  min(50,np.log1p(s[0]/max(s[-1],1e-12))),residual,np.log1p(np.linalg.norm(dw))]


def qc_from_rec(rec):
    q=QuantumCircuit(rec['n_qubits'])
    for name,qs,ps in rec['operations']:getattr(q,name)(*ps,*qs)
    return q


def born(q):
    result=[]
    for basis in ('Z','X','Y'):
        t=QuantumCircuit(q.num_qubits)
        for i in range(q.num_qubits):
            if basis=='Y':t.sdg(i)
            if basis!='Z':t.h(i)
        initial=Statevector.from_instruction(q).data
        initial=initial/np.linalg.norm(initial)
        sv=Statevector(initial).evolve(t).data
        p=abs(sv)**2;p=p/p.sum();result.append(p)
    return np.stack(result)


def independent_candidates(q):
    ops=[(item.operation.name,[q.find_bit(b).index for b in item.qubits]) for item in q.data]
    boundaries={b for k,(name,_) in enumerate(ops) if name in ('cx','cz') for b in (k,k+1) if 1<=b<len(ops)}
    return [[wire,b] for b in sorted(boundaries) for wire in range(q.num_qubits)
            if any(wire in qs for _,qs in ops[:b]) and any(wire in qs for _,qs in ops[b:])]


def sampled_candidate_probabilities(q,candidates):
    a=[]
    for wire,boundary in candidates:
        axes=[]
        for gate in ('rz','rx','ry'):
            signs=[]
            for angle in (.12,-.12):
                z=QuantumCircuit(q.num_qubits)
                for item in q.data[:boundary]:z.append(item.operation,[q.find_bit(w).index for w in item.qubits])
                getattr(z,gate)(angle,wire)
                for item in q.data[boundary:]:z.append(item.operation,[q.find_bit(w).index for w in item.qubits])
                signs.append(born(z))
            axes.append(signs)
        a.append(axes)
    return np.array(a)


def audit(product,out):
    manifest=read(product/'DATASET_MANIFEST.json');roots=read(product/'roots.json');families=read(product/'families.json')
    files=manifest['files'];listed={x['path'] for x in files}
    actual={str(p.relative_to(product)) for p in product.rglob('*') if p.is_file()}-{'DATASET_MANIFEST.json'}
    if listed!=actual:raise RuntimeError('File inventory completeness mismatch')
    for row in files:
        path=product/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path.read_bytes())!=row['sha256']:raise RuntimeError('Content integrity mismatch: '+row['path'])
    expected_id=sha(canonical({'sources':manifest['source_hashes'],'files':files,'namespace':NAMESPACE}))
    if expected_id!=manifest['cohort_identity']:raise RuntimeError('Cohort identity mismatch')
    if len(roots)!=manifest['roots'] or len(families)!=manifest['families']:raise RuntimeError('Inventory size mismatch')
    fammap={f['family_id']:f for f in families}
    if len(fammap)!=len(families):raise RuntimeError('Duplicate family identities')
    for key in ['family_signature_sha256','context_free_signature_sha256','wire_signature_sha256','operation_signature']:
        if len({f[key] for f in families})!=len(families):raise RuntimeError('Structural duplicate across families: '+key)
    if len({r['graph_sha256'] for r in roots})!=len(roots):raise RuntimeError('Duplicate parameterized root')
    if len({r['root_id'] for r in roots})!=len(roots):raise RuntimeError('Duplicate root identities')
    if set(Counter(r['family_id'] for r in roots).values())!={4}:raise RuntimeError('Family variant count mismatch')
    family_counts=Counter((f['partition'],f['sampling_stratum']) for f in families)
    pilot=manifest['status']=='ENGINEERING_PILOT_ONLY'
    if not pilot:
        expected={('FIT','representative'):96,('FIT','stress'):24,('SELECTION','representative'):48,('SELECTION','stress'):12,('TEST','representative'):48,('TEST','stress'):12}
        if dict(family_counts)!=expected:raise RuntimeError('Split/stratum drift')
        if len(list((product/'sealed').glob('*.aesgcm')))!=240:raise RuntimeError('TEST seal incomplete')
        if len((product/'wrapped_test_key.bin').read_bytes())<256:raise RuntimeError('Missing wrapped TEST key')
    old=read(product/'historical-exclusions.json')['identities']
    if {f['family_signature_sha256'] for f in families}&set(old['family_signature_sha256']):raise RuntimeError('Historical family overlap')
    if {r['graph_sha256'] for r in roots}&set(old['graph_sha256']):raise RuntimeError('Historical graph overlap')
    if {f['operation_signature'] for f in families}&set(old['operation_signature']):raise RuntimeError('Historical operation overlap')
    sample_ids={r['root_id'] for r in sorted(roots,key=lambda r:sha(r['root_id'].encode()))[:24]}
    global_oids=set();max_feature_error=0.;replays=0;reconstructed=0
    for i,root in enumerate(roots):
        fam=fammap[root['family_id']]
        if root['partition']!=fam['partition']:raise RuntimeError('Family split leakage')
        if root['partition']=='TEST' and any(x in root for x in ('truth','boundary','affected_qubit','mechanism','strength')):raise RuntimeError('TEST truth in blind root manifest')
        z=np.load(product/root['raw_path'],allow_pickle=False);table=validate_raw(z,root,manifest['draws_per_budget'])
        oids=set(z['observation_ids'].tolist())
        if global_oids&oids:raise RuntimeError('Cross-root observation duplication')
        global_oids.update(oids)
        qc=qc_from_rec(root['circuit'])
        if independent_candidates(qc)!=root['candidates']:raise RuntimeError('Candidate generation mismatch')
        # Input graph supports only original frozen gate set, maximum 17 events.
        if len(qc.data)>17 or any(x.operation.name not in ('h','rx','ry','rz','cx','cz') for x in qc.data):raise RuntimeError('Circuit grammar drift')
        if root['root_id'] in sample_ids:
            p=sampled_candidate_probabilities(qc,root['candidates'])
            replay=np.random.Generator(np.random.PCG64(audit_seed('calibration',root['root_id'],4096,0))).multinomial(4096,p)
            if not np.array_equal(replay,z['calibration_counts'][0,0]):raise RuntimeError('Independent simulator/RNG replay mismatch')
            replays+=1
        if root['derived_path']:
            derived=np.load(product/root['derived_path'],allow_pickle=False)
            if not all(np.isfinite(derived[k]).all() for k in derived.files if derived[k].dtype.kind not in 'SU'):raise RuntimeError('Nonfinite derived values')
            if root['root_id'] in sample_ids:
                for budget in range(3):
                    for draw in (0,manifest['draws_per_budget']-1):
                        for candidate in (0,root['candidate_count']-1):
                            j=independent_jacobian(z['calibration_counts'][budget,draw,candidate],table)
                            np.testing.assert_allclose(j,derived['sample_jacobians'][budget,0 if draw==0 else 1,0 if candidate==0 else 1],atol=1e-12)
                            for obs in (0,12):
                                dc=z['diagnostic_counts'][obs].astype(float)
                                means=np.einsum('abo,ok->abk',dc/dc.sum(axis=-1)[...,None],table)
                                d=(means[1]-means[0]).ravel();w=np.full(d.size,np.sqrt(z['diagnostic_shots'][obs]/2))
                                f=independent_canonical(d,j,w);actual_f=derived['sample_features'][budget,0 if draw==0 else 1,0 if candidate==0 else 1,0 if obs==0 else 1]
                                error=float(abs(f-actual_f).max());max_feature_error=max(error,max_feature_error)
                                np.testing.assert_allclose(f,actual_f,rtol=1e-8,atol=1e-8)
                                reconstructed+=1
        z.close()
        if (i+1)%100==0:print(f'Independently audited {i+1}/{len(roots)} roots',flush=True)
    coverage=manifest['historical_exclusion_coverage_complete']
    result={'schema':'triqto.cohort.independent_audit.v1','cohort_identity':manifest['cohort_identity'],
       'status':'INTERNAL_INTEGRITY_PASS_EXTERNAL_INDEPENDENCE_UNVERIFIED' if not coverage else 'INTERNAL_INTEGRITY_PASS',
       'files_hash_checked':len(files),'families':len(families),'roots':len(roots),'observations':len(global_oids),
       'split_stratum_counts':{k[0]+'/'+k[1]:v for k,v in family_counts.items()},
       'independent_full_count_replays':replays,'independent_canonical_reconstructions':reconstructed,
       'maximum_canonical_absolute_error':max_feature_error,'test_labels_accessed':False,
       'historical_coverage_complete':coverage,'blockers':manifest['unverified_exclusions'],
       'limitations':['independent code path, same investigator and simulator library; not an independent human replication',
                      'reconstruction sample, exhaustive shape/hash/mass checks; not every floating-point cell independently recomputed']}
    Path(out).write_text(json.dumps(result,indent=2)+chr(10));print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.product,args.output)
