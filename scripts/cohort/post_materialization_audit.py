"""Additional non-selecting identity, development-truth and structural audit.

Never decrypts TEST. This supplements the acquisition-time audit without
altering, filtering, regenerating or relabeling any materialized observation.
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import audit


def wire_identity(sig):
    ops=[];mx=0
    for item in json.loads(sig):
        name,qs=item.split(':');w=tuple(int(x[1:]) for x in qs.split('-'));mx=max(mx,*w);ops.append((name,w))
    return audit.sha(audit.canonical(min(tuple((n,tuple(p[q] for q in qs)) for n,qs in ops) for p in itertools.permutations(range(mx+1)))))


def run(product,extra,output):
    output.mkdir(parents=True,exist_ok=True)
    roots=audit.read(product/'roots.json');families=audit.read(product/'families.json');manifest=audit.read(product/'DATASET_MANIFEST.json');old=audit.read(extra)
    identities=old['identities']
    overlaps={
      'additional_family_signature':len({f['family_signature_sha256'] for f in families}&set(identities['family_signature_sha256'])),
      'additional_graph_sha256':len({r['graph_sha256'] for r in roots}&set(identities['graph_sha256'])),
      'additional_operation_signature':len({f['operation_signature'] for f in families}&set(identities['operation_signature'])),
      'additional_wire_relabelled_signature':len({f['wire_signature_sha256'] for f in families}&{wire_identity(x) for x in identities['operation_signature']})}
    original=audit.read(product/'historical-exclusions.json')['identities']
    overlaps['original_wire_relabelled_signature_independent']=len({f['wire_signature_sha256'] for f in families}&{wire_identity(x) for x in original['operation_signature']})
    errors=[];replays=0;truth_checked=0;count_rows=[];rootarray_ids=[]
    sample={r['root_id'] for r in sorted([r for r in roots if r['partition']!='TEST'],key=lambda r:audit.sha(r['root_id'].encode()))[:12]}
    for root in roots:
        # Blind raw file hash and known-circuit identity only: no TEST geometry or truth.
        rootarray_ids.append({'root_id':root['root_id'],'graph_sha256':root['graph_sha256'],'raw_sha256':root['raw_sha256']})
        if root['partition']=='TEST':continue
        rid=root['root_id'];truth=audit.read(product/'development_truth'/root['partition']/(rid[7:]+'.json'))['truth']
        byid={x['observation_id']:x for x in truth}
        expected={None:1,'rz_drift':4,'rx_overrotation':4,'ry_overrotation':4}
        if dict(Counter(t['mechanism'] for t in truth))!=expected:errors.append(rid+':mechanism count mismatch')
        for bin_id in range(4):
            triplet=[t for t in truth if t['strength_bin']==bin_id]
            if len(triplet)!=3 or len({t['strength'] for t in triplet})!=1:errors.append(rid+':strength triplet mismatch')
            if len({next(o['shots'] for o in root['observations'] if o['observation_id']==t['observation_id']) for t in triplet})!=1:errors.append(rid+':shot/label association')
        with np.load(product/root['raw_path'],allow_pickle=False) as z:
            cc=z['calibration_counts']
            for b,shots in enumerate([4096,8192,16384]):
                vals=cc[b].astype(float).ravel()/shots
                q=np.quantile(vals,[0,.1,.5,.9,.99,1])
                count_rows.append({'root_id':rid,'partition':root['partition'],'stratum':root['sampling_stratum'],
                    'n_qubits':root['n_qubits'],'budget':shots,**dict(zip(['min','p10','median','p90','p99','max'],map(float,q)))})
            if rid in sample:
                qc=audit.qc_from_rec(root['circuit']);clean_p=audit.born(qc)
                for obs in range(13):
                    oid=str(z['observation_ids'][obs]);t=byid[oid];sh=int(z['diagnostic_shots'][obs]);seen=qc
                    if t['mechanism'] is not None:
                        from qiskit import QuantumCircuit
                        seen=QuantumCircuit(qc.num_qubits);b=t['boundary']
                        for item in qc.data[:b]:seen.append(item.operation,[qc.find_bit(w).index for w in item.qubits])
                        gate={'rz_drift':'rz','rx_overrotation':'rx','ry_overrotation':'ry'}[t['mechanism']]
                        getattr(seen,gate)(t['strength'],t['affected_qubit'])
                        for item in qc.data[b:]:seen.append(item.operation,[qc.find_bit(w).index for w in item.qubits])
                    expected_ref=np.random.Generator(np.random.PCG64(audit.audit_seed('diagnostic-reference',oid))).multinomial(sh,clean_p)
                    expected_obs=np.random.Generator(np.random.PCG64(audit.audit_seed('diagnostic-observed',oid))).multinomial(sh,audit.born(seen))
                    if not np.array_equal(expected_ref,z['diagnostic_counts'][obs,0]) or not np.array_equal(expected_obs,z['diagnostic_counts'][obs,1]):errors.append(oid+':diagnostic replay mismatch')
                    replays+=1
        truth_checked+=1
    if any(overlaps.values()):errors.append('Historical structural overlap found; no rows were removed')
    f=pd.DataFrame([{k:v for k,v in x.items() if k not in ('geometry','admission')} for x in families])
    tables={}
    for col in ['n_qubits','topology_class','reference_layers','injection_context_class']:
        tables[col]=pd.crosstab(f.partition,f[col])
    joint=pd.crosstab([f.partition,f.n_qubits],f.topology_class)
    # Explicitly include structurally empty qubit rows, to expose missing cells.
    joint=joint.reindex(pd.MultiIndex.from_product([['FIT','SELECTION','TEST'],[2,3,4,5]],names=['partition','n_qubits']),fill_value=0)
    joint.to_csv(output/'partition_qubit_topology.csv')
    pd.DataFrame(count_rows).to_csv(output/'count_frequency_quantiles_development.csv',index=False)
    result={'cohort_identity':manifest['cohort_identity'],'statistical_evidence_identity':audit.sha(audit.canonical(sorted(rootarray_ids,key=lambda x:x['root_id']))),
        'additional_identity_inventory_sha256':audit.sha(Path(extra).read_bytes()),'additional_source_count':len(old['sources']),
        'overlaps':overlaps,'development_roots_truth_schedule_checked':truth_checked,'independent_development_diagnostic_replays':replays,
        'test_labels_accessed':False,'test_array_files_opened':0,'errors':errors,
        'status':'PASS_WITH_REMAINING_EXCLUSION_GAPS' if not errors else 'FAIL',
        'remaining_historical_gaps':old.get('unverified',[]),'missing_historical_paths':old.get('missing',[]),
        'global_verified_status_permitted':False}
    (output/'post_materialization_audit.json').write_text(json.dumps(result,indent=2)+chr(10))
    lines=['# Additional non-selecting post-materialization audit',
      'This expands identity coverage and audit checks only. No data, admissions, split, measurement, labels or remedy settings changed.',
      '## Structural differences across partitions (TEST label-blind)']
    for col,t in tables.items():lines.extend([col,t.to_markdown()])
    lines.extend(['Joint qubit/topology coverage by partition:',joint.to_markdown(),
      'Zero cells are coverage gaps, not evidence that a circuit is impossible. They are retained rather than filled after observing the sample.',
      '## Diagnostic and content audit',json.dumps(result,indent=2),
      'Statistical evidence identity binds sorted root IDs, parameterized graph hashes and raw evidence bundles. It excludes wall-clock timing and randomized encryption envelopes; the materialization identity separately binds the full original archive inventory.',
      'Legacy products exposing only graph hashes do not certify parameter-free skeleton or wire-relabel exclusion. The inaccessible reserve and any unenumerated products remain unverified. No global development approval is inferred.',
      'Scope of structural equivalence: exact ordered gate/wire skeletons up to wire relabeling. Arbitrary commutation, algebraic gate simplification, unitary equivalence and measurement-equivalence classes are not certified.',
      'Count-frequency quantiles are development only, stratifiable by qubit count/budget/stratum in the companion CSV. Diagnostic replay checks generation correctness against permitted development labels, not prediction accuracy. No TEST label correctness claim is added.'])
    (output/'SUPPLEMENTAL_AUDIT.md').write_text((chr(10)*2).join(lines)+chr(10))
    print(json.dumps(result),flush=True)
    if errors:raise RuntimeError('Additional audit failed')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--product',type=Path,required=True);p.add_argument('--additional-exclusions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.product,a.additional_exclusions,a.output)
