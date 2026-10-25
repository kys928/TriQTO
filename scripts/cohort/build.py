"""Construct the preregistered cohort; never run the frozen candidate network."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import core as c


def geometric_screen(qc,bp):
    p=c.candidate_probabilities(qc,[(bp['affected_qubit'],bp['boundary'])])
    table=c.outcome_table(qc.num_qubits)
    means=np.einsum('...o,ok->...k',p,table)
    j=np.moveaxis((means[:,:,0]-means[:,:,1])/(2*c.EPSILON),1,-1).reshape(-1,3)
    norms=np.linalg.norm(j,axis=0);u=j/np.maximum(norms,1e-12)
    singular=np.linalg.svd(u,compute_uv=False)
    cos=np.abs(u.T@u);np.fill_diagonal(cos,0)
    return {'min_norm':float(norms.min()),'smin':float(singular[-1]),'max_cos':float(cos.max()),
            'structural_zero_axes':int(np.sum(norms<1e-10))}


def accepted_families(cfg,exclusions,pilot,secret_salt):
    old=exclusions['identities'];old_ops=set(old['operation_signature'])
    old_fam=set(old['family_signature_sha256']);old_graph=set(old['graph_sha256'])
    old_wire={c.wire_key_from_signature(x) for x in old_ops}
    step12=json.loads((c.ROOT/'configs/v0_2/step12_independent_phase_generalization.json').read_text())
    for motif in step12['generalization_design']['motifs']:
        old_ops.add(json.dumps(motif['reference_operation_signature'],separators=(',',':')))
    seen_family=set();seen_graph=set();seen_ops=set();seen_wire=set()
    families=[];roots=[];recipes={};logs=[]
    slots=[('PILOT','representative')]*4 if pilot else (
       [('FIT','representative')]*96+[('FIT','stress')]*24+
       [('SELECTION','representative')]*48+[('SELECTION','stress')]*12+
       [('TEST','representative')]*48+[('TEST','stress')]*12)
    for family_index,(split,stratum) in enumerate(slots):
        local_cfg=json.loads(json.dumps(cfg))
        salt=secret_salt if split=='TEST' else ('engineering' if pilot else 'development')
        local_cfg['cross_motif_dataset']['base_seed']=c.seed('grammar',salt,split)%2**63
        for offset in range(20000):
            bp=c.family_blueprint(family_index,offset,local_cfg)
            variants=[c.build_variant(bp,family_index,v,local_cfg) for v in range(4)]
            events=[(name,list(qs)) for name,qs in bp['events']]
            famsig=c.identity({'events':events,'context':bp['injection_context_class']})
            plain=c.identity({'events':events})
            op=c.operation_key(variants[0])
            wire=c.wire_key_from_signature(op)
            graphs=[c.BASE.graph_hash(c.BASE.serialize_graph(qc)) for qc in variants]
            reason=None
            if famsig in old_fam or famsig in seen_family or plain in seen_family:reason='family_signature_overlap'
            elif op in old_ops or op in seen_ops:reason='operation_signature_overlap'
            elif wire in old_wire or wire in seen_wire:reason='wire_relabel_signature_overlap'
            elif any(g in old_graph or g in seen_graph for g in graphs):reason='parameterized_graph_overlap'
            elif any((bp['affected_qubit'],bp['boundary']) not in c.plausible_candidates(qc) for qc in variants):reason='illegal_true_location'
            admission=None;geometry=None
            if reason is None:
                admission=c.identifiability(bp,variants,local_cfg)
                if admission['status']!='PASS': reason='nonidentifiable_at_original_admission'
            if reason is None:
                geometry=[geometric_screen(qc,bp) for qc in variants]
                stress=any(x['min_norm']<.30 or x['smin']<.15 or x['max_cos']>.98 for x in geometry)
                if stratum=='stress' and not stress: reason='outside_preregistered_stress_rule'
            # TEST screening details remain in the encrypted generation recipe.
            entry={'family_slot':family_index,'partition':split,'sampling_stratum':stratum,
                   'attempt':offset,'accepted':reason is None,'reason':reason}
            if split!='TEST':
                entry.update({'admission':admission,'geometry':geometry})
                logs.append(entry)
            if reason is not None:continue
            fid=c.identity(['family',c.NAMESPACE,split,stratum,famsig,c.circuit_record(variants[0])])
            public={'family_id':fid,'partition':split,'sampling_stratum':stratum,
                    'n_qubits':bp['n_qubits'],'topology_class':bp['topology_class'],
                    'reference_layers':len([1 for name,_ in bp['events'] if name not in ('cx','cz')])-2,
                    'injection_context_class':bp['injection_context_class'],
                    'family_signature_sha256':famsig,'context_free_signature_sha256':plain,
                    'wire_signature_sha256':wire,'operation_signature':op}
            if split!='TEST': public.update({'admission':admission,'geometry':geometry})
            families.append(public)
            for v,qc in enumerate(variants):
                rid=c.identity(['root',fid,v,c.circuit_record(qc)])
                candidates=c.plausible_candidates(qc)
                roots.append({'root_id':rid,'family_id':fid,'variant':v,'partition':split,
                    'sampling_stratum':stratum,'n_qubits':qc.num_qubits,'gate_count':len(qc.data),
                    'candidate_count':len(candidates),'candidates':[list(x) for x in candidates],
                    'graph_sha256':graphs[v],'circuit':c.circuit_record(qc)})
                recipes[rid]={'blueprint':bp,'generator_family_index':family_index,
                    'variant':v,'config':local_cfg,'test_salt':salt if split=='TEST' else None}
            seen_family.update([famsig,plain]);seen_ops.add(op);seen_wire.add(wire);seen_graph.update(graphs)
            break
        else:raise RuntimeError('Exceeded preregistered attempt cap')
        if (family_index+1)%10==0 or pilot:print(f'Screened {family_index+1}/{len(slots)} families',flush=True)
    return families,roots,recipes,logs


def materialize_one(root,recipe,out,draws,aes):
    start=time.perf_counter();rid=root['root_id'];split=root['partition']
    qc=c.circuit_from_record(root['circuit']);table=c.outcome_table(qc.num_qubits)
    p=c.candidate_probabilities(qc,root['candidates'])
    calibration=c.acquire_calibration(p,rid,draws)
    # Bound model-visible acquisition API does not consume true location here.
    diagnostics,observations,truth=c.acquire_diagnostics(qc,recipe['blueprint'],
        recipe['generator_family_index']*4+recipe['variant'],recipe['config'],rid)
    raw_dir=out/'blind'/split;raw_dir.mkdir(parents=True,exist_ok=True)
    path=raw_dir/(rid[7:]+'.npz')
    np.savez_compressed(path,calibration_counts=calibration,diagnostic_counts=diagnostics,
        outcome_features=table,candidates=np.asarray(root['candidates'],dtype=np.int16),
        observation_ids=np.asarray([x['observation_id'] for x in observations]),
        diagnostic_shots=np.asarray([x['shots'] for x in observations],dtype=np.int32),
        budgets=np.asarray(c.BUDGETS,dtype=np.int32))
    jac=c.counts_to_jacobian(calibration,table)
    delta,weights=c.diagnostic_delta(diagnostics,table)
    features=c.canonical_batch(delta,jac,weights)
    if not np.isfinite(features).all():raise RuntimeError('Nonfinite canonical input')
    # Persist rich derived geometry for development, only integrity checks on TEST.
    # All TEST geometry is reproducible later from the raw blind counts.
    derived_path=None
    if split!='TEST':
        target=out/'derived'/split;target.mkdir(parents=True,exist_ok=True)
        derived_path=target/(rid[7:]+'.npz')
        profiles=c.profile_batch(delta,jac,weights)
        sample_j=jac[:,[0,draws-1]][:,:,[0,len(root['candidates'])-1]]
        sample_f=features[:,[0,draws-1]][:,:,[0,len(root['candidates'])-1]][:,:,:,[0,12]]
        np.savez_compressed(derived_path,sample_jacobians=sample_j,sample_features=sample_f,
            jacobian_sha256=np.asarray(c.digest(np.ascontiguousarray(jac).tobytes())),
            feature_sha256=np.asarray(c.digest(np.ascontiguousarray(features).tobytes())),
            profile_sha256=np.asarray(c.digest(np.ascontiguousarray(profiles).tobytes())),delta=delta,weights=weights)
        target=out/'development_truth'/split;target.mkdir(parents=True,exist_ok=True)
        c.save_json(target/(rid[7:]+'.json'),{'root_id':rid,'truth':truth})
        # First candidate only; evaluation of count-estimator calibration, never estimator input.
        target=out/'acquisition_audit'/split;target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/(rid[7:]+'.npz'),first_candidate_probabilities=p[0])
    else:
        target=out/'sealed';target.mkdir(parents=True,exist_ok=True)
        nonce=os.urandom(12)
        payload=c.canonical({'root_id':rid,'truth':truth,'generation_recipe':recipe})
        (target/(rid[7:]+'.aesgcm')).write_bytes(nonce+aes.encrypt(nonce,payload,rid.encode()))
    obs={**root,'observations':observations,'raw_path':str(path.relative_to(out)),
         'raw_sha256':c.digest(path.read_bytes()),'raw_bytes':path.stat().st_size,
         'derived_path':str(derived_path.relative_to(out)) if derived_path else None,
         'elapsed_seconds':time.perf_counter()-start}
    # Remove recipe-only and calibrated simulator quantities before caller sees output.
    return obs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclusions',type=Path,required=True)
    parser.add_argument('--public-key',type=Path)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refuse overwriting an existing corpus')
    out.mkdir(parents=True,exist_ok=True)
    exclusions=json.loads(args.exclusions.read_text())
    if not exclusions.get('sources'):raise RuntimeError('Missing historical exclusion inventory')
    cfg=json.loads((c.ROOT/'configs/v0_2/step14_cross_motif_generalization_training.json').read_text())
    key=AESGCM.generate_key(bit_length=256);aes=AESGCM(key);secret_salt=os.urandom(32).hex()
    if not args.pilot:
        if args.public_key is None:raise RuntimeError('TEST public sealing key required')
        pub=serialization.load_pem_public_key(args.public_key.read_bytes())
        wrapped=pub.encrypt(key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=b'TriQTO-cohort-v1'))
        (out/'wrapped_test_key.bin').write_bytes(wrapped)
    start=time.perf_counter()
    families,roots,recipes,logs=accepted_families(cfg,exclusions,args.pilot,secret_salt)
    screening_seconds=time.perf_counter()-start
    c.save_json(out/'families.json',families);c.save_json(out/'screening_development.json',logs)
    c.save_json(out/'historical-exclusions.json',exclusions)
    draws=2 if args.pilot else c.DRAWS
    candidates=sum(r['candidate_count'] for r in roots)
    calls=candidates*18*3*draws
    simulated_shots=candidates*18*draws*sum(c.BUDGETS)
    feasibility={'families':len(families),'roots':len(roots),'observations':len(roots)*13,
        'candidate_count':candidates,'calibration_draws_per_budget':draws,'calibration_calls':calls,
        'simulated_calibration_shots':simulated_shots,'screening_seconds':screening_seconds,
        'uncompressed_calibration_count_bytes':sum(r['candidate_count']*18*3*draws*2**r['n_qubits']*2 for r in roots),
        'pilot_scaled_raw_bytes_projection':int(1218488*(candidates/484)*(draws/2)),
        'conservative_local_runtime_seconds_projection':4.25*(len(roots)/16)*(draws/2),
        'projection_limitations':'Pilot circuit dimensions and machine throughput differ; projections are not guarantees',
        'qpu_cost':0,'runpod_pods_started':0,'candidate_count_summary':dict(Counter(r['candidate_count'] for r in roots))}
    c.save_json(out/'FEASIBILITY.json',feasibility)
    print(json.dumps(feasibility),flush=True)
    if args.prepare_only:
        raise RuntimeError('Prepare-only is not a complete corpus; recipes held in memory are not released')
    completed=[]
    for i,root in enumerate(roots):
        result=materialize_one(root,recipes[root['root_id']],out,draws,aes)
        completed.append(result)
        if (i+1)%20==0 or args.pilot:print(f'Acquired {i+1}/{len(roots)} roots; bytes={sum(r["raw_bytes"] for r in completed)}',flush=True)
    c.save_json(out/'roots.json',completed)
    inventory=[]
    for path in sorted(out.rglob('*')):
        if path.is_file():inventory.append({'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':c.digest(path.read_bytes())})
    identity=c.identity({'sources':c.SOURCES,'files':inventory,'namespace':c.NAMESPACE})
    manifest={'schema':'triqto.independent_cohort.v1','cohort_identity':identity,
        'status':'ENGINEERING_PILOT_ONLY' if args.pilot else 'CONSTRUCTED_QUARANTINED_PENDING_INDEPENDENCE_AND_AUDIT',
        'families':len(families),'roots':len(roots),'examples':len(roots)*13,'draws_per_budget':draws,
        'source_hashes':c.SOURCES,'code_hashes':{str(p.relative_to(c.ROOT)):c.digest(p.read_bytes()) for p in sorted((c.ROOT/'scripts/cohort').glob('*.py'))},
        'protocol_sha256':c.digest((c.ROOT/'docs/evidence/finite_shot_cohort/COHORT_PROTOCOL.md').read_bytes()),
        'historical_exclusion_coverage_complete':bool(exclusions['complete_historical_coverage']),
        'unverified_exclusions':exclusions['unverified'],'sealed_truth_accessed':False,
        'model_inference_performed':False,'remedies_evaluated':False,'files':inventory,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__},
        'elapsed_seconds':time.perf_counter()-start,
        'rng':{'algorithm':'PCG64','namespace':c.NAMESPACE,'calibration_parts':['calibration','root_id','budget','draw'],
               'call_order':['candidate','axis Z/X/Y','sign plus/minus','basis Z/X/Y'],
               'diagnostic_parts':['diagnostic-reference OR diagnostic-observed','opaque observation_id'],
               'test_generation_seeds':'encrypted with TEST truth; no plaintext persisted'}}
    c.save_json(out/'DATASET_MANIFEST.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','rng','source_hashes','code_hashes')}),flush=True)

if __name__=='__main__':main()
