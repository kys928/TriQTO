#!/usr/bin/env python3
"""Prepare the *cryptographically blind* new TEST. No historical truth access.

Reconstruct exactly 60 proposed Step-14 families by matching their externally
published identity commitment. Never publish the blueprint's affected location,
mechanism, strength, secret ordering, random seeds or private RSA key.

Real acquisition is HARD-GATED on a complete method-freeze JSON and an external
RSA public key whose private counterpart stays outside the research runner.
Pilot mode can create ONE sacrificial fixture with an ephemeral test key, never
released as a scientific TEST. Do not upload pilot private key.
"""
from __future__ import annotations
import argparse,hashlib,json,os,secrets
from pathlib import Path
import numpy as np
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import padding,rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import core as c
import build as b
from blinded_test_contract import blinded_observation_metadata,private_order

EXPECTED="sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1"
ALLOWED_SHOTS=(512,1024,2048,4096)
BUDGETS=(4096,8192,16384)
MODELS=("rz_drift","rx_overrotation","ry_overrotation")

def reconstruct(public_manifest,limit=60):
    doc=json.loads(Path(public_manifest).read_text())
    if doc.get("cohort_identity")!=EXPECTED or doc.get("families")!=60 or doc.get("roots")!=240:
        raise RuntimeError("TEST identity commitment drift")
    rows=doc["candidates"]
    if len(rows)!=240:raise RuntimeError("Wrong TEST root commitment count")
    cfg=json.loads((c.ROOT/"configs/v0_2/step14_cross_motif_generalization_training.json").read_text())
    cfg["cross_motif_dataset"]["base_seed"]=c.seed("replacement-test-identity-proposal-20261025")%2**63
    for index in range(limit):
        expected=rows[index*4:(index+1)*4]
        for offset in range(20000):
            bp=c.family_blueprint(index,offset,cfg)
            variants=[c.build_variant(bp,index,v,cfg) for v in range(4)]
            hashes=[c.BASE.graph_hash(c.BASE.serialize_graph(q)) for q in variants]
            if hashes !=[row["graph_sha256"] for row in expected]:continue
            events=[(name,list(q)) for name,q in bp["events"]]
            family=c.identity({"events":events,"context":bp["injection_context_class"]})
            flat=c.identity({"events":events})
            op=c.operation_key(variants[0])
            wire=c.wire_key_from_signature(op)
            for entry,graph in zip(expected,hashes):
                if entry!={"family_signature_sha256":family,
                           "context_free_signature_sha256":flat,
                           "graph_sha256":graph,
                           "operation_signature":op,
                           "wire_signature_sha256":wire}:
                    raise RuntimeError("Public candidate structure mismatch")
            yield index,bp,variants,cfg,family
            break
        else:
            raise RuntimeError("Unable to reconstruct commitment for TEST family "+str(index))

def validate_freeze(file):
    p=json.loads(Path(file).read_text())
    if p.get("schema")!="triqto.stabilization_final_method_freeze.v1":
        raise ValueError("Missing genuine final method freeze")
    if p.get("status")!="FROZEN_BEFORE_ANY_REPLACEMENT_TEST_ACQUISITION":
        raise ValueError("Method not genuinely frozen before TEST")
    if p.get("selected_remedy") not in ("R0","R1","R2"):
        raise ValueError("Method winner not declared")
    for key in ("selected_code_sha256","estimator_coefficients_sha256","comparison_protocol_sha256",
                "candidate_checkpoint_sha256","source_lineage_report_sha256"):
        if not str(p.get(key,"")).startswith("sha256:") or len(str(p[key]))!=71:
            raise ValueError("Missing bound frozen hash: "+key)
    if not p.get("accuracy_noninferiority_gate_applied"):
        raise ValueError("Accuracy noninferiority not resolved")
    if not p.get("selection_completed_before_freeze"):
        raise ValueError("SELECTION outcomes missing")
    if not p.get("confirmation_claim_explicitly_limited_to_checked_training_lineage"):
        raise ValueError("Incomplete provenance limitation not acknowledged")
    return p

def counts_from_prob(p,n,rng):
    result=rng.multinomial(n,np.asarray(p,dtype=float))
    if result.sum()!=n or result.min()<0:raise RuntimeError("Invalid shot realization")
    return result

def materialize_one(index,bp,circuit,variant,cfg,cohort_key,secret_schedule_key,out):
    fid=c.identity(["prospective-test-family-v2",index,
        c.identity({"events":[(n,list(q)) for n,q in bp["events"]],
                    "context":bp["injection_context_class"]})])
    rid=c.identity(["replacement-test-root-v2",fid,variant,c.circuit_record(circuit)])
    rng=np.random.default_rng(secrets.randbits(128))
    expected=[(None,0.,int(rng.choice(ALLOWED_SHOTS)),None)]
    levels=np.array(ALLOWED_SHOTS,dtype=np.int64)
    for strength_bin,(lo,hi) in enumerate(cfg["cross_motif_dataset"]["strength_design"]["bins"]):
        strength=float(rng.uniform(float(lo),float(hi)))
        for mech in MODELS:
            expected.append((mech,strength,0,strength_bin))
    allocation=tuple(int(x) for x in rng.permutation(levels))
    expected[1:]=[(m,s,allocation[k],bin_no) for m,s,_,bin_no in expected[1:]
                  for k in []] if False else expected[1:]
    # Exactly the same single shot budget for each triplet at a fixed strength.
    for strength_bin in range(4):
        for j in range(3):
            m,st,_,idx=expected[1+3*strength_bin+j]
            expected[1+3*strength_bin+j]=(m,st,allocation[strength_bin],idx)
    shot_schedule=[int(row[2]) for row in expected]
    public,private=blinded_observation_metadata(secret_schedule_key,rid,shot_schedule)
    permutation=private_order(secret_schedule_key,rid)
    clean=c.probabilities(circuit)
    candidates=c.plausible_candidates(circuit)
    candidate_probs=c.candidate_probabilities(circuit,candidates)
    # independently random finite-shot calibration counts, no public-ID seeding
    cal=np.asarray([[rng.multinomial(sh,candidate_probs)
                     for _ in range(c.DRAWS)] for sh in c.BUDGETS],dtype=np.uint16)
    observed=[]
    truth=[]
    for pub,secret_slot in zip(public,permutation):
        mech,strength,shot,bin_no=expected[secret_slot]
        other=(circuit if mech is None else
               c.BASE.inject_hidden_rotation(circuit,bp["boundary"],bp["affected_qubit"],mech,strength))
        reference_probs=clean
        actual_probs=c.probabilities(other)
        ref=np.stack([counts_from_prob(p,shot,rng) for p in reference_probs])
        seen=np.stack([counts_from_prob(p,shot,rng) for p in actual_probs])
        observed.append(np.stack([ref,seen]))
        truth.append({"observation_id":pub["observation_id"],"mechanism":mech,
          "strength":strength,"strength_bin":bin_no,"shots":shot,
          "affected_qubit":bp["affected_qubit"],"boundary":bp["boundary"],
          "clean_control":mech is None})
    if any(str(x["observation_id"]) not in {p["observation_id"] for p in public} for x in truth):
        raise RuntimeError("Truth/metadata mismatch")
    file_name=rid.split(":",1)[1]
    obs_path=out/"blind"/(file_name+".npz")
    obs_path.parent.mkdir(parents=True,exist_ok=True)
    table=c.outcome_table(circuit.num_qubits)
    np.savez_compressed(obs_path,calibration_counts=cal,diagnostic_counts=np.asarray(observed,dtype=np.uint16),
       outcome_features=table,candidates=np.asarray(candidates,dtype=np.int16),
       observation_ids=np.array([x["observation_id"] for x in public]),
       diagnostic_shots=np.asarray(shot_schedule,dtype=np.int32),
       budgets=np.asarray(c.BUDGETS,dtype=np.int32))
    public_rec={"root_id":rid,"family_id":fid,"variant":variant,
       "n_qubits":circuit.num_qubits,"circuit":c.circuit_record(circuit),
       "graph_sha256":c.BASE.graph_hash(c.BASE.serialize_graph(circuit)),
       "raw_path":str(obs_path.relative_to(out)),"raw_sha256":c.digest(obs_path.read_bytes()),
       "observations":public}
    plain=c.canonical({"root_id":rid,"truth":truth,
            "secret_permutation":list(permutation),
            "blueprint":bp,"generation_namespace":"replacement-test-v2"})
    nonce=os.urandom(12)
    cipher=AESGCM(cohort_key).encrypt(nonce,plain,rid.encode())
    sealed=out/"sealed"/(file_name+".aesgcm")
    sealed.parent.mkdir(parents=True,exist_ok=True)
    sealed.write_bytes(nonce+cipher)
    return public_rec

def build(candidate_file,out,public_key_file,freeze=None,pilot=False):
    if out.exists() and any(out.iterdir()):raise RuntimeError("Refuse overwrite existing test")
    if not pilot:validate_freeze(freeze)
    key=serialization.load_pem_public_key(Path(public_key_file).read_bytes())
    if not isinstance(key,rsa.RSAPublicKey) or key.key_size<3072:
        raise ValueError("Require custodian RSA public key >=3072 bits")
    out.mkdir(parents=True,exist_ok=True)
    data_key=AESGCM.generate_key(bit_length=256)
    secret_schedule=os.urandom(32)
    wrapped=key.encrypt(data_key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                             algorithm=hashes.SHA256(),label=b"TriQTO-replacement-TEST-v2"))
    (out/"wrapped_test_key.bin").write_bytes(wrapped)
    public_roots=[]
    for index,bp,variants,cfg,fid in reconstruct(candidate_file,1 if pilot else 60):
        for j,circuit in enumerate(variants):
            public_roots.append(materialize_one(index,bp,circuit,j,cfg,data_key,secret_schedule,out))
        print(json.dumps({"completed_families":index+1,"truth_revealed":False}),flush=True)
    c.save_json(out/"public_roots.json",public_roots)
    manifest={"schema":"triqto.blind_replacement_test_v2",
      "status":"SYNTHETIC_SACRIFICIAL_PILOT_NOT_A_TEST" if pilot else "BLIND_ACQUIRED_SEALED_NOT_YET_SCORED",
      "source_identity_commitment":EXPECTED,
      "root_count":len(public_roots),"family_count":len(public_roots)//4,
      "observation_count":13*len(public_roots),
      "truth_public":False,"private_schedule_public":False,
      "measurement_seed_public":False,"seal_algorithm":"AES-256-GCM, RSA-OAEP-SHA256 key wrapping",
      "recipient_public_key_sha256":c.digest(Path(public_key_file).read_bytes()),
      "source_completeness_claim":"KNOWN_TRAINING_LINEAGE_ONLY_NOT_ALL_PROTECTED_RESERVES",
      "model_predictions_used":False,"sealed_files_count":len(public_roots)}
    c.save_json(out/"public_manifest.json",manifest)
    return manifest

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidates",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--recipient-public-key",required=True,type=Path)
    p.add_argument("--final-method-freeze",type=Path)
    p.add_argument("--pilot",action="store_true")
    a=p.parse_args()
    result=build(a.candidates,a.output,a.recipient_public_key,a.final_method_freeze,a.pilot)
    print(json.dumps({k:result[k] for k in ("status","family_count","root_count","observation_count","truth_public")}))
