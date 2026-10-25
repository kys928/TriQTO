#!/usr/bin/env python3
"""Generate 96 fresh SELECTION family structural identities only, never measurements.

Must be structurally disjoint from original 240-family corpus and specifically
reserved 60-family genuinely blind replacement TEST. No mechanism truth or
private TEST schedule is read anywhere in this code.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import core as c
import build as b

N_FAMILIES=96
REPRESENTATIVE=76
TEST_IDENTITY="sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1"
KEYS=("family_signature_sha256","context_free_signature_sha256",
      "graph_sha256","operation_signature","wire_signature_sha256")

def make(existing,early,extra,reserved):
    rf=json.loads((existing/"families.json").read_text())
    rr=json.loads((existing/"roots.json").read_text())
    histories=[json.loads(early.read_text()),json.loads(extra.read_text())]
    test=json.loads(reserved.read_text())
    if (test.get("cohort_identity")!=TEST_IDENTITY or test.get("families")!=60 or
         len(test.get("candidates",[]))!=240):
        raise RuntimeError("Reserved original blind TEST identity not correctly bound")
    exclusions={k:set() for k in KEYS}
    for entry in rf:
        for k in KEYS:
            if entry.get(k):exclusions[k].add(entry[k])
    exclusions["graph_sha256"].update(x["graph_sha256"] for x in rr)
    for h in histories:
        for key in KEYS:
            exclusions[key].update(h.get("identities",{}).get(key,()))
    exclusions["wire_signature_sha256"].update(
        c.wire_key_from_signature(x) for x in exclusions["operation_signature"])
    for entry in test["candidates"]:
        for key in KEYS:
            if not entry.get(key):raise RuntimeError("Reserved TEST missing an identity key: "+key)
            exclusions[key].add(entry[key])
    cfg=json.loads((c.ROOT/"configs/v0_2/step14_cross_motif_generalization_training.json").read_text())
    cfg["cross_motif_dataset"]["base_seed"]=c.seed("fresh-i2-selection-identities-20261025")%2**63
    output=[];families=[]
    for index in range(N_FAMILIES):
        stratum="representative" if index<REPRESENTATIVE else "stress"
        for attempt in range(20000):
            bp=c.family_blueprint(index,attempt,cfg)
            circuits=[c.build_variant(bp,index,j,cfg) for j in range(4)]
            events=[(name,list(q)) for name,q in bp["events"]]
            family_signature=c.identity({"events":events,"context":bp["injection_context_class"]})
            context_free=c.identity({"events":events})
            op=c.operation_key(circuits[0])
            wire=c.wire_key_from_signature(op)
            graphs=[c.BASE.graph_hash(c.BASE.serialize_graph(q)) for q in circuits]
            family_fields={"family_signature_sha256":family_signature,
                "context_free_signature_sha256":context_free,
                "operation_signature":op,"wire_signature_sha256":wire}
            if any(val in exclusions[k] for k,val in family_fields.items()):continue
            if len(set(graphs))!=4 or any(g in exclusions["graph_sha256"] for g in graphs):continue
            if any((bp["affected_qubit"],bp["boundary"]) not in c.plausible_candidates(q) for q in circuits):
                continue
            if c.identifiability(bp,circuits,cfg)["status"]!="PASS":continue
            if stratum=="stress":
                if not any(
                    b.geometric_screen(q,bp)["min_norm"]<.30 or
                    b.geometric_screen(q,bp)["smin"]<.15 or
                    b.geometric_screen(q,bp)["max_cos"]>.98
                    for q in circuits):continue
            family_id=c.identity(["fresh-I2-selection-family",index,family_signature])
            families.append({"public_family_id":family_id,"sampling_stratum":stratum,
                             "n_qubits":bp["n_qubits"]})
            for g in graphs:output.append({**family_fields,"graph_sha256":g})
            for k,val in family_fields.items():exclusions[k].add(val)
            exclusions["graph_sha256"].update(graphs)
            break
        else:
            raise RuntimeError("No eligible disjoint selection family in 20k attempts")
        if (index+1)%16==0:
            print(json.dumps({"identity_only_selection_families_proposed":index+1}),flush=True)
    if len(families)!=96 or len(output)!=384:raise RuntimeError("Incomplete proposed cohort")
    return {"schema":"triqto.fresh_selection_structural_identities.v1",
      "identity_commitment":c.identity(["fresh-I2-selection-candidates-v1",output]),
      "source_repository_family_count":240,"reserved_test_identity_commitment":TEST_IDENTITY,
      "family_count":96,"root_count":384,"representative_families":REPRESENTATIVE,
      "stress_families":N_FAMILIES-REPRESENTATIVE,
      "source_registry_exhaustive":False,
      "source_independence_claim":"LOCAL_STRUCTURAL_EXCLUSION_ONLY",
      "observations_materialized":0,"mechanism_labels_acquired":False,
      "test_metadata_or_truth_opened":False,"candidate_identities":output,
      "family_strata":[v["sampling_stratum"] for v in families]}

def audit_against_reserved(candidate,reserved):
    new=json.loads(candidate.read_text())
    test=json.loads(reserved.read_text())
    a=new["candidate_identities"];b=test["candidates"]
    collisions={k:len({v[k] for v in a}&{v[k] for v in b}) for k in KEYS}
    if any(collisions.values()):raise RuntimeError("New SELECTION overlaps reserved TEST")
    if len({x["graph_sha256"] for x in a})!=384:raise RuntimeError("Duplicate candidate circuit roots")
    if len({x["family_signature_sha256"] for x in a})!=96:raise RuntimeError("Incorrect families")
    return collisions

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for k in ("existing","early","extra","reserved-test","output"):a.add_argument("--"+k,required=True,type=Path)
    v=vars(a.parse_args())
    planned=make(v["existing"],v["early"],v["extra"],v["reserved_test"])
    c.save_json(v["output"],planned)
    collisions=audit_against_reserved(v["output"],v["reserved_test"])
    print(json.dumps({"selection_identity_commitment":planned["identity_commitment"],
       "new_families":96,"new_roots":384,
       "reserved_60_family_test_overlap":collisions,
       "observations_created":0,"source_exhaustive_certified":False}))
