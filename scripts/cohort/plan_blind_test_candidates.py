"""Prepare 60 new Step-14 TEST candidate-family structural identities only.

NO observation labels, measurement counts, or TEST predictions are generated.
These candidate identities require their OWN external disjointness attestation.
"""
import argparse,json
from pathlib import Path
import core as c
import build as b

def plan(existing,early,extra):
    cohort_f=json.loads((existing/"families.json").read_text())
    cohort_r=json.loads((existing/"roots.json").read_text())
    histories=[json.loads(early.read_text()),json.loads(extra.read_text())]
    seen={k:set() for k in ("family_signature_sha256","graph_sha256","operation_signature")}
    seen_wire=set()
    for f in cohort_f:
        seen["family_signature_sha256"].add(f["family_signature_sha256"])
        seen["operation_signature"].add(f["operation_signature"])
        seen_wire.add(f["wire_signature_sha256"])
    seen["graph_sha256"].update(r["graph_sha256"] for r in cohort_r)
    for h in histories:
        for k in seen:seen[k].update(h["identities"].get(k,[]))
    seen_wire.update(c.wire_key_from_signature(x) for x in seen["operation_signature"])
    cfg=json.loads((c.ROOT/"configs/v0_2/step14_cross_motif_generalization_training.json").read_text())
    cfg["cross_motif_dataset"]["base_seed"]=c.seed("replacement-test-identity-proposal-20261025")%2**63
    output=[]
    metadata=[]
    for index in range(60):
        stratum="representative" if index<48 else "stress"
        for attempt in range(20000):
            bp=c.family_blueprint(index,attempt,cfg)
            quanta=[c.build_variant(bp,index,j,cfg) for j in range(4)]
            ev=[(name,list(q)) for name,q in bp["events"]]
            fam=c.identity({"events":ev,"context":bp["injection_context_class"]})
            op=c.operation_key(quanta[0])
            wire=c.wire_key_from_signature(op)
            graphs=[c.BASE.graph_hash(c.BASE.serialize_graph(q)) for q in quanta]
            if fam in seen["family_signature_sha256"] or op in seen["operation_signature"] or wire in seen_wire:continue
            if len(set(graphs))<4 or any(g in seen["graph_sha256"] for g in graphs):continue
            if any((bp["affected_qubit"],bp["boundary"]) not in c.plausible_candidates(q) for q in quanta):continue
            if c.identifiability(bp,quanta,cfg)["status"]!="PASS":continue
            if stratum=="stress" and not any(b.geometric_screen(q,bp)["min_norm"]<.30 or b.geometric_screen(q,bp)["smin"]<.15 or b.geometric_screen(q,bp)["max_cos"]>.98 for q in quanta):continue
            context_free=c.identity({"events":ev})
            fid=c.identity(["prospective-test-family-v2",index,fam])
            for graph in graphs:
                output.append({"family_signature_sha256":fam,"context_free_signature_sha256":context_free,
                               "graph_sha256":graph,"operation_signature":op,"wire_signature_sha256":wire})
            metadata.append({"proposed_family_id":fid,"sampling_stratum":stratum})
            seen["family_signature_sha256"].add(fam)
            seen["operation_signature"].add(op)
            seen_wire.add(wire)
            seen["graph_sha256"].update(graphs)
            break
        else:raise RuntimeError("No admissible independently sampled family")
    if len(output)!=240 or len(metadata)!=60:raise RuntimeError("Incorrect candidate count")
    return {"schema":"triqto.blind_candidate_identities.v1","cohort_identity":c.identity(["replacement-test-candidates",output]),
            "purpose":"PROSPECTIVE_UNACQUIRED_TEST_CANDIDATES",
            "families":60,"roots":240,"observations_materialized":0,
            "source_exclusion_complete":False,"candidates":output}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--existing",required=True,type=Path)
    p.add_argument("--early",required=True,type=Path)
    p.add_argument("--extra",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    result=plan(a.existing,a.early,a.extra)
    c.save_json(a.out,result)
    print(json.dumps({"candidate_identity":result["cohort_identity"],"families":60,
                      "roots":240,"measurements_created":0,"external_identity_attested":False}))
