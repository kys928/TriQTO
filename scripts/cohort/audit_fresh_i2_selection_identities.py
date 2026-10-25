#!/usr/bin/env python3
"""Independent preacquisition audit of I2 SELECTION identities vs used & reserved.

Never reads measurement records, protected truth, or sealed TEST answers.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import core as c

FIELDS=("family_signature_sha256","context_free_signature_sha256",
        "graph_sha256","operation_signature","wire_signature_sha256")
EXPECTED="sha256:76b79f0a5ff2cea01a3bb97e8c6dc3634099679805d6bc8fdc3c5dca90b044e7"
TEST="sha256:ebecad22eacc41247a3954f73da11427dc10eabb32fd19796c91ed28daa524d1"

def audit(plan,existing,early,extra,test):
    p=json.loads(plan.read_text());t=json.loads(test.read_text())
    if (p.get("schema")!="triqto.fresh_selection_structural_identities.v1" or
        p.get("identity_commitment")!=EXPECTED or p.get("family_count")!=96 or
        p.get("root_count")!=384 or p.get("observations_materialized")!=0 or
        p.get("mechanism_labels_acquired") or p.get("test_metadata_or_truth_opened")):
        raise RuntimeError("New selection identity planning contract violated")
    if t.get("cohort_identity")!=TEST or len(t["candidates"])!=240:
        raise RuntimeError("Blind TEST metadata commitment changed")
    rows=p["candidate_identities"]
    if len(rows)!=384 or any(set(row)!=set(FIELDS) for row in rows):
        raise RuntimeError("Incomplete five-key structural signatures")
    if len({row["graph_sha256"] for row in rows})!=384:
        raise RuntimeError("Duplicate selection graph")
    counts=Counter(r["family_signature_sha256"] for r in rows)
    if len(counts)!=96 or set(counts.values())!={4}:
        raise RuntimeError("Expected 96 family groups of 4 roots")
    if c.identity(["fresh-I2-selection-candidates-v1",rows])!=EXPECTED:
        raise RuntimeError("Candidate SHA-256 commitment mismatch")
    oldf=json.loads((existing/"families.json").read_text())
    oldr=json.loads((existing/"roots.json").read_text())
    registries=[json.loads(early.read_text()),json.loads(extra.read_text())]
    old={k:set() for k in FIELDS}
    for f in oldf:
        for k in FIELDS:
            if f.get(k):old[k].add(f[k])
    old["graph_sha256"].update(x["graph_sha256"] for x in oldr)
    for reg in registries:
        for k in FIELDS:old[k].update(reg["identities"].get(k,[]))
    old["wire_signature_sha256"].update(c.wire_key_from_signature(s) for s in old["operation_signature"])
    reserved={k:{row[k] for row in t["candidates"]} for k in FIELDS}
    known_overlaps={k:len({x[k] for x in rows}&old[k]) for k in FIELDS}
    test_overlaps={k:len({x[k] for x in rows}&reserved[k]) for k in FIELDS}
    if any(known_overlaps.values()) or any(test_overlaps.values()):
        raise RuntimeError("Identity collision against known spent or reserved sources")
    return {"schema":"triqto.fresh_selection_structural_audit.v1",
            "status":"LOCAL_FIVE_IDENTITY_CLASSES_PASS_SOURCE_COVERAGE_LIMITED",
            "candidate_commitment":EXPECTED,"old_reserved_test_commitment":TEST,
            "families":96,"roots":384,"original_prior_known_source_collisions":known_overlaps,
            "reserved_blind_test_collisions":test_overlaps,
            "test_observations_generated":0,"selection_measurements_generated":0,
            "sealed_truth_accessed":False,"independent_historical_reserve_coverage_complete":False,
            "scientific_qualification":"Only provided historical and public metadata; other protected historical sources unverified"}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for n in ("plan","existing","early","extra","test","out"):a.add_argument("--"+n,required=True,type=Path)
    p=vars(a.parse_args())
    result=audit(p["plan"],p["existing"],p["early"],p["extra"],p["test"])
    p["out"].write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
