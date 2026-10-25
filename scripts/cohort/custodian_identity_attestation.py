#!/usr/bin/env python3
"""Custodian-side, identity-only historical/reserve overlap attestation.

Run INSIDE an independently authorized historical-data custodian environment.
Never give this script protected labels or publish reserve identity manifests.
It does not access S3, project model outputs, sealed truth or test observations.

Inputs are identity-only CSV manifests obtained by the custodian under its own
permissions. Output contains only source commitments, coverage and overlap
counts, signed with a custodian-held Ed25519 key. No private key is generated or
stored by this program.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ("family_signature_sha256","context_free_signature_sha256",
          "graph_sha256","operation_signature","wire_signature_sha256")
REQUIRED = FIELDS  # fail closed on missing family, context-free, graph, ordered and relabelled identities

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(raw):
    return "sha256:"+hashlib.sha256(raw).hexdigest()

def normalize(rec):
    return {field: str(rec.get(field,"")).strip() for field in FIELDS}

def source_records(path):
    p=Path(path)
    if p.suffix.lower() != ".csv":
        raise ValueError("Identity sources must be CSV only")
    with p.open(newline="",encoding="utf-8") as fh:
        reader=csv.DictReader(fh)
        if any(x not in reader.fieldnames for x in REQUIRED):
            raise ValueError("Missing required identity columns: "+str(p.name))
        if any(col.lower() in ("mechanism","truth","label","affected_qubit","strength","injection_boundary") for col in reader.fieldnames):
            raise ValueError("Forbidden label-bearing column")
        rows=[normalize(x) for x in reader]
    if not rows:
        raise ValueError("Empty identity manifest")
    if any(any(not row.get(f) for f in REQUIRED) for row in rows):
        raise ValueError("Incomplete required identity values")
    return rows,sha(p.read_bytes())

def candidate_records(manifest):
    m=json.loads(Path(manifest).read_text())
    if m.get("cohort_identity")!="sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d":
        raise ValueError("Unexpected candidate cohort identity")
    if not m.get("candidates") or not isinstance(m["candidates"],list):
        raise ValueError("Expected identity-only exported candidate rows")
    rows=[normalize(x) for x in m["candidates"]]
    if any(not x.get("graph_sha256") for x in rows):
        raise ValueError("Candidate graph hashes missing")
    return m,rows

def attest(candidate_path,inventory_path,source_dir,secret_key_path=None):
    manifest, candidates = candidate_records(candidate_path)
    spec=json.loads(Path(inventory_path).read_text())
    inventory=spec.get("required_sources")
    if not isinstance(inventory,list) or not inventory:
        raise ValueError("Explicit exhaustive source declaration required")
    if not spec.get("custodian_asserts_source_inventory_exhaustive"):
        raise ValueError("Custodian has not asserted exhaustive registry")
    all_covered=True
    sources=[]
    totals={x:0 for x in FIELDS}
    missing_coverage=[]
    candidate_by_field={f:{x[f] for x in candidates if x[f]} for f in FIELDS}
    seen_ids=set()
    for item in inventory:
        name=item.get("name","")
        if not name or name in seen_ids or "/" in name or ".." in name:
            raise ValueError("Invalid or repeated source name")
        seen_ids.add(name)
        path=Path(source_dir)/(name+".csv")
        if not path.is_file():
            all_covered=False
            missing_coverage.append(name)
            sources.append({"source":name,"status":"MISSING"})
            continue
        rows,digest=source_records(path)
        required_fields=item.get("required_identity_fields",list(REQUIRED))
        if set(required_fields) != set(REQUIRED):
            raise ValueError("Full five-identity-type coverage required for complete attestation")
        if not set(required_fields).issubset(FIELDS):
            raise ValueError("Invalid requested identity fields")
        gaps=[f for f in required_fields if any(not x.get(f) for x in rows)]
        if gaps:
            all_covered=False
            missing_coverage.append(name+":"+','.join(gaps))
        counts={}
        for f in FIELDS:
            values={x[f] for x in rows if x[f]}
            matches=len(values & candidate_by_field[f])
            counts[f]=matches
            totals[f]+=matches
        sources.append({"source":name,"sha256":digest,"rows":len(rows),
                        "missing_fields":gaps,"overlap_count_by_identity_type":counts,
                        "status":"COVERED" if not gaps else "INCOMPLETE"})
    zero=all(v==0 for v in totals.values())
    payload={"schema":"triqto.blind_identity_attestation.v1",
             "candidate_cohort_identity":manifest["cohort_identity"],
             "candidate_manifest_sha256":sha(Path(candidate_path).read_bytes()),
             "source_registry_sha256":sha(Path(inventory_path).read_bytes()),
             "custodian_inventory_exhaustive_claim":True,
             "sources":sources,
             "overlap_total_by_identity_type":totals,
             "missing_coverage":missing_coverage,
             "source_coverage_complete":all_covered,
             "no_overlap":zero,
             "passed":bool(all_covered and zero),
             "model_evaluation":False,"labels_read":False,
             "scope":"exact available identity signatures, not arbitrary unitary equivalence"}
    if secret_key_path:
        from cryptography.hazmat.primitives import serialization
        key=serialization.load_pem_private_key(Path(secret_key_path).read_bytes(),password=None)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        if not isinstance(key,Ed25519PrivateKey):
            raise ValueError("Only externally provided Ed25519 key allowed")
        payload["signature_ed25519_hex"]=key.sign(canonical(payload)).hex()
    return payload

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate-identities",type=Path,required=True)
    p.add_argument("--required-source-registry",type=Path,required=True)
    p.add_argument("--identity-manifest-dir",type=Path,required=True)
    p.add_argument("--custodian-ed25519-private-key",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=attest(a.candidate_identities,a.required_source_registry,a.identity_manifest_dir,
                  a.custodian_ed25519_private_key)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"passed":result["passed"],
                      "source_coverage_complete":result["source_coverage_complete"],
                      "no_overlap":result["no_overlap"],
                      "missing_source_count":len(result["missing_coverage"]),
                      "attestation_signed":bool(result.get("signature_ed25519_hex"))}))
    if not result["passed"]:
        raise SystemExit(2)
if __name__=="__main__":main()
