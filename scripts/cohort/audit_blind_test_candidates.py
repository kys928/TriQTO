"""Validate fresh structural-only TEST candidate manifest and known-source exclusion."""
import argparse,json
from collections import Counter
from pathlib import Path
import core as c

KEYS=("family_signature_sha256","context_free_signature_sha256","graph_sha256",
      "operation_signature","wire_signature_sha256")

def validate(plan_path,existing,early,extra):
    p=json.loads(plan_path.read_text())
    rows=p["candidates"]
    if p["families"]!=60 or p["roots"]!=240 or len(rows)!=240:
        raise RuntimeError("Candidate quantity failure")
    if p["observations_materialized"]!=0 or p["source_exclusion_complete"] is not False:
        raise RuntimeError("Pre-attestation plan must not contain measurements or approval")
    if p["cohort_identity"]!=c.identity(["replacement-test-candidates",rows]):
        raise RuntimeError("Candidate commitment mismatch")
    if len({x["graph_sha256"] for x in rows})!=240:
        raise RuntimeError("Repeated graph across proposed roots")
    groups=Counter(x["family_signature_sha256"] for x in rows)
    if len(groups)!=60 or set(groups.values())!={4}:
        raise RuntimeError("Family structural grouping incorrect")
    f=json.loads((existing/"families.json").read_text())
    r=json.loads((existing/"roots.json").read_text())
    historical=[json.loads(early.read_text()),json.loads(extra.read_text())]
    known={k:set() for k in KEYS}
    for item in f:
        for k in KEYS:
            if item.get(k):known[k].add(item[k])
    known["graph_sha256"].update(x["graph_sha256"] for x in r)
    for d in historical:
        for k in KEYS:known[k].update(d["identities"].get(k,[]))
    known["wire_signature_sha256"].update(c.wire_key_from_signature(x) for x in known["operation_signature"])
    collisions={k:len({x[k] for x in rows}&known[k]) for k in KEYS}
    if any(collisions.values()):
        raise RuntimeError("Collision with known historical sources: "+str(collisions))
    return {"status":"LOCAL_IDENTITIES_PASS_EXTERNAL_INDEPENDENCE_UNVERIFIED",
            "candidate_identity":p["cohort_identity"],"families":60,"roots":240,
            "known_structural_identity_collisions":collisions,
            "test_observations_created":0,"external_custodian_signed":False,
            "coverage_limit":"Prior explicitly known identity products only; historical/reserve registry incomplete"}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    for n in ("candidate-manifest","existing","early","extra","out"):a.add_argument("--"+n,required=True,type=Path)
    args=vars(a.parse_args())
    result=validate(args["candidate_manifest"],args["existing"],args["early"],args["extra"])
    args["out"].write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
