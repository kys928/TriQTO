#!/usr/bin/env python3
"""Export only structural identities of an immutable blind cohort for a custodian.

No observations, outcome counts, TEST decryption, truth schedules or labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def make(corpus):
    families=json.loads((corpus/"families.json").read_text())
    roots=json.loads((corpus/"roots.json").read_text())
    fam={f["family_id"]:f for f in families}
    if len(roots)!=960 or len(families)!=240:raise RuntimeError("Original cohort counts drift")
    items=[]
    for r in roots:
        f=fam[r["family_id"]]
        items.append({
            "family_signature_sha256":f["family_signature_sha256"],
            "context_free_signature_sha256":f["context_free_signature_sha256"],
            "wire_signature_sha256":f["wire_signature_sha256"],
            "operation_signature":f["operation_signature"],
            "graph_sha256":r["graph_sha256"]
        })
    unique={canonical(x) for x in items}
    return {"schema":"triqto.blind_candidate_identities.v1",
            "cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
            "families":240,"roots":960,"rows_exposed":len(unique),
            "only_identity_fields":True,"truth_accessed":False,"test_raw_accessed":False,
            "candidates":[json.loads(v) for v in sorted(unique)]}
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();res=make(a.corpus)
    a.output.write_text(json.dumps(res,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in res.items() if k!="candidates"}))
