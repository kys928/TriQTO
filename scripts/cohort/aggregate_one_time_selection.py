#!/usr/bin/env python3
"""Fail-closed complete selection shard join and prospective one-time verdict."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from select_stabilization_remedy import run as select

LOCK="docs/evidence/finite_shot_cohort/STABILIZATION_PRE_SELECTION_LOCK_20261025.json"

def sha(p):return "sha256:"+hashlib.sha256(p.read_bytes()).hexdigest()

def gather(directory,lockfile,output):
    lock=json.loads(lockfile.read_text())
    if lock.get("status")!="FROZEN_CANDIDATES_BEFORE_SELECTION":
        raise RuntimeError("Selection freeze absent")
    locksha=sha(lockfile)
    inputs=sorted(directory.rglob("selection-N-*.json"))
    if len(inputs)!=15:raise RuntimeError(f"Expected exactly 15 complete selection shards, got {len(inputs)}")
    data=[json.loads(x.read_text()) for x in inputs]
    data=sorted(data,key=lambda d:d["family_slice"][0])
    if [d["family_slice"] for d in data] != [[n,n+4] for n in range(0,60,4)]:
        raise RuntimeError("Missing / overlapping selection families")
    for d in data:
        if d.get("schema")!="triqto.stabilization_selection_scored_repeats.v1" or d.get("partition")!="SELECTION":
            raise RuntimeError("Wrong partition in shard")
        if d.get("test_read",True) or d.get("original_test_opened",True) or d.get("replacement_test_opened",True):
            raise RuntimeError("Selection shard contaminated by TEST")
        if d.get("method_selection_lock_sha256")!=locksha or len(d.get("rows",[]))!=192:
            raise RuntimeError("Shard not matching prospective freeze / row count")
    full={"schema":"triqto.stabilization_selection_scored_repeats.v1",
          "partition":"SELECTION","test_read":False,
          "rows":[r for d in data for r in d["rows"]]}
    raw=output.parent/"selection-joined-ephemeral.json"
    raw.write_text(json.dumps(full,separators=(",",":"),sort_keys=True)+"\n")
    verdict=select(raw,output)
    raw.unlink()
    verdict["selection_shard_commitments"]={x.name:sha(x) for x in inputs}
    verdict["pre_selection_lock_sha256"]=locksha
    output.write_text(json.dumps(verdict,indent=2,sort_keys=True)+"\n")
    return verdict

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--shards",required=True,type=Path)
    p.add_argument("--lock",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    v=gather(a.shards,a.lock,a.output)
    print(json.dumps({"selected_remedy":v["selected_remedy"],
                      "any_remedy_supported":v["any_remedy_supported"],
                      "families":v["families"],"records":v["records"],
                      "R1":v["contrasts"]["R1"],
                      "R2":v["contrasts"]["R2"]},indent=2))
