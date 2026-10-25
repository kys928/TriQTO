#!/usr/bin/env python3
"""Adversarial metadata-only label inference. FIT/SELECTION truth for verification only.

NEVER read TEST labels, root truth, ciphertext or TEST raw measurements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

NAMESPACE = "triqto-independent-stability-cohort-v1-20261025"
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")

def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical([NAMESPACE, *parts])).digest()[:16], "big")

def expected_id(root_id, ordinal):
    return "sha256:" + hashlib.sha256(canonical(["observation", root_id, ordinal])).hexdigest()

def recover_labels(root):
    """Use root's public ID alone to recover labels in public observation order."""
    rid = root["root_id"]
    perm = np.random.Generator(np.random.PCG64(seed("observation-order",rid))).permutation(13)
    schedule = [(None, None)] + [(m, k) for k in range(4) for m in MECHANISMS]
    recovered = []
    for ordinal, index in enumerate(perm):
        obs = root["observations"][ordinal]
        oid = expected_id(rid, ordinal)
        if oid != obs["observation_id"]:
            raise RuntimeError("Public observation-ID reconstruction failed")
        mechanism, strength_bin = schedule[int(index)]
        recovered.append({
            "observation_id": oid,
            "mechanism": mechanism,
            "strength_bin": strength_bin,
            "clean_control": mechanism is None,
        })
    return recovered

def recover_location(root, family):
    """Infer structural location proxy from public circuit and family context."""
    operations = root["circuit"]["operations"]
    first_ry = next(x for x in operations if x[0] == "ry")
    affected_qubit = int(first_ry[1][0])
    entangling = [i for i, (name, _, _) in enumerate(operations) if name in ("cx","cz")]
    if not entangling:
        raise RuntimeError("Invalid frozen circuit without entangling gates")
    context = family["injection_context_class"]
    if context == "pre_entangling":
        boundary = max(1, entangling[0])
    elif context == "inter_entangling":
        boundary = min(len(operations)-1, max(1, entangling[len(entangling)//2]+1))
    elif context == "post_entangling_recombination":
        boundary = min(len(operations)-1, entangling[-1]+1)
    else:
        raise ValueError("Unexpected public context")
    return affected_qubit, boundary

def verify(root_dir: Path):
    roots = json.loads((root_dir / "roots.json").read_text())
    families = {f["family_id"]:f for f in json.loads((root_dir/"families.json").read_text())}
    counts = {"roots_checked":0,"observations_checked":0,"mechanism_matches":0,
              "strength_bin_matches":0,"clean_control_matches":0,
              "affected_qubit_matches":0,"injection_boundary_matches":0,
              "TEST_roots_used":0,"TEST_truth_accessed":False}
    for root in roots:
        if root["partition"] == "TEST":
            continue
        if root["partition"] not in ("FIT","SELECTION"):
            raise RuntimeError("Unexpected scientific partition")
        rid = root["root_id"]
        path = root_dir/"development_truth"/root["partition"]/(rid[7:]+".json")
        stored = json.loads(path.read_text())
        truth = {t["observation_id"]:t for t in stored["truth"]}
        recovered = recover_labels(root)
        affected,boundary = recover_location(root,families[root["family_id"]])
        if len(recovered)!=13 or len(truth)!=13:
            raise RuntimeError("Unexpected observation count")
        for row in recovered:
            expected = truth[row["observation_id"]]
            counts["observations_checked"] += 1
            for name,key in [("mechanism_matches","mechanism"),
                             ("strength_bin_matches","strength_bin"),
                             ("clean_control_matches","clean_control")]:
                counts[name] += int(row[key] == expected[key])
            counts["affected_qubit_matches"] += int(affected == int(expected["affected_qubit"]))
            counts["injection_boundary_matches"] += int(boundary == int(expected["boundary"]))
        counts["roots_checked"] += 1
    if counts["roots_checked"] != 720:
        raise RuntimeError("Expected exactly 720 permitted development roots")
    for k in ("mechanism_matches","strength_bin_matches","clean_control_matches",
              "affected_qubit_matches","injection_boundary_matches"):
        if counts[k]!=counts["observations_checked"]:
            raise RuntimeError(f"Public-metadata recovery mismatched permitted truth: {k}")
    counts["scientific_conclusion"] = "Original mechanism/strength-bin/clean and location truth is predictable from public blind metadata under the original grammar."
    return counts

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=verify(a.corpus)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
