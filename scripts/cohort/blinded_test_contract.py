#!/usr/bin/env python3
"""Secret-scheduled mechanism-blind TEST metadata primitives. DESIGN/PILOT ONLY.

The signing/scoring custodian, not a PR or ordinary CI runner, must own the
secret key and all truth schedules. This module does not generate circuits,
measurements, protected labels, or an actual TEST population.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import secrets
import numpy as np

def private_order(master_key: bytes, opaque_root_id: str):
    if not isinstance(master_key,bytes) or len(master_key)<32:
        raise ValueError("Custodian secret key must have >=256 bits")
    if not opaque_root_id:
        raise ValueError("Missing root ID")
    digest=hmac.new(master_key,("TriQTO-TEST-v2|"+opaque_root_id).encode(),hashlib.sha256).digest()
    generator=np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16],"big")))
    return tuple(int(x) for x in generator.permutation(13))

def blinded_observation_metadata(master_key:bytes,root_id:str,shot_schedule:list[int]):
    """Return public blind metadata + private ordinal mapping held by custodian.

    All identifiers are randomly generated and have no deterministic link to
    mechanism/strength/original root slot. Private output MUST NEVER enter
    public release, ordinary CI logs or an untrusted research workflow.
    """
    if len(shot_schedule)!=13:
        raise ValueError("Need 13 observations per root under original task")
    perm=private_order(master_key,root_id)
    public=[]
    private=[]
    for public_position,internal_index in enumerate(perm):
        identifier=secrets.token_hex(16)
        public.append({"observation_id":identifier,"shots":int(shot_schedule[internal_index])})
        private.append({"observation_id":identifier,"private_schedule_index":internal_index})
    if len({x["observation_id"] for x in public}) !=13:
        raise RuntimeError("CSPRNG observation IDs must be unique")
    return public,private

def validate_public_metadata(rows):
    if len(rows)!=13:
        raise ValueError("Expected 13 public rows")
    allowed={"observation_id","shots"}
    for row in rows:
        if set(row)!=allowed:
            raise ValueError("Unexpected field in blind metadata")
        if not isinstance(row["observation_id"],str) or len(row["observation_id"])!=32:
            raise ValueError("Opaque identifier must have 128-bit random encoding")
        if int(row["shots"]) not in (512,1024,2048,4096):
            raise ValueError("Unexpected diagnostic shot budget")
    return True
