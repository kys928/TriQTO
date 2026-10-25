#!/usr/bin/env python3
"""Verify a custodian-issued identity-only attestation without protected data."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives import serialization

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def sha(data):
    return "sha256:"+hashlib.sha256(data).hexdigest()

def verify(attestation,public_key,candidate_path,source_registry_path,trust_fingerprint):
    record=json.loads(Path(attestation).read_text())
    if record.get("schema")!="triqto.blind_identity_attestation.v1":
        raise ValueError("Unexpected attestation schema")
    key_bytes=Path(public_key).read_bytes()
    pub=serialization.load_pem_public_key(key_bytes)
    if not isinstance(pub,Ed25519PublicKey):
        raise ValueError("Expected Ed25519 custodian public key")
    # Trust anchor must be predeclared OUTSIDE and BEFORE this research run.
    canonical_key=pub.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    fingerprint=sha(canonical_key)
    if fingerprint!=trust_fingerprint:
        raise ValueError("Custodian signer not in trusted frozen allowlist")
    signature=bytes.fromhex(record.get("signature_ed25519_hex",""))
    unsigned=dict(record)
    unsigned.pop("signature_ed25519_hex",None)
    pub.verify(signature,canonical(unsigned))
    if record["candidate_manifest_sha256"]!=sha(Path(candidate_path).read_bytes()):
        raise ValueError("Candidate identity manifest not the one attested")
    if record["source_registry_sha256"]!=sha(Path(source_registry_path).read_bytes()):
        raise ValueError("Source registry not the one attested")
    if not record.get("passed") or not record.get("source_coverage_complete") or not record.get("no_overlap"):
        raise ValueError("Attestation includes failures or incomplete registry")
    if record.get("missing_coverage") or any(any(v!=0 for v in s.get("overlap_count_by_identity_type",{}).values()) for s in record["sources"]):
        raise ValueError("Inconsistent overlap or coverage record")
    return {"verified":True,"attestation_sha256":sha(Path(attestation).read_bytes()),
            "custodian_pubkey_fingerprint":fingerprint,
            "sources":len(record["sources"]),"candidate":record["candidate_cohort_identity"],
            "NOTE":"Cryptographic verification alone does not prove the source registry really is exhaustive."}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("attestation","public-key","candidate-identities","source-registry","trusted-fingerprint","output"):
        p.add_argument("--"+name,required=True)
    a=vars(p.parse_args())
    result=verify(a["attestation"],a["public_key"],a["candidate_identities"],a["source_registry"],a["trusted_fingerprint"])
    Path(a["output"]).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
