"""Synthetic tests for identity-only attestation and candidate estimators."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import custodian_identity_attestation as ci
import compare_nonlinear_estimators_fit as f

def synthetic(tmp_path,overlap=False,missing=False):
    candidates={"cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
      "candidates":[{"graph_sha256":"sha256:a","operation_signature":"ry:q0",
          "family_signature_sha256":"sha256:fa","context_free_signature_sha256":"sha256:ca",
          "wire_signature_sha256":"sha256:wa"}]}
    (tmp_path/"candidate.json").write_text(json.dumps(candidates))
    (tmp_path/"registry.json").write_text(json.dumps({
        "custodian_asserts_source_inventory_exhaustive":True,
        "required_sources":[{"name":"protected_identity_only",
                             "required_identity_fields":list(ci.FIELDS)}]}))
    sources=tmp_path/"sources"
    sources.mkdir()
    if not missing:
        header=",".join(ci.FIELDS)
        values=(
            ["sha256:fa","sha256:ca","sha256:a","ry:q0","sha256:wa"]
            if overlap else
            ["sha256:fb","sha256:cb","sha256:b","rx:q1","sha256:wb"]
        )
        (sources/"protected_identity_only.csv").write_text(header+"\n"+",".join(values)+"\n")
    return (tmp_path/"candidate.json",tmp_path/"registry.json",sources)

def test_custodian_fail_closed_if_source_absent(tmp_path):
    r=ci.attest(*synthetic(tmp_path,missing=True))
    assert not r["passed"] and not r["source_coverage_complete"]

def test_custodian_aggregate_disjoint(tmp_path):
    r=ci.attest(*synthetic(tmp_path))
    assert not r["passed"] and r["provisional_coverage_and_disjointness"] and r["overlap_total_by_identity_type"]["graph_sha256"]==0
    assert "sha256:b" not in json.dumps(r)

def test_custodian_signed_and_verified(tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    import verify_custodian_attestation as va
    private=Ed25519PrivateKey.generate()
    keypath=tmp_path/"signing_key.pem"
    keypath.write_bytes(private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    pubpath=tmp_path/"pub.pem"
    pubpath.write_bytes(private.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
    candidate,registry,sources=synthetic(tmp_path)
    att=ci.attest(candidate,registry,sources,keypath)
    assert att["passed"]
    attpath=tmp_path/"attestation.json";attpath.write_text(json.dumps(att))
    fingerprint=va.sha(private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw))
    result=va.verify(attpath,pubpath,candidate,registry,fingerprint)
    assert result["verified"]

def test_custodian_rejects_overlap(tmp_path):
    r=ci.attest(*synthetic(tmp_path,overlap=True))
    assert not r["passed"] and not r["no_overlap"]

def test_custodian_requires_exhaustive_registry(tmp_path):
    c,reg,s=synthetic(tmp_path)
    j=json.loads(reg.read_text());j["custodian_asserts_source_inventory_exhaustive"]=False
    reg.write_text(json.dumps(j))
    with pytest.raises(ValueError):
        ci.attest(c,reg,s)

def test_custodian_refuses_missing_signature_columns(tmp_path):
    c,reg,s=synthetic(tmp_path)
    (s/"protected_identity_only.csv").write_text("graph_sha256,operation_signature\nsha256:b,rx:q1\n")
    with pytest.raises(ValueError):
        ci.attest(c,reg,s)

def test_only_observed_counts_accepted_by_estimators():
    rng=np.random.default_rng(11)
    table=np.array([[1.,1.],[-1.,1.],[1.,-1.],[-1.,-1.]])
    counts=rng.multinomial(100,[.25,.25,.25,.25],size=(3,2,3))
    delta=np.linspace(-.3,.3,6);w=np.ones(6)
    for method in f.METHODS:
        cov=f.predict_cov(counts,table,delta,w,100,method,np.random.default_rng(88))
        assert cov.shape==(6,6)
        assert np.linalg.eigvalsh(cov).min()>-1e-7
        assert np.all(np.isfinite(cov))

def test_estimator_refuses_shot_mismatch():
    t=np.ones((4,2));c=np.zeros((3,2,3,4),dtype=int);c[...,0]=100
    c[0,0,0,0]=99
    with pytest.raises(ValueError):
        f.predict_cov(c,t,np.ones(6),np.ones(6),100,"plugin_multinomial",np.random.default_rng(0))

def test_posterior_predictive_never_accepts_true_probabilities():
    assert "true_p" not in f.predict_cov.__code__.co_varnames
    assert f.METHODS==("plugin_multinomial","jeffreys_posterior_predictive")
