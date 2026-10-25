"""Fast synthetic tests for attestation coverage and count-only candidate uncertainty."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

COHORT=Path(__file__).resolve().parents[1]/"scripts/cohort"
sys.path.insert(0,str(COHORT))
import custodian_identity_attestation as ci
import compare_nonlinear_estimators_fit as f

def synthetic(tmp_path,overlap=False,missing=False):
    candidates={"cohort_identity":"sha256:1448ca1ebbdf50149d0200dae4c6ed9eb8a39156c18e329e42ea74ca3d10203d",
               "candidates":[{"graph_sha256":"sha256:a","operation_signature":"ry:q0"}]}
    (tmp_path/"candidate.json").write_text(json.dumps(candidates))
    (tmp_path/"registry.json").write_text(json.dumps({
        "custodian_asserts_source_inventory_exhaustive":True,
        "required_sources":[{"name":"protected_identity_only",
                             "required_identity_fields":["graph_sha256","operation_signature"]}]}))
    imports=tmp_path/"sources";imports.mkdir()
    if not missing:
        (imports/"protected_identity_only.csv").write_text(
            "graph_sha256,operation_signature\n"+
            ("sha256:a,ry:q0\n" if overlap else "sha256:b,rx:q1\n"))
    return (tmp_path/"candidate.json",tmp_path/"registry.json",imports)

def test_custodian_fail_closed_if_source_absent(tmp_path):
    params=synthetic(tmp_path,missing=True)
    r=ci.attest(*params)
    assert not r["passed"] and not r["source_coverage_complete"]

def test_custodian_aggregate_disjoint(tmp_path):
    r=ci.attest(*synthetic(tmp_path))
    assert r["passed"] and r["overlap_total_by_identity_type"]["graph_sha256"]==0
    assert "sha256:b" not in json.dumps(r)

def test_custodian_rejects_overlap(tmp_path):
    r=ci.attest(*synthetic(tmp_path,overlap=True))
    assert not r["passed"] and not r["no_overlap"]

def test_custodian_requires_exhaustive_registry(tmp_path):
    c,reg,s=synthetic(tmp_path)
    j=json.loads(reg.read_text());j["custodian_asserts_source_inventory_exhaustive"]=False
    reg.write_text(json.dumps(j))
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

def test_posterior_predictive_does_not_require_true_probabilities():
    assert "true_p" not in f.predict_cov.__code__.co_varnames
    assert f.METHODS==("plugin_multinomial","jeffreys_posterior_predictive")
