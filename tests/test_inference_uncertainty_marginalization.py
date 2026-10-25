"""Numerical contracts for posterior inference, no model weights or SELECTION."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import inference_uncertainty_marginalization as m

def counts(n=4096):
    x=np.array([.24,.27,.28,.21])
    raw=np.floor(x*n).astype(int)
    raw[0]+=n-raw.sum()
    return np.broadcast_to(raw,(3,3,2,3,4)).copy()

def test_sampling_reproducible_both_methods():
    for method in ("I1","I2"):
        a=m.posterior_calibration_frequencies(counts(),method,seed=123,replicates=32)
        b=m.posterior_calibration_frequencies(counts(),method,seed=123,replicates=32)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(32,3,3,2,3,4)
        np.testing.assert_allclose(a.sum(axis=-1),1,atol=1e-14)

def test_zero_data_source_rejected():
    x=counts()
    x[0,0,0,0,:]=0
    with pytest.raises(ValueError):m.posterior_calibration_frequencies(x,"I1",seed=0)

def test_unknown_method_rejected():
    with pytest.raises(ValueError):m.posterior_calibration_frequencies(counts(),"I3",seed=0)

def test_perfectly_high_shot_posterior_concentrates():
    a=m.posterior_calibration_frequencies(counts(4096),"I1",seed=9,replicates=64)
    b=m.posterior_calibration_frequencies(counts(409600),"I1",seed=9,replicates=64)
    assert b.std(axis=0).mean() < a.std(axis=0).mean()/8

def test_jacobian_responds_to_axis_differences():
    p=np.ones((4,2,3,2,3,4))/4
    p[:,:,0,0,:,:]=np.array([.1,.2,.3,.4])
    p[:,:,0,1,:,:]=np.array([.4,.3,.2,.1])
    table=np.array([[1,-1],[-1,-1],[1,1],[-1,1]])
    j=m.jacobians_from_frequencies(p,table,epsilon=.12)
    assert j.shape==(4,2,6,3)
    assert np.abs(j[:,:,:,0]).sum()>0

def test_posterior_expected_proba_is_normalized():
    logits=np.array([[10.,0,0],[0,10.,0]],float)
    r=m.marginalized_prediction(logits)
    np.testing.assert_allclose(np.sum(r["posterior_class_probabilities"]),1)
    assert 0<=r["label"]<=2
    assert r["replica_vote_disagreement"]==.5

def test_nonfinite_model_output_is_not_silently_accepted():
    with pytest.raises(ValueError):m.marginalized_prediction(np.full((4,3),np.nan))
