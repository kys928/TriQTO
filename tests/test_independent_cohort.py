"""Scientific regression checks for the construction boundary, not model efficacy."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/cohort'))
import core as c


def test_multinomial_jacobian_preserves_axis_sign_basis_order():
    table=c.outcome_table(2)
    counts=np.zeros((3,2,3,4),dtype=int)
    for axis in range(3):
        for basis in range(3):
            counts[axis,0,basis]=[50+axis+basis,20,10,20-axis-basis]
            counts[axis,1,basis]=[20,30,20,30]
    actual=c.counts_to_jacobian(counts,table)
    expected=np.zeros((table.shape[1]*3,3))
    for axis in range(3):
        for basis in range(3):
            expected[basis*table.shape[1]:(basis+1)*table.shape[1],axis]=(counts[axis,0,basis]/100-counts[axis,1,basis]/100)@table/(.24)
    np.testing.assert_allclose(actual,expected,atol=1e-14)
    assert not np.allclose(actual,c.counts_to_jacobian(counts[:,::-1],table))


def test_batch_reconstructs_frozen_canonical_and_profiles():
    gen=np.random.default_rng(81)
    j=gen.normal(size=(3,2,4,12,3));delta=gen.normal(size=(3,12))
    weights=np.repeat(np.array([16,32,45.])[:,None],12,axis=1)
    features=c.canonical_batch(delta,j,weights)
    profiles=c.profile_batch(delta,j,weights)
    for b in range(3):
        for d in range(2):
            for a in range(4):
                for e in range(3):
                    want=c.canonicalize_evidence(delta[e],j[b,d,a],weights[e])[0]
                    np.testing.assert_allclose(features[b,d,a,e],want,rtol=1e-9,atol=1e-9)
                    want_p=c.profile_log_likelihoods(delta[e],weights[e],[j[b,d,a]])[0]
                    np.testing.assert_allclose(profiles[b,d,a,e],want_p,rtol=1e-12,atol=1e-8)


def test_null_frames_finite_and_absolute_consistency():
    j=np.zeros((3,2,1,12,3));delta=np.zeros((2,12));weights=np.ones_like(delta)
    x=c.canonical_batch(delta,j,weights)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[...,:3],0)
    np.testing.assert_array_equal(x[...,3:6],abs(x[...,:3]))
    np.testing.assert_array_equal(x[...,12:15],abs(x[...,9:12]))


def test_raw_observed_covariance_retains_shared_count_correlations():
    counts=np.array([31,19,13,37]);t=c.outcome_table(2)
    mean,cov=c.UNC.observed_mean_covariance(counts,t)
    observations=np.repeat(t,counts,axis=0)
    np.testing.assert_allclose(mean,observations.mean(axis=0))
    np.testing.assert_allclose(cov,np.cov(observations,rowvar=False,ddof=1)/100)
    assert abs(cov[0,1])>1e-4
    assert np.linalg.eigvalsh(cov).min()>-1e-12


@pytest.mark.parametrize('bad',[-1,1.5,np.nan])
def test_covariance_refuses_invalid_counts(bad):
    with pytest.raises(ValueError):c.UNC.observed_mean_covariance(np.array([5,bad]),np.ones((2,1)))


def test_jacobian_covariance_uses_independent_signs_and_basis_blocks():
    counts=np.tile(np.array([20,30,15,35]),(3,2,3,1));t=np.repeat(c.outcome_table(2)[None,:,:],3,axis=0)
    j,cov=c.UNC.finite_difference_jacobian_covariance(counts,t,.12)
    np.testing.assert_allclose(j,c.counts_to_jacobian(counts,t[0]))
    _,one=c.UNC.observed_mean_covariance(counts[0,0,0],t[0])
    np.testing.assert_allclose(cov[0,:4,:4],2*one/(.24**2))
    np.testing.assert_array_equal(cov[0,:4,4:8],0)


def test_joint_bootstrap_reproduces_and_retains_cross_covariance():
    counts=np.tile(np.array([35,15,20,30]),(3,2,3,1));counts[:,0,:,0]+=7;counts[:,0,:,3]-=7
    t=c.outcome_table(2);delta=np.linspace(-.3,.4,12);weights=np.full(12,16.)
    a=c.bootstrap_joint(counts,t,delta,weights,32,19)
    b=c.bootstrap_joint(counts,t,delta,weights,32,19)
    np.testing.assert_array_equal(a['samples'],b['samples'])
    assert np.linalg.norm(a['covariance'][:3,3:])>1e-5
    assert np.linalg.eigvalsh(a['covariance']).min()>-1e-10


def test_rng_namespaces_separate_diagnostics_budgets_draws():
    ids=[c.seed(kind,'root',n,d) for kind in ['calibration','diagnostic-reference','diagnostic-observed'] for n in c.BUDGETS for d in range(16)]
    assert len(ids)==len(set(ids))
    assert c.seed('calibration','root',4096,0)==c.seed('calibration','root',4096,0)


def test_wire_identity_detects_relabelled_structural_duplicates():
    a=[('ry',(0,)),('cx',(0,1)),('h',(1,))]
    b=[('ry',(1,)),('cx',(1,0)),('h',(0,))]
    assert c.wire_canonical_key(a,2)==c.wire_canonical_key(b,2)
    assert c.wire_canonical_key(a,2)!=c.wire_canonical_key(a[::-1],2)


def test_diagnostic_shot_weights_and_independent_reference_reconstruction():
    counts=np.array([[[[50,20,10,20]]*3,[[20,20,30,30]]*3]])
    delta,w=c.diagnostic_delta(counts,c.outcome_table(2))
    np.testing.assert_allclose(w,np.sqrt(50))
    np.testing.assert_allclose(delta[0,:4],(counts[0,1,0]-counts[0,0,0])@c.outcome_table(2)/100)


def _raw_fixture(extra=None,corrupt=False):
    import io
    import audit
    root={'n_qubits':2,'candidate_count':1,'candidates':[[0,1]]}
    cc=np.zeros((3,16,1,3,2,3,4),dtype=np.uint16)
    for i,n in enumerate(c.BUDGETS):cc[i,...,0]=n
    if corrupt:cc[0,0,0,0,0,0,0]-=1
    dc=np.zeros((13,2,3,4),dtype=np.uint16);dc[...,0]=512
    values={'calibration_counts':cc,'diagnostic_counts':dc,'outcome_features':c.outcome_table(2),
        'candidates':np.array([[0,1]]),'observation_ids':np.array([str(i) for i in range(13)]),
        'diagnostic_shots':np.full(13,512),'budgets':np.array(c.BUDGETS)}
    values.update(extra or {})
    b=io.BytesIO();np.savez(b,**values);b.seek(0)
    return audit,np.load(b),root


def test_independent_audit_rejects_true_label_in_blind_payload():
    audit,z,root=_raw_fixture({'true_mechanism':np.array(1)})
    with pytest.raises(ValueError,match='Unapproved field'):audit.validate_raw(z,root,16)


def test_independent_audit_rejects_one_missing_shot():
    audit,z,root=_raw_fixture(corrupt=True)
    with pytest.raises(ValueError,match='shot mass'):audit.validate_raw(z,root,16)


def test_independent_audit_rejects_swapped_basis_convention():
    table=c.outcome_table(2).copy();table[:,0]*=-1
    audit,z,root=_raw_fixture({'outcome_features':table})
    with pytest.raises(ValueError,match='convention'):audit.validate_raw(z,root,16)
