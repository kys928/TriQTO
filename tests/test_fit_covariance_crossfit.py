"""Scientific mechanics for 6D FIT-family heldout covariance calibration."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import crossfit_nonlinear_covariance_fit as f

def test_single_source_uncertainty_psd_and_finite():
    counts=np.tile(np.array([49,51,46,54]),(3,2,3,1))
    table=np.array([[1,1],[-1,1],[1,-1],[-1,-1]],dtype=float)
    cov=f.boot_cov(counts,table,np.array([.3,-.2,.1,.14,-.3,.2]),
                   np.full(6,14.),200,f.fixed_seed("synthetic"))
    assert cov.shape==(6,6)
    np.testing.assert_allclose(cov,cov.T,atol=1e-10)
    assert np.linalg.eigvalsh(cov).min()>-1e-9
    assert np.isfinite(cov).all()

def test_family_folds_deterministic():
    assert f.fold("family_123")==f.fold("family_123")
    assert len({f.fold("family_"+str(i)) for i in range(100)})==5

def test_crossfit_coefficient_preserves_correlated_psd():
    rg=np.random.default_rng(42);x=rg.normal(size=(6,6));cov=x@x.T
    diag=np.diag([2.,2.,2.,.5,.5,.5])
    result=diag@cov@diag
    assert np.linalg.eigvalsh(result).min()>-1e-8
    assert np.linalg.norm(result[:3,3:])>0
