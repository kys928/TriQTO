"""Mechanism-independent physics and feature consistency for score interventions."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/cohort"))
import remedy_feature_transforms as r

def example():
    c=np.array([.5,-.2,.15])
    m=np.array([.12,-.15,.3])
    x=np.r_[c,np.abs(c),c/np.linalg.norm(c),m,np.abs(m),np.arange(9.)]
    b=np.array([[.01,.004,-.002,.003,.001,.006],
                [.004,.015,0,-.003,.004,0],
                [-.002,0,.018,0,-.001,.002],
                [.003,-.003,0,.012,0,.001],
                [.001,.004,-.001,0,.02,.003],
                [.006,0,.002,.001,.003,.015]])
    vals,vec=np.linalg.eigh(b)
    b=vec@np.diag(np.maximum(vals,1e-3))@vec.T
    return x,b

def test_r0_is_bitwise_exact():
    x,c=example()
    np.testing.assert_array_equal(x,r.apply(x,c,"R0"))

def test_r1_preserves_coords_and_derived_slots():
    x,c=example();y=r.apply(x,c,"R1")
    np.testing.assert_array_equal(x[:9],y[:9])
    np.testing.assert_array_equal(x[15:],y[15:])
    np.testing.assert_allclose(y[12:15],np.abs(y[9:12]))
    assert np.linalg.norm(y[9:12])<np.linalg.norm(x[9:12])

def test_r2_uses_cross_covariance_and_is_consistent():
    x,c=example()
    y=r.apply(x,c,"R2")
    np.testing.assert_allclose(y[3:6],np.abs(y[:3]))
    np.testing.assert_allclose(y[6:9],y[:3]/np.linalg.norm(y[:3]))
    np.testing.assert_allclose(y[12:15],np.abs(y[9:12]))
    np.testing.assert_array_equal(x[15:],y[15:])
    c0=np.diag(np.diag(c))
    y0=r.apply(x,c0,"R2")
    assert np.linalg.norm(y[:6]-y0[:6])>1e-9

def test_observed_covariance_scale_is_psd_preserving():
    x,c=example();s=r.calibrated_covariance(c,2.0,1.2)
    assert np.linalg.eigvalsh(s).min()>-1e-8
    assert s[0,3]!=0

def test_refuse_inconsistent_or_nonfinite_features():
    x,c=example();x[12]=0
    with pytest.raises(ValueError):r.apply(x,c,"R1")
    x,c=example();c[0,0]=np.nan
    with pytest.raises(ValueError):r.apply(x,c,"R2")

def test_zero_alpha_r2_identity_and_abs_consistency():
    x,c=example();y=r.apply(x,c,"R2",alpha=0)
    np.testing.assert_allclose(y,x,atol=1e-12)

def test_global_scale_fail_closed():
    x,c=example()
    with pytest.raises(ValueError):r.calibrated_covariance(c,100,1)
