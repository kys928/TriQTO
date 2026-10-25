#!/usr/bin/env python3
"""R0/R1/R2 observed-shot canonical-feature interventions, frozen heads untouched.

This module implements SCORE-PATH interventions only: it does not change
candidate generation, Jacobian, profiles, pair geometry, frame kernel or
candidate pooling. It is deployable at the feature-input level but does not
claim to fix the separate pooling path.

R0 is exact identity. R1 attenuates 3 matched values. R2 applies a full
6x6 correlation-aware Bayesian/Wiener-style damping on (c,m), maintaining
all redundant 24D coordinate and magnitude slots consistently.

The covariance must be estimated from observed single-acquisition counts;
normalization factors are FIT-trained global coefficients, never oracle inputs.
"""
from __future__ import annotations
import numpy as np

ORDER=("R0","R1","R2")
SIGNAL_FLOOR=0.05
ALPHA_R1=1.0
ALPHA_R2=1.0

def validate(feature,cov):
    x=np.asarray(feature,dtype=np.float64)
    c=np.asarray(cov,dtype=np.float64)
    if x.shape!=(24,) or c.shape!=(6,6):
        raise ValueError("Expected one 24-feature vector and joint 6x6 covariance")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)):
        raise ValueError("Non-finite feature or covariance")
    if not np.allclose(c,c.T,atol=1e-9,rtol=1e-9):
        raise ValueError("Non-symmetric covariance")
    if np.linalg.eigvalsh(c).min() < -1e-8:
        raise ValueError("Covariance must be PSD")
    if not np.allclose(x[3:6],np.abs(x[:3]),atol=1e-7):
        raise ValueError("Input coordinates have inconsistent abs slots")
    if not np.allclose(x[12:15],np.abs(x[9:12]),atol=1e-7):
        raise ValueError("Input matched-response abs slots inconsistent")
    norm=np.linalg.norm(x[:3])
    if norm >1e-9 and not np.allclose(x[6:9],x[:3]/norm,atol=1e-7):
        raise ValueError("Input unit direction inconsistent")
    return x,c

def calibrated_covariance(cov,coord_factor,matched_factor):
    sigma=np.asarray(cov,dtype=np.float64)
    if sigma.shape!=(6,6) or not np.isfinite(sigma).all():
        raise ValueError("Invalid joint covariance")
    if not (0.25<=coord_factor<=8 and 0.25<=matched_factor<=8):
        raise ValueError("Unfrozen/out-of-range FIT covariance factor")
    d=np.diag([np.sqrt(coord_factor)]*3+[np.sqrt(matched_factor)]*3)
    adjusted=d@sigma@d
    if np.linalg.eigvalsh(adjusted).min() < -1e-8:
        raise ValueError("Calibration not PSD")
    return adjusted

def apply(feature,cov,remedy,*,alpha=None):
    x,c=validate(feature,cov)
    if remedy not in ORDER:raise ValueError("Unknown intervention")
    if remedy=="R0":
        return x.copy()
    strength=(ALPHA_R1 if remedy=="R1" else ALPHA_R2) if alpha is None else float(alpha)
    if not np.isfinite(strength) or strength<0 or strength>4:
        raise ValueError("Unbounded intervention strength")
    ans=x.copy()
    if remedy=="R1":
        m=x[9:12]
        v=np.maximum(np.diag(c)[3:],0.)
        rel=m*m/(m*m+strength*v+SIGNAL_FLOOR**2)
        ans[9:12]=m*rel
        ans[12:15]=np.abs(ans[9:12])
    else:
        orig=np.r_[x[:3],x[9:12]]
        # PSD signal prior, with floor guarding zero directions.
        signal=np.maximum(orig*orig,SIGNAL_FLOOR**2)
        S=np.diag(signal)
        z=S@np.linalg.solve(S+strength*c,orig)
        ans[:3]=z[:3]
        ans[3:6]=np.abs(z[:3])
        # Unit direction for new coordinate vector, not independent attenuation.
        length=np.linalg.norm(z[:3])
        ans[6:9]=z[:3]/max(length,1e-12)
        ans[9:12]=z[3:]
        ans[12:15]=np.abs(z[3:])
    if not np.isfinite(ans).all():
        raise RuntimeError("Non-finite intervention output")
    if not np.array_equal(ans[15:24],x[15:24]):
        raise RuntimeError("Other geometry unexpectedly altered")
    return ans
