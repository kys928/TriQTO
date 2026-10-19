"""Read-only finite-shot replay audit has no effect on the frozen sampler."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
V02 = ROOT / "scripts" / "v0_2"
if str(V02) not in sys.path:
    sys.path.insert(0, str(V02))
SPEC = importlib.util.spec_from_file_location(
    "phasea_calibration_trace_test", V02 / "diagnose_step14_phasea_replay.py"
)
assert SPEC and SPEC.loader
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def test_calibration_trace_preserves_sequential_sampling_and_jacobians():
    audit = diag.CalibrationTrace()
    original_finite = audit._finite
    probabilities = np.asarray([0.3, 0.7], dtype=np.float64)
    eig = np.asarray([[1.], [-1.]], dtype=np.float64)
    pairs = np.empty((0, 2), dtype=np.int64)

    def stub_original_finite(_clean, _pairs, _candidates, _root, _cfg):
        audit._empirical(probabilities, 4096, eig, pairs, 17)
        # The injected wrapper replays the identical seed and checks equality.
        observed = diag.latent.BASE.empirical_stats(
            probabilities, 4096, eig, pairs, 42
        )
        return [np.full((3, 3), float(observed[0][0]), dtype=np.float64)]

    audit._finite = stub_original_finite
    cfg = {"hardware_facing_frame_calibration": {
        "calibration_seed": 2026090402,
        "shots_per_basis_per_sign": 4096
    }}
    audit.install()
    try:
        for name in ("phasea", "posthoc"):
            audit.select_path(name)
            result = diag.latent.finite_shot_candidate_jacobians(
                None, pairs, [(0, 1)], 40025, cfg
            )
            assert len(result) == 1
    finally:
        audit.uninstall()
        diag.latent.finite_shot_candidate_jacobians = original_finite

    outcome = audit.summarize()
    assert outcome["traced_root_count"] == 1
    assert outcome["total_sample_calls"] == 1
    assert outcome["per_root_sampled_count_and_probability_hash_equal_across_paths"]
    assert outcome["per_root"]["40025"]["sample_call_count"] == 1
    assert outcome["per_root"]["40025"]["finite_shot_jacobians"]["shape"] == [1, 3, 3]


def test_calibration_trace_rejects_divergent_sampled_count(monkeypatch):
    audit=diag.CalibrationTrace()
    probabilities=np.asarray([0.5,0.5],dtype=np.float64)
    eig=np.asarray([[1.],[-1.]],dtype=np.float64)
    pairs=np.empty((0,2),dtype=np.int64)
    monkeypatch.setattr(audit, "_empirical",
                        lambda p,s,e,pa,se: (np.asarray([999.]),np.asarray([]),0.))
    audit.active_root=123
    audit.hashes={k:diag.hashlib.sha256() for k in
        ("probabilities","sample_counts","sampled_statistics","sampling_seed")}
    with pytest.raises(RuntimeError,match="instrumentation changed sampled evidence"):
        audit.empirical(probabilities,4096,eig,pairs,123)


def test_portable_count_intervention_bypasses_sampler_and_preserves_jacobians(tmp_path, monkeypatch):
    from qiskit import QuantumCircuit
    from portable_calibration import CalibrationBundle
    clean = QuantumCircuit(1)
    clean.h(0)
    clean.ry(0.4, 0)
    pairs = np.empty((0, 2), dtype=np.int16)
    cfg = {'hardware_facing_frame_calibration': {
        'calibration_seed': 17, 'shots_per_basis_per_sign': 32,
        'probe_strength_radians': 0.01}}
    capture = CalibrationBundle()
    audit = diag.CalibrationTrace(capture)
    audit.install()
    try:
        for path in ('phasea', 'posthoc'):
            audit.select_path(path)
            expected = diag.latent.finite_shot_candidate_jacobians(clean, pairs, [(0, 1)], 7, cfg)
    finally:
        audit.uninstall()
    record = capture.save(tmp_path, {'frozen': 'scientific-source'})
    replay = CalibrationBundle(record['directory'], record['manifest_sha256'])
    audit = diag.CalibrationTrace(replay)
    # Force a different simulated probability vector. The integer donor evidence
    # must still produce the captured Jacobians without ever calling a sampler.
    monkeypatch.setattr(diag.latent.BASE, 'basis_probabilities',
                        lambda *a, **k: np.array([0.9, 0.1]))
    def forbidden_sampler(*a, **k):
        raise AssertionError('count substitution must not sample')
    monkeypatch.setattr(diag.np.random, 'default_rng', forbidden_sampler)
    audit.install()
    try:
        for path in ('phasea', 'posthoc'):
            audit.select_path(path)
            actual = diag.latent.finite_shot_candidate_jacobians(clean, pairs, [(0, 1)], 7, cfg)
            np.testing.assert_array_equal(actual, expected)
    finally:
        audit.uninstall()
    replay.save(None, {'frozen': 'scientific-source'})
