"""Recorded-count replay cannot sample; fresh draws are explicit interventions."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/v0_2'))
from portable_calibration import CalibrationBundle, canonical, sha
from calibration_replay import reconstruct_jacobians, checked_archive
from run_calibration_resampling_pilot import pairwise_disagreement, root_cluster_interval
import diagnose_step14_phasea_replay as diag


@pytest.fixture
def captured(tmp_path):
    from qiskit import QuantumCircuit
    clean = QuantumCircuit(1)
    clean.h(0)
    clean.ry(0.4, 0)
    pairs = np.empty((0, 2), dtype=np.int16)
    cfg = {'hardware_facing_frame_calibration': {
        'calibration_seed': 17, 'shots_per_basis_per_sign': 32, 'probe_strength_radians': 0.01}}
    bundle = CalibrationBundle()
    trace = diag.CalibrationTrace(bundle)
    trace.install()
    try:
        for path in ('phasea', 'posthoc'):
            trace.select_path(path)
            expected = diag.latent.finite_shot_candidate_jacobians(clean, pairs, [(0, 1)], 7, cfg)
    finally:
        trace.uninstall()
    rec = bundle.save(tmp_path / 'bundle', {'frozen': 'test'})
    return CalibrationBundle(rec['directory'], rec['manifest_sha256']), pairs, expected


def test_direct_replay_reconstructs_original_jacobian_without_sampler_or_simulator(captured, monkeypatch):
    bundle, pairs, expected = captured
    def forbidden(*args, **kwargs):
        raise AssertionError('replay attempted simulation or sampling')
    monkeypatch.setattr(np.random, 'default_rng', forbidden)
    monkeypatch.setattr(np.random, 'Generator', forbidden)
    monkeypatch.setattr(diag.latent.BASE, 'normalized_state', forbidden)
    actual, _ = reconstruct_jacobians(bundle, '7', pairs)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('kind', ['incomplete', 'seed', 'basis', 'pairs'])
def test_direct_replay_rejects_incomplete_or_mismatched_provenance(captured, kind):
    bundle, pairs, _ = captured
    if kind == 'incomplete': bundle.roots['7']['calls'].pop()
    if kind == 'seed': bundle.roots['7']['calls'][0]['seed'] += 1
    if kind == 'basis': bundle.roots['7']['calls'][0]['basis'] = 'Q'
    if kind == 'pairs': pairs = np.array([[0, 0]], dtype=np.int16)
    with pytest.raises(ValueError, match='contract|provenance'):
        reconstruct_jacobians(bundle, '7', pairs)


def test_fresh_draws_have_explicit_reproducible_seed_and_shot_budget(captured):
    bundle, pairs, _ = captured
    a, ca = reconstruct_jacobians(bundle, '7', pairs, mode='resample', realization=0, multiplier=2)
    b, cb = reconstruct_jacobians(bundle, '7', pairs, mode='resample', realization=0, multiplier=2)
    _, cc = reconstruct_jacobians(bundle, '7', pairs, mode='resample', realization=1, multiplier=2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ca, cb)
    assert np.all(ca.sum(axis=1) == 64)
    assert not np.array_equal(ca, cc)
    with pytest.raises(ValueError, match='preregistered'):
        reconstruct_jacobians(bundle, '7', pairs, mode='resample', realization=32)


def test_archive_hash_and_dependency_closure_are_required(tmp_path):
    (tmp_path / 'input.bin').write_bytes(b'evidence')
    doc = {'schema': 'triqto.calibration_evidence_archive.v1', 'files': {'input.bin': sha(b'evidence')}}
    raw = canonical(doc)
    (tmp_path / 'archive_manifest.json').write_bytes(raw)
    checked_archive(tmp_path, sha(raw))
    (tmp_path / 'input.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='file hash'):
        checked_archive(tmp_path, sha(raw))


def test_pairwise_disagreement_excludes_self_pairs_and_cluster_unit_is_root():
    labels = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 1]])
    np.testing.assert_allclose(pairwise_disagreement(labels), [2 / 3, 0, 0])
    # Two rows sharing root 7 are always resampled together. Draws stay within roots.
    values = np.tile([0., 0., 1.], (32, 1))
    result = root_cluster_interval(values, [7, 7, 8])
    assert result['mean'] == 1 / 3
    assert result['root_cluster_percentile_95_interval'] == [0., 1.]
