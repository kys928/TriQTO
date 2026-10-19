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


def test_sampler_reproducer_identifies_half_boundary_with_equal_rng_prefix(tmp_path):
    from reproduce_calibration_sampler import reproduce
    from portable_calibration import array_identity
    # A minimal binary case exercises the same direct/complement boundary.
    bundles = []
    for i, q in enumerate([0.5, np.nextafter(0.5, 1.0)]):
        p = np.array([q, 1.0 - q])
        counts = np.random.Generator(np.random.PCG64(17)).multinomial(4096, p)
        bundle = CalibrationBundle()
        context = {'circuit': 'binary', 'basis_order': ['Z', 'X', 'Y']}
        eig, pairs = np.array([[1.], [-1.]]), np.empty((0, 2), dtype=np.int16)
        for path in ('phasea', 'posthoc'):
            bundle.begin(7, context, path)
            bundle.sample(p, 4096, eig, pairs, 17, counts)
            bundle.end()
        rec = bundle.save(tmp_path / str(i), {'frozen': 'test'})
        bundles.append(CalibrationBundle(rec['directory'], rec['manifest_sha256']))
    result = reproduce(*bundles, tmp_path / 'reproducer')
    assert result['half_probability_branch_crossed']
    assert result['same_rng_state_before_first_difference']
    assert result['both_conditional_binomial_loops_exact']
    controls = result['half_boundary_binary_controls']
    assert controls[1]['counts'] != controls[2]['counts']


def test_posthoc_portable_extraction_uses_verified_rows_without_product_manifest(tmp_path, monkeypatch):
    """Exercise the missing-manifest boundary with a minimal portable product."""
    product = tmp_path / 'product'
    product.mkdir()
    rows = [{'example_id': 'selected', 'evaluation_role': 'target', 'root_index': '7',
             'artifact_path': 'blind_artifacts/selected.npz', 'artifact_sha256': 'sha256:pinned'}]
    def forbidden_manifest(*args, **kwargs):
        raise AssertionError('portable extraction must not read a product manifest')
    monkeypatch.setattr(diag.baseline, 'read_csv', forbidden_manifest)
    loads = []
    def load_payload(path, expected_hash, example_id):
        loads.append((path, expected_hash, example_id))
        return {'x__only': np.zeros(1)}
    monkeypatch.setattr(diag.confirm, 'load_x_only', load_payload)
    monkeypatch.setattr(diag.frame, 'measured_delta_and_weights',
        lambda data: (np.ones(1), np.ones(1), np.empty((0, 2), dtype=np.int16)))
    monkeypatch.setattr(diag.frame, 'circuit_from_serialized', lambda data: object())
    monkeypatch.setattr(diag.latent, 'plausible_candidates', lambda clean: [(0, 1)])
    monkeypatch.setattr(diag.latent, 'finite_shot_candidate_jacobians', lambda *args: [np.ones((1, 3))])
    monkeypatch.setattr(diag.equiv, 'finite_frame_pair_geometry',
        lambda j: (np.zeros((1, 1)), np.ones((1, 1), dtype=bool)))
    monkeypatch.setattr(diag.frame, 'canonicalize_evidence', lambda *args: (np.zeros(24), None, None))
    monkeypatch.setattr(diag.latent, 'profile_log_likelihoods', lambda *args: np.zeros((1, 3)))
    result = diag.posthoc_x_only_subset(product, rows, {}, manifest_rows=rows)
    assert result['x'].shape == (1, 1, 24)
    assert loads == [(product / 'blind_artifacts/selected.npz', 'sha256:pinned', 'selected')]
    assert not (product / 'manifests').exists()


@pytest.mark.parametrize('provided', [[], ['selected', 'selected'], ['unselected']])
def test_posthoc_portable_manifest_rejects_missing_duplicate_or_extra_rows(tmp_path, provided):
    rows = [{'example_id': 'selected'}]
    with pytest.raises(RuntimeError, match='cohort mismatch'):
        diag.posthoc_x_only_subset(tmp_path, rows, {},
            manifest_rows=[{'example_id': example_id} for example_id in provided])
