"""Count interventions bind scientific identity and reject corrupted evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/v0_2'))
from portable_calibration import CalibrationBundle, compare_bundles, probability_difference

CONTEXT = {'circuit': 'test-circuit', 'basis_order': ['Z', 'X', 'Y']}
P = np.array([0.25, 0.75])
EIG = np.array([[1.], [-1.]])
PAIRS = np.empty((0, 2), dtype=np.int16)


def capture(tmp_path, counts=(2, 6), probabilities=P):
    bundle = CalibrationBundle()
    for path in ('phasea', 'posthoc'):
        bundle.begin(7, CONTEXT, path)
        observed = bundle.sample(probabilities, 8, EIG, PAIRS, 17, np.array(counts))
        np.testing.assert_array_equal(observed, counts)
        bundle.end()
    return bundle.save(tmp_path, {'scientific_hash': 'frozen'})


def load(record):
    return CalibrationBundle(record['directory'], record['manifest_sha256'])


def test_replay_substitutes_integer_counts_despite_different_host_probabilities(tmp_path):
    record = capture(tmp_path)
    bundle = load(record)
    for path in ('phasea', 'posthoc'):
        bundle.begin(7, CONTEXT, path)
        # No native sample is needed or supplied during replay.
        counts = bundle.sample(np.array([0.99, 0.01]), 8, EIG, PAIRS, 17, None)
        np.testing.assert_array_equal(counts, [2, 6])
        bundle.end()
    assert bundle.save(None, {'scientific_hash': 'frozen'})['mode'] == 'substitute_counts'


@pytest.mark.parametrize('field', ['seed', 'shots', 'eig', 'pairs', 'shape'])
def test_replay_rejects_measurement_or_sampling_drift(tmp_path, field):
    record = capture(tmp_path)
    bundle = load(record)
    bundle.begin(7, CONTEXT, 'phasea')
    values = {'probabilities': P, 'shots': 8, 'eig': EIG, 'pairs': PAIRS, 'seed': 17, 'native_counts': None}
    if field == 'seed': values['seed'] = 18
    if field == 'shots': values['shots'] = 9
    if field == 'eig': values['eig'] = -EIG
    if field == 'pairs': values['pairs'] = np.array([[0, 0]], dtype=np.int16)
    if field == 'shape': values['probabilities'] = np.array([1.])
    with pytest.raises(ValueError, match='provenance mismatch'):
        bundle.sample(**values)


def test_replay_rejects_circuit_and_partial_coverage(tmp_path):
    record = capture(tmp_path)
    bundle = load(record)
    with pytest.raises(ValueError, match='circuit/configuration'):
        bundle.begin(7, {**CONTEXT, 'circuit': 'another'}, 'phasea')
    bundle.begin(7, CONTEXT, 'phasea')
    with pytest.raises(ValueError, match='call coverage'):
        bundle.end()


def test_bundle_rejects_manifest_and_archive_tampering(tmp_path):
    record = capture(tmp_path)
    with pytest.raises(ValueError, match='manifest hash'):
        CalibrationBundle(record['directory'], 'sha256:' + '0' * 64)
    archive = Path(record['directory']) / 'arrays.npz'
    archive.write_bytes(archive.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='archive hash'):
        load(record)


def test_numeric_comparison_quantifies_probabilities_and_count_changes(tmp_path):
    a = capture(tmp_path / 'a')
    b = capture(tmp_path / 'b', counts=(3, 5), probabilities=np.array([0.375, 0.625]))
    result = compare_bundles(a['directory'], a['manifest_sha256'], b['directory'], b['manifest_sha256'])
    assert result['max_probability_abs'] == 0.125
    assert result['max_probability_tv'] == 0.125
    assert result['count_different_roots'] == 1
    assert result['per_root']['7']['max_count_abs'] == 1


def test_replay_requires_all_roots_on_both_extraction_paths(tmp_path):
    bundle = load(capture(tmp_path))
    bundle.begin(7, CONTEXT, 'phasea')
    bundle.sample(P, 8, EIG, PAIRS, 17, None)
    bundle.end()
    with pytest.raises(ValueError, match='root coverage'):
        bundle.save(None, {'scientific_hash': 'frozen'})


def test_different_scientific_source_cannot_finalize_replay(tmp_path):
    bundle = load(capture(tmp_path))
    for path in ('phasea', 'posthoc'):
        bundle.begin(7, CONTEXT, path)
        bundle.sample(P, 8, EIG, PAIRS, 17, None)
        bundle.end()
    with pytest.raises(ValueError, match='scientific source'):
        bundle.save(None, {'scientific_hash': 'changed'})
