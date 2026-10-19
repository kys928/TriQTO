#!/usr/bin/env python3
"""Preregistered, label-free frozen-model calibration stability diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from calibration_replay import PortableReplay, environment, ROOT
from portable_calibration import canonical, sha
import analyze_step14_local_frame_canonicalization as frame


def softmax(logits):
    scores = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def margins(logits):
    ordered = np.sort(logits, axis=-1)
    return ordered[..., -1] - ordered[..., -2]


def pairwise_disagreement(labels):
    """Per-example disagreement for two distinct draws, not self-pairs."""
    draws = len(labels)
    if draws < 2:
        raise ValueError('two or more draws required')
    counts = np.stack([(labels == c).sum(axis=0) for c in range(3)], axis=-1)
    return 1.0 - (counts * (counts - 1)).sum(axis=-1) / (draws * (draws - 1))


def root_cluster_interval(values, root_indices):
    """Values [draw,example]; resample whole roots, keep repeated draws together."""
    values = np.asarray(values, dtype=np.float64)
    roots = np.asarray(root_indices)
    unique = np.unique(roots)
    by_example = values.mean(axis=0)
    sums = np.array([by_example[roots == r].sum() for r in unique])
    sizes = np.array([(roots == r).sum() for r in unique])
    rng = np.random.Generator(np.random.PCG64(2026101901))
    sampled = rng.integers(0, len(unique), size=(2000, len(unique)))
    estimates = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return {'mean': float(values.mean()), 'root_cluster_percentile_95_interval':
            np.quantile(estimates, [0.025, 0.975]).tolist(),
            'draw_mean_range': [float(x) for x in [values.mean(axis=1).min(), values.mean(axis=1).max()]],
            'interpretation': 'descriptive uncertainty on selected roots; no population or accuracy claim'}


def candidate_proxy(ensemble, mask):
    scores = np.max(ensemble, axis=-1)
    return np.argmax(np.where(mask, scores, -np.inf), axis=1)


def geometry_arrays(replay, calibration, recorded, expected, table, reference_table):
    arrays = {}
    root_rows = []
    for root, (jacs, counts) in calibration.items():
        j = np.stack(jacs)
        r, e = np.stack(recorded[root][0]), np.stack(expected[root][0])
        arrays['jacobian_' + root] = j
        if counts is not None:
            arrays['counts_' + root] = counts
        norms = np.linalg.norm(j, axis=1)
        normalized = j / np.maximum(norms[:, None, :], 1e-12)
        arrays['raw_column_norms_' + root] = norms
        arrays['normalized_singular_values_' + root] = np.linalg.svd(normalized, compute_uv=False)
        for name, ref in [('recorded', r), ('expected', e)]:
            relative = np.linalg.norm(j - ref, axis=(1, 2)) / np.maximum(np.linalg.norm(ref, axis=(1, 2)), 1e-12)
            arrays['jacobian_relative_to_' + name + '_' + root] = relative
        root_rows.append({'root': root,
            'jacobian_relative_expected_median': float(np.median(arrays['jacobian_relative_to_expected_' + root])),
            'jacobian_relative_expected_max': float(np.max(arrays['jacobian_relative_to_expected_' + root]))})
    nc = table['mask'].shape[1]
    ne = len(replay.evidence)
    weighted_norms, weighted_singular, conditions, normalized_change = (
        np.zeros((ne, nc, 3)), np.zeros((ne, nc, 3)), np.zeros((ne, nc)), np.zeros((ne, nc)))
    for i, row in enumerate(replay.document['selected_rows']):
        root = str(row['root_index'])
        _, w = replay.evidence[i]
        for c, jac in enumerate(calibration[root][0]):
            jw = jac * w[:, None]
            norms = np.linalg.norm(jw, axis=0)
            normalized = jw / np.maximum(norms, 1e-12)[None, :]
            old = recorded[root][0][c] * w[:, None]
            old /= np.maximum(np.linalg.norm(old, axis=0), 1e-12)[None, :]
            weighted_norms[i, c] = norms
            singular = np.linalg.svd(normalized, compute_uv=False)
            weighted_singular[i, c] = singular
            ridge = frame.PROBE_SPEC['ridge']
            conditions[i, c] = (singular[0] ** 2 + ridge) / (singular[-1] ** 2 + ridge)
            normalized_change[i, c] = np.linalg.norm(normalized - old)
    arrays.update(weighted_column_norms=weighted_norms, weighted_normalized_singular_values=weighted_singular,
                  regularized_system_condition=conditions, normalized_frame_change=normalized_change)
    mask = table['mask']
    delta = table['x'] - reference_table['x']
    std = np.asarray(replay.checkpoints[0]['normalization_std'])
    arrays['feature_change'] = delta
    arrays['standardized_feature_change'] = delta / std[None, None, :]
    return arrays, {'per_root': root_rows,
        'feature_change_rms': float(np.sqrt(np.mean(delta[mask] ** 2))),
        'standardized_feature_change_rms': float(np.sqrt(np.mean(arrays['standardized_feature_change'][mask] ** 2))),
        'normalized_frame_change_median': float(np.median(normalized_change[mask])),
        'regularized_system_condition_max': float(conditions[mask].max())}


def save_draw(path, table, pooled, ensemble, geometry):
    np.savez_compressed(path, **geometry, features=table['x'], mask=table['mask'], profiles=table['profiles'],
                        distance=table['distance'], gate=table['gate'], pooled_logits=pooled,
                        softmax_scores=softmax(pooled), top_two_margin=margins(pooled),
                        candidate_logits=ensemble, candidate_proxy=candidate_proxy(ensemble, table['mask']))


def run(evidence, expected_hash, output):
    if np.__version__ != '2.3.2':
        raise RuntimeError('preregistered sampler requires NumPy 2.3.2')
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    replay = PortableReplay(evidence, expected_hash)
    started = time.monotonic()
    # The recorded-count gate must run without either simulation or regeneration.
    original = np.random.default_rng
    np.random.default_rng = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('sampler forbidden in replay'))
    try:
        recorded, table, logits, candidates, gate = replay.replay()
    finally:
        np.random.default_rng = original
    print(json.dumps({'replay_gate': gate, 'jacobian_roots_exact': len(recorded)}), flush=True)
    expected = replay.reconstruct('expected')
    expected_table = replay.extract(expected)
    expected_logits, expected_candidates = replay.infer(expected_table)
    np.savez_compressed(output / 'recorded_reference.npz', features=table['x'], pooled_logits=logits,
        candidate_logits=candidates, mask=table['mask'], softmax_scores=softmax(logits))
    geom, _ = geometry_arrays(replay, expected, recorded, expected, expected_table, table)
    save_draw(output / 'noiseless_finite_difference_reference.npz', expected_table, expected_logits, expected_candidates, geom)
    reference_labels = np.argmax(logits, axis=1)
    expected_labels = np.argmax(expected_logits, axis=1)
    roots = [int(r['root_index']) for r in replay.document['selected_rows']]
    seed_arms, all_metrics = {}, []
    for multiplier in (1, 2, 4):
        pooled_draws, candidate_draws = [], []
        arm_dir = output / f'shots_{multiplier}x'
        arm_dir.mkdir()
        for draw in range(32):
            calibration = replay.reconstruct('resample', realization=draw, multiplier=multiplier)
            current = replay.extract(calibration)
            pooled, ensemble = replay.infer(current)
            geometry, metrics = geometry_arrays(replay, calibration, recorded, expected, current, table)
            save_draw(arm_dir / f'draw_{draw:02d}.npz', current, pooled, ensemble, geometry)
            pooled_draws.append(pooled)
            candidate_draws.append(candidate_proxy(ensemble, current['mask']))
            all_metrics.append({'shot_multiplier': multiplier, 'draw': draw, **metrics})
            print(json.dumps({'shot_multiplier': multiplier, 'draw': draw,
                'label_changes_vs_recorded': int(np.count_nonzero(np.argmax(pooled, axis=1) != reference_labels)),
                'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
        pooled_draws = np.stack(pooled_draws)
        candidate_draws = np.stack(candidate_draws)
        labels = np.argmax(pooled_draws, axis=-1)
        flips = labels != reference_labels[None, :]
        against_expected = labels != expected_labels[None, :]
        candidate_changes = candidate_draws != candidate_proxy(candidates, table['mask'])[None, :]
        probabilities = softmax(pooled_draws)
        arm = {'draw_count': 32, 'shot_multiplier': multiplier,
            'flip_vs_recorded': root_cluster_interval(flips, roots),
            'flip_vs_noiseless_finite_difference': root_cluster_interval(against_expected, roots),
            'candidate_proxy_change_vs_recorded': root_cluster_interval(candidate_changes, roots),
            'pairwise_disagreement_per_example': pairwise_disagreement(labels).tolist(),
            'pairwise_disagreement_mean': float(pairwise_disagreement(labels).mean()),
            'softmax_score_std_per_example_class': np.std(probabilities, axis=0, ddof=1).tolist(),
            'softmax_score_std_mean': float(np.std(probabilities, axis=0, ddof=1).mean()),
            'margin_mean_per_example': margins(pooled_draws).mean(axis=0).tolist(),
            'margin_std_per_example': margins(pooled_draws).std(axis=0, ddof=1).tolist(),
            'margin_change_vs_recorded_mean': float((margins(pooled_draws) - margins(logits)[None, :]).mean()),
            'flips_per_draw': flips.sum(axis=1).tolist(),
            'per_example': [{'example_id': row['example_id'], 'root_index': int(row['root_index']),
                'label_frequencies': np.bincount(labels[:, i], minlength=3).tolist(),
                'flip_fraction_vs_recorded': float(flips[:, i].mean()),
                'flip_fraction_vs_noiseless': float(against_expected[:, i].mean())}
                for i, row in enumerate(replay.document['selected_rows'])]}
        seed_arms[str(multiplier)] = arm
        (output / f'summary_{multiplier}x.json').write_bytes(canonical(arm))
    replay.verify_unchanged()
    result = {'schema': 'triqto.calibration_resampling_pilot.v1',
        'protocol_sha256': sha((ROOT / 'docs/experiments/CALIBRATION_RESAMPLING_PILOT.md').read_bytes()),
        'archive_manifest_sha256': expected_hash, 'environment': environment(),
        'cohort_examples': 36, 'cohort_roots': len(recorded), 'replay_gate': gate,
        'noiseless_reference_label_changes_vs_recorded': int(np.count_nonzero(expected_labels != reference_labels)),
        'arms': seed_arms, 'draw_metrics': all_metrics,
        'elapsed_seconds': time.monotonic() - started,
        'scientific_boundary': {'sealed_truth_accessed': False, 'weights_changed': False,
            'hyperparameters_selected': False, 'full_holdout_rescored': False,
            'accuracy_measured': False, 'abstention_supported': False,
            'noiseless_reference': 'analysis-only expected finite-difference statistics at fixed epsilon; not exact derivative'},
        'files': {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'result.json').write_bytes(canonical(result))
    print(json.dumps({'completed': True, 'result_sha256': sha((output / 'result.json').read_bytes()),
          'flip_means': {k: v['flip_vs_recorded']['mean'] for k, v in seed_arms.items()}}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--archive-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.evidence, args.archive_manifest_sha256, args.output)
