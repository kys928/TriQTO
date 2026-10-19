#!/usr/bin/env python3
"""Portable CPU replay or explicit fresh calibration draws. Never reads truth."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from portable_calibration import CalibrationBundle, array_identity, circuit_identity, sha, canonical
import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import predict_step14_equivalence_aware_fresh_holdout_blind as blind
import step14_equivalence_aware_common as equiv
import diagnose_step14_phasea_replay as diag

ROOT = Path(__file__).resolve().parents[2]


def checked_archive(directory, expected):
    directory = Path(directory)
    raw = (directory / 'archive_manifest.json').read_bytes()
    if sha(raw) != expected:
        raise ValueError('archive manifest hash mismatch')
    doc = json.loads(raw)
    if doc.get('schema') != 'triqto.calibration_evidence_archive.v1':
        raise ValueError('unsupported archive schema')
    for rel, expected_hash in doc['files'].items():
        path = directory / rel
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path.read_bytes()) != expected_hash:
            raise ValueError('archive file hash mismatch: ' + rel)
    return doc


def draw_counts(bundle, root, realization, multiplier):
    """Explicitly fresh draws; never used by recorded-count replay."""
    if realization not in range(32) or multiplier not in (1, 2, 4):
        raise ValueError('draw outside preregistered pilot')
    calls = bundle.roots[root]['calls']
    return np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [20261019, realization, int(root), i]))).multinomial(call['shots'] * multiplier, p)
        for i, (call, p) in enumerate(zip(calls, bundle.arrays['p' + root]))])


def reconstruct_jacobians(bundle, root, pairs, *, mode='replay', realization=None, multiplier=1):
    """Reconstruct with original statistic/order/arithmetic; verify full call contract."""
    rec = bundle.roots[root]
    ctx, calls = rec['context'], rec['calls']
    spec = ctx['calibration_configuration']
    candidates = ctx['ordered_candidates']
    if (ctx['basis_order'] != list(frame.AXES) or ctx['mechanism_order'] != list(frame.MECHANISMS)
            or ctx['loop_order'] != ['candidate', 'mechanism', 'sign', 'basis']
            or array_identity(pairs) != ctx['ordered_pairs'] or len(calls) != len(candidates) * 18):
        raise ValueError('incomplete or mismatched calibration call contract')
    eig = latent.BASE.eigenvalue_table(ctx['circuit']['qubits'])
    for i, call in enumerate(calls):
        candidate, mechanism, sign, basis = i // 18, (i // 6) % 3, ['plus', 'minus'][(i // 3) % 2], frame.AXES[i % 3]
        q, b = candidates[candidate]
        expected_seed = latent.BASE.stable_seed('step14-latent-frame', spec['calibration_seed'], int(root), q, b,
                                                frame.MECHANISMS[mechanism], sign, basis)
        expected = {'ordinal': i, 'seed': expected_seed, 'shots': spec['shots_per_basis_per_sign'],
                    'probability_shape': [len(eig)], 'measurement_eigenvalues': array_identity(eig),
                    'ordered_pairs': array_identity(pairs), 'candidate_index': candidate,
                    'mechanism_index': mechanism, 'sign': sign, 'basis': basis}
        if call != expected:
            raise ValueError('sampling/measurement provenance mismatch')
    if mode == 'replay':
        counts = bundle.arrays['c' + root]
    elif mode == 'resample':
        counts = draw_counts(bundle, root, realization, multiplier)
    elif mode == 'expected':
        counts = None
    else:
        raise ValueError('explicit replay, resample, or expected mode required')
    stats = []
    for i, call in enumerate(calls):
        distribution = (bundle.arrays['p' + root][i] if counts is None else
                        counts[i].astype(np.float64) / float(call['shots'] * (multiplier if mode == 'resample' else 1)))
        stats.append(latent.BASE.stats_from_distribution(distribution, eig, pairs))
    width = 3 * (ctx['circuit']['qubits'] + len(pairs) + 1)
    jacs = []
    for c in range(len(candidates)):
        jac = np.zeros((width, 3), dtype=np.float64)
        for m in range(3):
            start = c * 18 + m * 6
            plus = latent.flatten_stats(stats[start:start + 3])
            minus = latent.flatten_stats(stats[start + 3:start + 6])
            jac[:, m] = (plus - minus) / (2.0 * float(spec['probe_strength_radians']))
        jacs.append(jac)
    return jacs, counts


class PortableReplay:
    def __init__(self, directory, expected):
        self.directory = Path(directory)
        self.archive_hash = expected
        diag.oracle_compat.apply_frozen_support_bound()
        self.document = checked_archive(self.directory, expected)
        rec = self.document['bundles']['capture_native']
        self.bundle = CalibrationBundle(self.directory / rec['directory'], rec['manifest_sha256'])
        self.cfg = latent.load_frozen_config()
        self.method_cfg = json.loads((ROOT / 'configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json').read_text())
        prov = self.document['scientific_provenance']
        expected_prov = {k: v for k, v in prov.items() if k.endswith('sha256') or k == 'checkpoint_provenance'}
        if self.bundle.document['provenance'] != expected_prov:
            raise ValueError('scientific provenance mismatch')
        for rel, key in [('configs/v0_2/step14_latent_frame_inference.json', 'latent_config_sha256'),
                         ('configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json', 'confirm_config_sha256')]:
            if sha((ROOT / rel).read_bytes()) != prov[key]:
                raise ValueError('frozen configuration drift')
        self.loaded = []
        self.pairs = {}
        self.evidence = []
        rows = self.document['selected_rows']
        if len(rows) != 36 or len({r['example_id'] for r in rows}) != len(rows):
            raise ValueError('targeted cohort identity mismatch')
        for row in rows:
            data = blind.load_x_only_artifact(self.directory / 'product', row)
            delta, weights, pairs = frame.measured_delta_and_weights(data)
            root = str(row['root_index'])
            ctx = self.bundle.roots[root]['context']
            clean = frame.circuit_from_serialized(data)
            if (ctx['circuit'] != circuit_identity(clean) or
                ctx['ordered_candidates'] != [list(c) for c in latent.plausible_candidates(clean)] or
                ctx['calibration_configuration'] != self.cfg['hardware_facing_frame_calibration']):
                raise ValueError('circuit or candidate provenance mismatch')
            self.pairs[root] = pairs
            self.loaded.append(data)
            self.evidence.append((delta, weights))
        if set(self.pairs) != set(self.bundle.roots):
            raise ValueError('calibration root coverage mismatch')
        self.checkpoints = []
        method = self.method_cfg['method_identity']
        for seed in method['candidate_network_seeds']:
            name = f'candidate_network_seed{seed}.pt'
            path = self.directory / 'checkpoints' / name
            if sha(path.read_bytes()) != method['checkpoint_sha256'][name]:
                raise ValueError('checkpoint hash mismatch')
            ck = torch.load(path, map_location='cpu', weights_only=False)
            if (ck['method_id'] != method['method_id'] or int(ck['seed']) != seed or
                float(ck['selected_tau']) != method['tau'] or
                float(ck['selected_frame_temperature']) != method['frame_temperature']):
                raise ValueError('checkpoint internal identity mismatch')
            self.checkpoints.append(ck)
        self.source_hashes = dict(self.document['files'])

    def verify_unchanged(self):
        checked_archive(self.directory, self.archive_hash)
        if self.document['files'] != self.source_hashes:
            raise ValueError('source inventory changed')

    def reconstruct(self, mode, realization=None, multiplier=1):
        return {root: reconstruct_jacobians(self.bundle, root, pairs, mode=mode,
                realization=realization, multiplier=multiplier)
                for root, pairs in sorted(self.pairs.items(), key=lambda v: int(v[0]))}

    def extract(self, calibration):
        used = set()
        original = latent.finite_shot_candidate_jacobians
        def recorded(clean, pairs, candidates, root, cfg):
            root = str(root)
            ctx = self.bundle.roots[root]['context']
            if (circuit_identity(clean) != ctx['circuit'] or array_identity(pairs) != ctx['ordered_pairs']
                or [list(c) for c in candidates] != ctx['ordered_candidates']
                or cfg['hardware_facing_frame_calibration'] != ctx['calibration_configuration']):
                raise ValueError('extraction provenance mismatch')
            used.add(root)
            return calibration[root][0]
        latent.finite_shot_candidate_jacobians = recorded
        try:
            table = blind.extract_blind_table(self.directory / 'product', self.document['selected_rows'],
                {int(k): v for k, v in self.document['safe_roots'].items()}, self.cfg, 0)
        finally:
            latent.finite_shot_candidate_jacobians = original
        if used != set(self.bundle.roots):
            raise ValueError('extraction root coverage mismatch')
        return table

    def infer(self, table):
        candidates = []
        for ck in self.checkpoints:
            _, logits = equiv.infer_candidate_network(table['x'], table['mask'], state_dict=ck['state_dict'],
                mean=np.asarray(ck['normalization_mean'], dtype=np.float32),
                std=np.asarray(ck['normalization_std'], dtype=np.float32), latent_cfg=self.cfg,
                device=torch.device('cpu'))
            candidates.append(logits)
        ensemble = np.mean(np.stack(candidates), axis=0)
        method = self.method_cfg['method_identity']
        pooled = equiv.equivalence_aware_pool_batch(ensemble, table['profiles'], table['distance'],
                 table['gate'], table['mask'], tau=method['tau'], frame_temperature=method['frame_temperature'])
        return pooled, ensemble

    def replay(self):
        cal = self.reconstruct('replay')
        donor = json.loads((self.directory / 'reports/capture.json').read_text())['reports']['native']['result']
        for root, (jacs, _) in cal.items():
            if diag.digest_array(np.stack(jacs)) != donor['sampled_calibration_provenance']['per_root'][root]['finite_shot_jacobians']:
                raise ValueError('recorded-count Jacobian is not exact donor')
        table = self.extract(cal)
        # Independently reconstructed extraction must use the same recorded frames.
        original = latent.finite_shot_candidate_jacobians
        latent.finite_shot_candidate_jacobians = lambda c, p, ca, r, cfg: cal[str(r)][0]
        try:
            other = diag.posthoc_x_only_subset(self.directory / 'product', self.document['selected_rows'], self.cfg)
        finally:
            latent.finite_shot_candidate_jacobians = original
        if any(not np.array_equal(table[k], other[k]) for k in ('x', 'mask', 'profiles', 'distance', 'gate')):
            raise ValueError('independent extraction parity failure')
        pooled, candidates = self.infer(table)
        frozen = np.asarray([e['frozen_logits'] for e in self.document['donor_examples']])
        check = diag.logits_compare(frozen, pooled)
        if check['argmax_mismatch_count'] or check['max_abs_diff'] > 1e-4:
            raise ValueError('portable replay execution gate failed: ' + json.dumps(check))
        self.verify_unchanged()
        return cal, table, pooled, candidates, check


def environment():
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_runtime()
    return {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
            'platform': platform.platform(), 'numpy_runtime': output.getvalue(), 'device': 'cpu'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('replay',), required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--archive-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    replay = PortableReplay(args.evidence, args.archive_manifest_sha256)
    # Any attempt to regenerate evidence in replay mode is a hard failure.
    original = np.random.default_rng
    np.random.default_rng = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('sampler forbidden in replay mode'))
    try:
        _, _, pooled, _, check = replay.replay()
    finally:
        np.random.default_rng = original
    result = {'mode': 'recorded_count_replay', 'comparison': check, 'pooled_logits': pooled.tolist(),
              'environment': environment(), 'sampler_calls': 0, 'exact_donor_jacobian_roots': len(replay.bundle.roots)}
    args.output.write_bytes(canonical(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
