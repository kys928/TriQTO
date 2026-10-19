#!/usr/bin/env python3
"""Small same-process reproducer from original binary evidence, no simulator."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from portable_calibration import CalibrationBundle, canonical, sha, probability_difference


def sequential_binomials(p, shots, seed):
    """Mirror NumPy 2.3.2 random_multinomial's conditional-binomial loop."""
    rng = np.random.Generator(np.random.PCG64(seed))
    remaining_p, n = 1.0, int(shots)
    counts, trace = np.zeros(len(p), dtype=np.int64), []
    for i in range(len(p) - 1):
        q = float(p[i]) / remaining_p
        before = copy.deepcopy(rng.bit_generator.state)
        value = int(rng.binomial(n, q))
        trace.append({'category': i, 'remaining_shots': n, 'conditional_probability': q,
                      'conditional_probability_hex': q.hex(), 'remaining_mass_hex': remaining_p.hex(),
                      'direct_branch_p_le_half': q <= 0.5, 'count': value,
                      'rng_state_sha256': sha(canonical(before)), 'rng_state': before})
        counts[i] = value
        n -= value
        if n <= 0:
            break
        remaining_p -= float(p[i])
    if n > 0:
        counts[-1] = n
    return counts, trace


def reproduce(left, right, output):
    if np.__version__ != '2.3.2':
        raise RuntimeError('historical sampler reproduction requires NumPy 2.3.2')
    if left.document['provenance'] != right.document['provenance'] or left.roots != right.roots:
        raise ValueError('scientific or sampling provenance mismatch')
    first = None
    for root in sorted(left.roots, key=int):
        a, b = left.arrays['c' + root], right.arrays['c' + root]
        changed = np.flatnonzero(np.any(a != b, axis=1))
        if len(changed):
            first = root, int(changed[0])
            break
    if first is None:
        raise ValueError('no captured count divergence')
    root, ordinal = first
    call = left.roots[root]['calls'][ordinal]
    traces, probabilities, counts = [], [], []
    for bundle in (left, right):
        p = bundle.arrays['p' + root][ordinal]
        expected = bundle.arrays['c' + root][ordinal]
        observed = np.random.Generator(np.random.PCG64(call['seed'])).multinomial(call['shots'], p)
        sequential, trace = sequential_binomials(p, call['shots'], call['seed'])
        if not np.array_equal(expected, observed) or not np.array_equal(sequential, observed):
            raise ValueError('same-process captured count or conditional-loop reproduction failed')
        probabilities.append(p)
        counts.append(observed)
        traces.append(trace)
    divergence = next(i for i, (a, b) in enumerate(zip(*traces)) if a['count'] != b['count'])
    a, b = traces[0][divergence], traces[1][divergence]
    branch_crossed = a['direct_branch_p_le_half'] != b['direct_branch_p_le_half']
    same_rng = a['rng_state_sha256'] == b['rng_state_sha256']
    controls = []
    if branch_crossed and same_rng and a['remaining_shots'] == b['remaining_shots']:
        for q in (np.nextafter(0.5, 0.0), 0.5, np.nextafter(0.5, 1.0)):
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = copy.deepcopy(a['rng_state'])
            observed = rng.multinomial(a['remaining_shots'], [q, 1.0 - q])
            controls.append({'conditional_probability_hex': float(q).hex(), 'counts': observed.tolist()})
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'first_divergent_call.npz', probabilities_left=probabilities[0],
                       probabilities_right=probabilities[1], counts_left=counts[0], counts_right=counts[1])
    result = {'schema': 'triqto.multinomial_discontinuity_reproducer.v1', 'numpy': np.__version__,
              'bit_generator': 'PCG64', 'root': root, 'call': call,
              'first_different_category': divergence, 'same_rng_state_before_first_difference': same_rng,
              'half_probability_branch_crossed': branch_crossed,
              'first_difference_left': a, 'first_difference_right': b,
              'half_boundary_binary_controls': controls,
              'numeric_comparison': probability_difference(*probabilities),
              'counts_left': counts[0].tolist(), 'counts_right': counts[1].tolist(),
              'both_historical_draws_reproduced_in_one_process': True,
              'both_conditional_binomial_loops_exact': True,
              'fixture_sha256': sha((output / 'first_divergent_call.npz').read_bytes()),
              'numpy_source': 'https://github.com/numpy/numpy/blob/v2.3.2/numpy/random/src/distributions/distributions.c',
              'source_interpretation': 'random_multinomial uses conditional random_binomial; random_binomial selects direct/complement at p <= 0.5. No modified sampler used in pilot.'}
    (output / 'result.json').write_bytes(canonical(result))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--left-manifest-sha256', required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--right-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reproduce(CalibrationBundle(a.left, a.left_manifest_sha256),
              CalibrationBundle(a.right, a.right_manifest_sha256), a.output)
