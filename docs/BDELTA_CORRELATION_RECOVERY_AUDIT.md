# Step 4.1 — bounded correlation recovery for hardware-valid B_delta

## Why this follow-up exists

Step 4 v1 validated the paired-reference hardware acquisition contract, but the proposed one-local Pauli expectation-delta core failed its frozen identifiability gate. Overall effectful separation remained high (`0.9599`), while the GHZ family fell to `0.0829` strong separation across 362 effectful pairs.

The completed Step 3.5 full Z/X/Y distribution evidence had separated those same GHZ mechanisms. The failure therefore localizes the missing information to joint bitstring structure rather than to hardware acquisition itself.

Step 4.1 does not reinterpret or erase the v1 failure. It asks a narrower question:

> What is the smallest bounded correlation summary, derivable from the same paired Z/X/Y shots, that recovers the missing GHZ information?

## Frozen feature ladder

The variants are evaluated in this order and the first passing variant is selected:

1. `local_only`
   - single-qubit X/Y/Z expectation deltas;
   - output width `O(n)`.
2. `local_plus_global_parity`
   - local deltas plus one global parity per basis;
   - output width `O(n) + 3`.
3. `local_plus_pairwise_same_basis`
   - local deltas plus all same-basis two-qubit correlation deltas;
   - output width `O(n^2)`.
4. `local_plus_pairwise_plus_global_parity`
   - local, pairwise, and global parity deltas;
   - output width `O(n^2)`.

All correlation quantities are derived from the exact same sampled bitstrings used by Step 4 v1. No new basis programs are introduced.

For a basis `P in {X,Y,Z}`:

- local: `<P_i>`;
- pairwise: `<P_i P_j>` for every `i < j`;
- global parity: `<P_0 P_1 ... P_(n-1)>`.

The simulator audit computes the exact expectation of those shot-derived statistics. Finite-shot variance is explicitly not established here.

## Frozen selection gates

The primary population contains mechanism pairs where both injected mechanisms have non-negligible privileged simulator effect.

A variant passes only if:

- overall effectful strong-pair fraction >= `0.90`;
- effectful nonterminal + non-q0 strong-pair fraction >= `0.90`;
- every eligible >=100-pair stratum across family, pair type, qubit count, strength, and insertion-depth bin is >= `0.80`;
- GHZ strong-pair fraction >= `0.90`.

The pairwise raw/relative separation thresholds remain `1e-6` and `0.25`, with numerical collision threshold `1e-10`.

## Possible decisions

- `CORRELATION_CORE_RECOVERED`
- `HARDWARE_VALID_CORRELATION_SUMMARIES_INSUFFICIENT`

A positive result approves the smallest passing summary as the Step 5 hardware-facing diagnostic evidence contract. A negative result means Step 5 must retain a richer hardware-valid representation rather than pretending bounded low-order summaries are enough.

## Scientific boundaries

- Step 4 v1 remains frozen as `DEPLOYABLE_CONTRACT_CORE_IDENTIFIABILITY_UNPROVEN`.
- No IBM/QPU execution occurs.
- No classifier is trained.
- No new diagnostic basis executions are added.
- Statevectors remain privileged audit-only regeneration machinery.
- Finite-shot/noisy robustness remains a later empirical gate.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_bdelta_correlation_recovery_audit.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_bdelta_correlation_recovery.py
```

Outputs are written under:

`/workspace/triqto-data/step4_1_bdelta_correlation_recovery/audit_*`
