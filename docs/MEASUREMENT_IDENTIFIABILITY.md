# Measurement Settings and Diagnosis Identifiability

TriQTO Phase 7 schema v2 makes the paper's measurement family `M` an explicit
part of observable evidence.  Diagnosis data is modeled as conditioned evidence
`p(y | M)`, not as an unlabeled computational-basis vector.

## Executable measurement contract

The implemented offline simulator accepts Pauli-product settings over every
active qubit.  Repository defaults generate the uniform settings `Z`, `X`, and
`Y`; mixed product settings such as `ZX` are also valid when their width matches
the circuit.  Each setting has a stable content-derived ID, and every exact
probability or finite-shot record joins to that ID.

For each setting, the simulator removes final measurements, applies the physical
basis rotation, evaluates an ideal statevector, and emits a separately
normalized distribution.  `X` uses `H`; `Y` uses `Sdg` followed by `H`; `Z`
needs no rotation.  Exact probabilities remain simulator privilege.  Finite
counts are sampled from the explicitly identified setting distribution.

The executable readout distortion is an independent symmetric classical bit-flip
channel applied to the measured distribution.  It therefore changes observable
evidence without pretending to be a circuit-unitary error.  The former
`readout_bitflip_marker` configuration name is rejected.  A legacy Python
function alias remains only to fail safely for old callers while producing the
real observable channel.

## Identifiability contract

Every generated sample records one of:

- `identifiable`: every selected setting changes beyond the configured numeric
  tolerance;
- `conditionally_identifiable`: at least one selected setting is informative
  and at least one is blind;
- `unidentifiable`: none of the allowed evidence distinguishes the target.

Non-identifiable records carry a machine-readable reason.  Current reasons are
`marker_only_no_observable_change`,
`computational_basis_phase_blindness`,
`insufficient_measurement_settings`, and
`backend_feature_unavailable`.  Conditional samples use
`requires_selected_measurement_settings`.

The default `unidentifiable_policy: mask` keeps an unidentifiable row for audit
and coverage accounting but sets diagnosis supervision false.  `error` rejects
such generation.  `allow` is an explicit scientific override and is recorded as
`unidentifiable_supervision_override: true`; evaluation still excludes that row
from headline metrics unless a second evaluation-time override is supplied.

Layout permutation markers remain audit-only because no backend coupling,
calibration, transpilation, or routed-layout evidence exists yet.  They cannot
be converted into diagnosis targets by metadata alone.

## Leakage prevention

The diagnosis view uses the programmed clean graph plus distorted measurement
evidence.  It never exposes the synthetically injected error gate as an input.
An observable-evidence fingerprint is computed from only the programmed graph,
measurement settings, and distorted distributions.  Labels, distortion IDs,
paths, seeds, and provenance are excluded.  Dataset construction rejects
conflicting supervised labels that share the same allowed-evidence fingerprint.

Diagnosis class, strength, and affected-qubit masks are false for unidentifiable
rows by default.  Action supervision is also disabled for those rows.  The
evaluation helper reports status/reason counts, default-scored coverage, excluded
count, and explicit override count, and filters unidentifiable rows from
diagnosis metrics and rankings by default.

## Artifact and model migration

Phase 7 writes `measurement_setting_manifest` plus setting-specific probability
and optional shot run records.  Dataset samples carry ordered setting and run
joins.  Phase 8 pair artifacts v2 persist setting IDs, per-qubit basis codes,
row-to-setting indices, and clean/distorted probability tables normalized per
setting.  Phase 12 view artifacts v2 preserve the same context.

The Phase 13 tensor contract v3 requires basis codes and measurement-setting
indices for Born inputs and queries.  The Born encoder jointly embeds outcome
bits, positions, and `Z/X/Y` context.  The prediction head normalizes per
measurement setting, and Phase 14 KL/Hellinger losses average complete distances
per setting rather than combining several normalized distributions as one
graph-level distribution.

Old Phase 7 v1, Phase 8 pair v1, Phase 12 v1, and Phase 13 tensor v2 artifacts do
not contain this evidence and must be regenerated; they are not silently
upgraded.

## Scientific boundary

This implementation supplies ideal simulator `p(y | M)` and an exact classical
readout-confusion channel.  It does not supply noisy Aer circuit execution,
density matrices, fake backends, IBM Runtime data, calibration streams,
transpilation/routing evidence, or hardware identifiability.  Those claims remain
unsupported until their later phases are implemented and tested.
