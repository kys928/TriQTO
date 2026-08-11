# Step 4.1 — B_delta correlation-recovery result

Frozen decision: `CORRELATION_CORE_RECOVERED`.

Selected smallest passing variant: `local_plus_pairwise_plus_global_parity`.

## Variant ladder

| Variant | Effectful strong | 95% clean-circuit bootstrap CI | Nonterminal + non-q0 strong | GHZ strong | Pass |
|---|---:|---:|---:|---:|---|
| local only | 0.959890 | 0.933122–0.982056 | 0.964672 | 0.082873 | no |
| local + global parity | 0.986755 | 0.976694–0.994742 | 0.984499 | 0.895028 | no |
| local + same-basis pairwise | 0.991378 | 0.982032–0.998055 | 0.991468 | 0.685083 | no |
| local + pairwise + global parity | 0.996439 | 0.993036–0.998794 | 0.994833 | 0.961326 | yes |

The frozen recovery policy required overall effectful strong fraction >= 0.90, effectful nonterminal+non-q0 strong fraction >= 0.90, every eligible major stratum >= 0.80, and GHZ >= 0.90. The first three variants fail because GHZ is the minimum eligible stratum. The fourth variant passes all gates.

## Interpretation

The Step 4 v1 one-body core failed because GHZ mechanism information is encoded in joint correlations that vanish from single-qubit marginals. Global parity alone recovers most of that information but stops just below the frozen GHZ threshold (`0.895028 < 0.90`). Same-basis pairwise correlations alone are insufficient (`0.685083`). Their combination is complementary and restores GHZ to `0.961326` while retaining `0.996439` overall effectful mechanism separation.

The selected representation adds **zero basis programs** beyond the paired Z/X/Y acquisition already required by Step 4 v1. Local expectations, same-basis two-body correlations, and global basis parity are all computed from the same basis-specific bitstrings.

## Step 5 implication

The default hardware-facing diagnostic evidence contract for the Step 5 training dataset should therefore contain, for each of X/Y/Z:

- signed local expectation deltas;
- signed same-basis pairwise correlation deltas;
- signed global parity delta;
- explicit masks for variable qubit count / valid pair entries;
- basis identity, observed/reference shot counts, reference kind, and reference-availability metadata.

Dense full-register probability deltas remain optional small-n audit/ablation evidence, not mandatory model input. Simulator statevectors, phase/population decomposition, and effect/negligible labels remain supervision/audit-only and must never enter deployable inputs.

Because all selected correlation summaries are derived from the same paired Z/X/Y shots, the measurement-program count remains six variants per observed/reference logical pair before shot replication.

## Evidence identity

Uploaded audit ZIP SHA-256: `c24f51b0b8945a13b506f3384f48d3999239f57c4fd8b892623f9fff4e43ec0f`.

File SHA-256 values from the uploaded archive:

- `audit_complete.json`: `e6f5c486ca4bc661a1edbc0c3de7ab7544e346bd67d6654559529f72899f9f37`
- `decision.json`: `8ac17bca5035f01805dadd66f967f267234514c0bb21b89bdf2cf216733d3f81`
- `variant_results.json`: `87cbd1a8259977f914802a7e220c698fa988b2cecb158f2a1642b18f69a9a0f1`
- `correlation_variant_summary.csv`: `33985c0967f2b12a2b14b2073e4ddd0bdc58746356d2cfa24bcd7a69b0c108de`
- `correlation_pair_metrics.csv`: `e91bfaa2fc0f8f5d9ff2a39b20f3a57733933868c00cf5716f713a05639c6403`

Historical v0.1 test accessed: NO.
Spent confirmatory cohort accessed: NO.
Hardware executed: NO.
Classifier trained: NO.
