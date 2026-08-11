# Step 4 v1 result — audit_066fbee3308c7e13c3308f17

Frozen decision: `DEPLOYABLE_CONTRACT_CORE_IDENTIFIABILITY_UNPROVEN`.

## What passed

The paired-reference hardware acquisition contract is valid. The proposed deployable inputs contain no privileged simulator-only quantities, and the primary reference kind is `paired_hardware_compatible_reference`.

Across 16,006 effectful mechanism pairs, the one-local X/Y/Z Pauli expectation-delta core achieved strong-pair fraction `0.959890` with 95% clean-circuit bootstrap CI `[0.933122, 0.982056]`. The effectful nonterminal + non-q0 subset achieved `0.964672` across 8,322 pairs.

## Why the frozen core did not pass

The GHZ family is a severe eligible failure:

- effectful GHZ pairs: `362`;
- strong-pair fraction: `0.082873`;
- numerical-collision fraction: `0.917127`;
- median local-only pair separation: approximately `1.44e-18`.

The uploaded Step 3.5 full-evidence archive shows that the same effectful GHZ mechanism pairs were strongly separated when full basis distributions were available. Representative medians by pair type were approximately:

- RX vs RY: full score `1.87e-3`, with local expectation components near numerical zero;
- RZ vs RX: full score `1.87e-3`, with local expectation components near numerical zero;
- RZ vs RY: full score `1.87e-3`, with local expectation components near numerical zero.

This localizes the missing signal to joint bitstring correlations rather than invalidating the paired-reference acquisition contract.

## Consequence

Step 5 may not treat one-local Pauli expectation deltas as a sufficient universal hardware-facing core. Step 4.1 therefore freezes a same-shot correlation-recovery ladder using global basis parity and same-basis two-body correlations, with zero additional basis programs beyond the paired Z/X/Y acquisition already required.

Historical v0.1 test accessed: NO.
Spent confirmatory cohort accessed: NO.
Hardware executed: NO.
Classifier trained: NO.
