# Step 5 500-root v2 full-artifact EDA — rejected for training

Product: `product_025667cdd6191f70d4a77957`

Decision: `REJECT_FOR_TRAINING_REGENERATE_V3`

## What passed

The v2 repairs worked exactly as intended:

- 500 independent clean roots, 6,500 artifacts;
- 400 train / 100 validation roots;
- family/split Cramer's V = `0.000000`;
- every family has exact 80/20 train/validation coverage;
- depth/strength Cramer's V = `0.012000` overall (`0.000000` train, `0.060000` validation);
- 88 independent 3-qubit roots: 72 train / 16 validation;
- 500 clean controls and 2,000 examples per injected RZ/RX/RY mechanism;
- no clean-group cross-split leakage and no duplicate clean graphs.

All uploaded artifacts were scanned independently:

- 6,500 / 6,500 artifact SHA-256 values matched the manifest;
- all NPZs loaded with `allow_pickle=False`;
- exactly one NPZ array schema across all artifacts;
- zero example-ID, clean-group, mechanism, phenomenology, effect-target, mask, affected-qubit, insertion-boundary, strength, or shot-count mismatches;
- zero forbidden label/audit tokens in `x__` keys;
- all empirical and exact diagnostic arrays were finite;
- maximum absolute empirical deltas: local `0.2890625`, pairwise `0.3046875`, global parity `0.25390625`;
- maximum absolute exact deltas were approximately `0.15`.

Train/validation target drift was small after the family split repair:

- injected effect-present rate: train `0.89375`, validation `0.890833`;
- phenomenology train/validation total-variation distance: approximately `0.01104`;
- family train/validation TV distance: `0.0`;
- qubit-count train/validation TV distance: `0.08`.

## New blocker found by the deeper EDA

The finite-shot schedule still reused the same modular expression as the hidden affected-qubit schedule:

- affected qubit: `(root_index + context_index) mod n_qubits`;
- shot setting: `shots_cycle[(root_index + context_index) mod 4]`.

Because shots are an explicit deployable model input, this creates an artificial acquisition-metadata shortcut to hidden perturbation location.

Observed shot/affected-qubit Cramer's V by qubit count:

- 2q: `1.0000`;
- 3q: `0.0265`;
- 4q: `1.0000`;
- 6q: `0.5801`;
- 8q: `1.0000`.

For 4q circuits the mapping is literally deterministic:

- q0 -> 512 shots;
- q1 -> 1024 shots;
- q2 -> 2048 shots;
- q3 -> 4096 shots.

For 8q circuits the shot setting identifies affected qubit modulo four.

Shot count is also moderately associated with hidden strength:

- overall shots/strength Cramer's V = `0.196`;
- weak (`0.05`) examples: 897/603/897/603 across 512/1024/2048/4096 shots;
- strong (`0.15`) examples: 603/897/603/897.

Clean controls have another family-conditioned shot shortcut under the old root-index cycle. In Bell-like roots all clean controls use 512 shots, while intervention examples span all four shot settings. Overall clean-control shot counts are balanced, so this interaction is invisible to a marginal-only gate.

These are synthetic scheduling artifacts, not desired physics. A future model could use explicit shot metadata together with circuit structure to infer hidden context distributions that would not hold on deployment hardware.

## Finite-shot difficulty (report-only)

The hardware-facing evidence remains intentionally difficult. Across injected examples, median empirical-vs-exact core RMS error fell with shots:

- 512: `0.054688`;
- 1024: `0.039233`;
- 2048: `0.027721`;
- 4096: `0.019319`.

Clean-control empirical core RMS similarly decreased from about `0.05628` at 512 shots to `0.01907` at 4096 shots, consistent with shot-noise scaling.

The empirical error exceeds the exact diagnostic signal for many weak perturbations. This remains a report-only scientific difficulty and is not a reason to leak exact simulator evidence into model inputs. Step 6 baselines must test whether the finite-shot signal is learnable.

## V3 repair requirement

V3 must preserve all successful v2 repairs and additionally:

1. independently schedule affected qubits using a frozen root-specific deterministic permutation/cycle;
2. independently schedule intervention shot counts using a separate frozen root-specific permutation of 512/1024/2048/4096;
3. schedule clean-control shots independently of global root position, with balanced within-family coverage;
4. gate shot/strength, shot/affected-qubit, shot/depth, and clean-shot/family associations before publication;
5. rerun the full-artifact EDA before 1,000 roots are unlocked.

V2 remains immutable evidence of a valid artifact contract and successful family/depth repairs, but it is not approved for model training.