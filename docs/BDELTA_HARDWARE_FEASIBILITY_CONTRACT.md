# Step 4 — B-delta hardware-feasibility contract audit

## Purpose

Step 3 and Step 3.5 established that exact clean-relative Z/X/Y `B_delta` evidence is highly informative for matched RZ/RX/RY mechanism separation in noiseless simulation. Step 4 does **not** execute IBM hardware. It asks a narrower design question before Step 5 training data are generated:

> Which parts of `B_delta` can legitimately exist in hardware mode, what extra diagnostic executions do they require, what reference semantics are required, and which quantities must remain privileged simulator-only supervision?

The audit also reuses the completed Step 3.5 pairwise evidence to test a hardware-scalable local-Pauli core before approving it as a mandatory model input.

## Current IBM/Qiskit measurement facts used by the contract

The contract is grounded in current IBM Quantum/Qiskit primitive semantics:

- Runtime `SamplerV2` returns sampled bitstrings/results from submitted measured circuits.
- IBM/Qiskit computational readout is in the Z basis.
- X-basis measurement is obtained by applying `H` before Z measurement.
- Y-basis measurement is obtained by applying `Sdg` then `H` before Z measurement.
- Runtime `EstimatorV2` can estimate expectation values for Pauli observables directly; equivalently, for the TriQTO diagnostic bundle, local X/Y/Z expectation values can be derived from the same basis-specific Sampler bitstrings used for distribution evidence.

Primary source pages:

- https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2
- https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/estimator-v2
- https://qiskit.qotlabs.org/docs/guides/specify-observables-pauli

These facts establish **measurement possibility**, not statistical sufficiency or QPU-cost acceptability.

## Three basis programs

For one circuit state:

```text
Z evidence: circuit -> measure
X evidence: circuit -> H on measured qubits -> measure
Y evidence: circuit -> Sdg, H on measured qubits -> measure
```

The primary Step 5 hardware-compatible reference mode is paired acquisition. Therefore one logical observed/reference pair requires six basis-program variants before shot replication:

```text
observed/current: Z, X, Y
paired reference: Z, X, Y
```

The paired reference is not assumed to be noise-free. It is simply acquired under a contract intended to make the subtraction meaningful: same backend, same physical layout, same basis policy, and a bounded calibration/time relationship.

## The clean-reference problem

`B_delta` is relational. It cannot exist without defining the reference side.

Step 4 therefore forbids an implicit or unlabeled notion of "clean". Every Step 5 example must carry `reference_kind` and reference metadata.

### Primary Step 5 reference

`paired_hardware_compatible_reference`

This means the intended reference circuit and the observed/current circuit are paired under the same hardware context. In simulator development this can be emulated exactly; in later hardware studies both sides would be empirical noisy executions.

### Other reference kinds

- `ideal_simulator_reference`: useful simulation side information/ablation, but not hardware-derived and therefore never allowed to masquerade as hardware mode.
- `rolling_hardware_reference`: physically possible but vulnerable to calibration/drift confounding; future ablation only.
- `fixed_historical_reference`: disallowed as a primary contract because a stale baseline can encode backend drift rather than the mechanism of interest.

## Feature classification

### Primary scalable hardware core

The proposed mandatory core is the signed local expectation-delta vector:

```text
Delta<X_0 ... X_n-1>
Delta<Y_0 ... Y_n-1>
Delta<Z_0 ... Z_n-1>
```

These are empirical finite-shot quantities and scale as `O(n)` in output width. They retain explicit basis identity and must also carry observed/reference shot counts and masks.

### Optional full `B_delta`

The complete full-register probability deltas remain useful for exact simulation and small-qubit diagnostics:

```text
Delta p_X(bitstring)
Delta p_Y(bitstring)
Delta p_Z(bitstring)
```

They are physically sampleable with Sampler, but their outcome support is `2^n`. Step 4 therefore does not allow dense full-register distributions to become a mandatory scalable hardware interface.

They may be retained in Step 5 as an explicitly masked `OPTIONAL_SMALL_N_FULL_BDELTA` stream for controlled ablations.

### Privileged simulator-only quantities

The following may be used for supervision/audit but never as deployable model inputs:

- statevector;
- exact Hilbert overlap;
- population component;
- phase component;
- exact effect-present/negligible label derived from privileged state comparison;
- other exact-state phenomenology targets.

This preserves the Step 3.5 lesson: mechanism and phenomenology remain separate, and negligible injected perturbations should mask mechanism supervision rather than teach the model that zero evidence implies a particular mechanism.

## Why Step 4 includes a scalable-core ablation

Physical measurability is not enough.

Step 3.5's separation score used both full-register distribution geometry and local expectation geometry. Before declaring local Pauli deltas the hardware core, Step 4 re-scores the already frozen 20,130 Step 3.5 mechanism pairs using only local expectation differences.

For each counterfactual:

```text
signal_core = mean(
    0.5 * RMS(Delta<X_i>),
    0.5 * RMS(Delta<Y_i>),
    0.5 * RMS(Delta<Z_i>)
)
```

For each mechanism pair:

```text
pair_core = mean(
    0.5 * RMS(Delta<X_i>_left - Delta<X_i>_right),
    0.5 * RMS(Delta<Y_i>_left - Delta<Y_i>_right),
    0.5 * RMS(Delta<Z_i>_left - Delta<Z_i>_right)
)
```

The same frozen Step 3 raw and relative separation thresholds are retained:

- raw separation >= `1e-6`;
- relative separation >= `0.25`;
- numerical collision <= `1e-10`.

The primary population excludes pairs where either mechanism is privileged-ground-truth `negligible`. Those examples remain valuable null/effect supervision, but forcing mechanism identifiability when the intervention produced no meaningful state effect would answer the wrong question.

The scalable core passes only if:

- effectful overall strong-pair fraction >= `0.90`;
- effectful nonterminal + non-q0 strong-pair fraction >= `0.90`;
- no eligible stratum with at least 100 pairs has strong-pair fraction below `0.50`.

Bootstrap uncertainty is grouped by independent clean circuit, not factorial derivative.

## Possible decisions

### `DEPLOYABLE_WITH_PAIRED_REFERENCE_CORE`

The acquisition/reference contract is hardware-valid and the local-Pauli core retains the frozen identifiability requirements without severe context failure.

### `DEPLOYABLE_CONTRACT_CORE_IDENTIFIABILITY_UNPROVEN`

The measurement/reference contract is physically legitimate, but local Pauli deltas alone lose important mechanism information in at least one frozen criterion. Step 5 must retain additional **hardware-valid bounded evidence** rather than pretending the local core is sufficient.

### `NOT_DEPLOYABLE_AS_DEFINED`

A mandatory input requires simulator-only privileged information or the reference semantics are undefined/not hardware-valid.

## Architecture boundary

Step 4 changes no model architecture.

However, it freezes the future interface direction:

```text
DiagnosticTensorBatch
        ->
DiagnosticEncoder
```

`B_delta` must not be forced into the existing `BornTensorBatch` probability contract because:

- `B_delta` is signed;
- it is relational;
- basis identity is semantically essential;
- observed/reference shot uncertainty matters;
- optional full-distribution evidence needs an explicit mask.

The Step 7 adapter should therefore be a dedicated diagnostic stream fused with the circuit graph.

## Step 5 fields frozen by this contract

At minimum:

- `diagnostic_basis_code`;
- `reference_kind`;
- `reference_available_mask`;
- `observed_shots`;
- `reference_shots`;
- `delta_local_expectations`;
- `delta_local_expectations_mask`;
- optional sparse/full distribution evidence plus mask;
- `effect_present_target`;
- `mechanism_target`;
- `mechanism_loss_mask`;
- phenomenology targets;
- `clean_circuit_group_id`.

## Boundaries

Step 4 establishes an acquisition/data contract only.

It does not establish:

- finite-shot robustness;
- optimal shot allocation;
- noise robustness;
- calibration robustness;
- QPU utility;
- real-hardware diagnosis accuracy.

Those remain later empirical stages.

## Run

```bash
cd /workspace/triqto

PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_bdelta_hardware_feasibility_contract_audit.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_bdelta_hardware_feasibility_contract.py
```

Outputs are written under:

`/workspace/triqto-data/step4_bdelta_hardware_feasibility_contract/audit_*`
