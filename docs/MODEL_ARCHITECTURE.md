# Phase 13 TriQTO Model Architecture

Phase 13 implements the untrained neural architecture that consumes the leakage-safe Phase 12 contracts. It defines forward computation, tensor validation, stream masking, model identities, output heads, and future loss primitives. It does **not** implement optimization, epochs, gradient steps, checkpoint scheduling, learned correction, evaluation, or hardware execution. Those remain later phases.

## Architectural position

```text
Phase 7/8/9/11 data
        ↓
Phase 12 task-specific views and per-head masks
        ↓
Phase 13 TriQTO architecture
        ↓
Phase 14 training engine
        ↓
Phase 15 evaluation
        ↓
Phase 16 hardware validation
```

The model is a classical PyTorch graph architecture designed around quantum-circuit structure. It is not a quantum neural network running on a QPU, and it does not claim quantum advantage.

## Core data flow

```text
variable-size circuit graph ───────────────┐
ragged circuit parameters ────────────────┤
sine/cosine phasors ──────────────────────┤
optional simulator Hilbert state ─────────┤
Born distribution evidence ───────────────┤→ mask-aware dual-mode fusion
optional backend metadata ────────────────┤          ↓
optional persistent-topology features ────┘  head-specific latent states
                                                     ↓
 diagnosis | action ranking | Born prediction | Hilbert deformation
 uncertainty | topology audit projection
```

The architecture follows the TriQTO chain:

```text
parameter manifold → Hilbert-state manifold → Born-probability manifold
```

The graph stream is the structural anchor. Parameter, phasor, Hilbert, Born, backend, and topology streams are optional and explicitly masked.

## Variable-size graph encoder

A batch concatenates multiple circuit graphs without imposing a fixed qubit count.

- logical qubits are nodes;
- directed interaction events are multiedges;
- gates are ordered event records;
- gate-to-qubit incidence is represented by CSR-style pointers;
- `node_batch` and `gate_batch` identify graph ownership;
- graph pooling is permutation-invariant mean/max pooling.

The input contract validates feature widths against the Phase 8 graph schema, rejects cross-graph edges, checks gate incidence ownership, and requires at least one node per graph. No global 4-, 8-, or 12-qubit padding shape is built into the model.

## Phase-coupled lattice interaction

TriQTO does not use transformer query/key/value attention. Directed messages are generated from a source node, edge features, and the associated gate event. Two learned message channels are mixed through learned phase quadratures:

```text
m_ij = m_in-phase(i,j) · cos(φ_ij) + m_quadrature(i,j) · sin(φ_ij)
```

where `φ_ij` is a bounded learned phase field derived from edge and gate context. Destination nodes receive normalized sums of incoming messages followed by a residual update and layer normalization.

This is a complex-inspired classical graph operation. The sine/cosine coupling preserves an explicit phasor language, but it is not itself a quantum operation.

## Input streams

### Circuit graph

The mandatory stream encodes node, edge, and ordered gate-event structure through shared projections and stacked phase-coupled message passing.

### Parameter manifold

Ragged parameter rows contain raw values plus sine and cosine encodings. Shared element encoders and segment pooling produce one embedding per circuit.

### Phasor stream

The phasor encoder emphasizes periodic coordinates, including sine, cosine, raw angle context, and unit-circle consistency. It remains separate from the generic parameter encoder so later ablations can test whether explicit periodic structure helps.

### Born stream

Born evidence is represented as a variable-support table of outcome strings, probabilities, per-qubit `Z/X/Y` basis codes, and measurement-setting indices. Outcome and measurement-basis values are encoded jointly with shared bit-position encoding and per-row active-qubit normalization, so representation does not depend on the widest item in a mixed batch. Outcome embeddings are pooled with probability-weighted, mean, and maximum summaries.

The Born-prediction head is hard-forbidden from consuming the Born input stream. Phase 12 also physically excludes target Born evidence from Born-prediction inputs.

### Hilbert stream

Pure-state amplitudes are represented as real/imaginary pairs over variable-width basis strings. Before encoding, every state is rotated by a deterministic reference amplitude so the representation is invariant to global phase. Relative phase, magnitude, and basis context remain available.

Hilbert tensors are privileged simulation inputs. Hardware-mode rows reject them before the forward pass. The Hilbert-deformation head is also hard-forbidden from directly consuming the Hilbert stream, preventing a trivial copy shortcut.

### Backend stream

A versioned dense backend vector can contain future hardware/calibration metadata. Rows marked unavailable must be exactly zero. Current simulator-only Phase 12 items normally mark this stream unavailable.

### Topology stream

A dense topology vector can carry Phase 11 persistent-homology and cross-manifold alignment summaries. Topology may influence diagnosis, action, uncertainty, and other allowed heads when the task mask permits it.

The topology prediction head is hard-forbidden from directly consuming topology input. It predicts an audit projection from other streams, and its supervised target mask remains false in Phase 13.

## Strict masking model

Masking is enforced at three levels:

1. **Stream availability** records whether data exists for a graph.
2. **Hard head-stream policy** prevents known target-copy shortcuts.
3. **Runtime per-head masks** carry the Phase 12 task-specific permissions.

A separate `head_active_mask` marks which heads are meaningful for each row. Active heads must retain at least one permitted stream. Inactive heads may have no streams; their latent states, fusion weights, and head outputs are forced to zero.

Runtime masks may remove a stream but can never enable a stream forbidden by the hard architecture policy.

Dense unavailable rows must be exactly zero, not merely ignored after encoding. Hardware rows reject Hilbert tensors and reject topology features known to have been computed with Hilbert access.

## Dual simulation/hardware mode

Every graph has a mode flag:

- `false`: simulation-compatible mode;
- `true`: hardware or hardware-masked mode.

The mode is embedded into fusion. More importantly, it activates hard validation:

- no Hilbert rows in hardware mode;
- no Hilbert-dependent topology in hardware mode;
- unavailable streams remain zero;
- output metadata does not reinterpret masked simulation as real hardware evidence.

The architecture can therefore share weights across both modes without pretending the information sets are identical.

## Mask-aware fusion

For each active head, available streams receive learned scalar gates. The fusion block combines:

- gated weighted sum;
- mean of available streams;
- an embedding of the availability pattern;
- simulation/hardware mode embedding.

This is not Q/K/V attention. The model exposes fusion weights for later interpretability audits, while unavailable streams receive exactly zero weight.

## Output heads

### Distortion diagnosis

Produces:

- coarse distortion-family logits;
- distortion-strength mean and log scale;
- one affected-qubit logit per graph node.

Raw Phase 12 distortion names must be mapped to the versioned coarse label vocabulary by the future Phase 14 data adapter. That mapping must be explicit and tested rather than guessed inside the model.

### Action ranking

Scores a variable number of candidates per graph. Candidate inputs include safe Phase 12 features and ragged edit descriptions. Scores are normalized with segment softmax independently per graph. Inactive or unavailable candidates receive zero probability.

Privileged oracle labels are not observable model inputs. Their masks remain target/provenance information for later training and evaluation policy.

### Born prediction

Scores an arbitrary queried support of basis strings conditioned on `M` and applies segment softmax independently per measurement setting. It does not assume a fixed `2^n` output tensor, although a caller may query the full support for small circuits.

### Hilbert deformation

Predicts a bounded-dimensional hidden-state deformation summary and uncertainty scale from observable/non-Hilbert streams. It does not reconstruct a full exponentially large statevector.

### Uncertainty

Predicts heteroscedastic log-variance summaries for the main task families. Calibration and use in decisions remain Phase 14–15 responsibilities.

### Topology audit projection

Predicts a bounded topology feature vector and confidence from non-topology streams. The supervised topology target is unavailable and `lambda_top` is exactly zero in Phase 13.

## Tensor contracts

The public dataclasses validate:

- dtype, rank, finite values, and common device;
- graph ownership and index bounds;
- CSR pointer monotonicity;
- probability and state normalization;
- exact stream availability coverage;
- zero-valued unavailable dense rows;
- action vocabulary and candidate/edit joins;
- basis bits and masks;
- hardware/Hilbert leakage;
- per-head stream and activation masks.

These contracts are intentionally strict. Silent coercion, implicit truncation, and automatic target inclusion are rejected.

## Identity and initialization

`model_schema_id` versions the tensor contract, stream/head order, hard mask policy, phase-coupled layer, fusion rule, output contract, global-phase behavior, and inactive-output semantics.

`model_architecture_id` depends only on architecture-defining scientific fields. It excludes the human-readable model name and initialization seed.

`model_config_id` identifies the complete constructor configuration, including name and initialization seed. The architecture manifest additionally records an exact initialized-state signature. Deterministic initialization runs inside a forked RNG context and does not mutate the caller's global CPU RNG state.

The Phase 13 architecture manifest explicitly records:

```text
trained = false
optimizer_state_present = false
training_checkpoint = false
topology_loss_weight = 0.0
```

## Loss primitives and Phase 14 boundary

Phase 13 includes transparent, mask-aware loss primitives so the output contract is testable:

- diagnosis classification/regression/localization;
- listwise action ranking;
- Born-distribution divergence;
- geometry consistency;
- multitask composition;
- topology discrepancy with an enforced zero multiplier.

There is no trainer, optimizer, scheduler, epoch loop, gradient clipping, mixed precision policy, checkpoint resume, or dataset adapter in Phase 13. Phase 14 must build those components without bypassing the Phase 12 per-head masks.

## Known limitations

- The model is untrained; forward outputs have no empirical meaning yet.
- The Hilbert encoder currently supports pure states, not density matrices or general channels.
- Full-support Born prediction is exponentially expensive for large qubit counts; sparse/query support is the intended scalable interface.
- Topology vectors need a versioned feature-map adapter before training.
- Backend features are structurally supported but current ideal datasets contain no real calibration stream.
- Topology-only audit items do not contain a circuit graph and therefore are consumed by audit tooling or a future adapter, not by the graph-anchored full model in isolation.
- No generalization, correction success, hardware performance, or quantum advantage has been demonstrated.
