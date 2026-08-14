# Step 6A — linear cheap-baseline benchmark result

Status: **BASELINE_BENCHMARK_COMPLETE**

This document archives the first controlled Step 6 baseline run on the frozen Step 5 v3 5,000-root development cohort. It is a development benchmark, not a confirmatory test and not a TriQTO architecture result.

## Identity

- benchmark ID: `benchmark_383d4c3070350f0bef6fdb23`
- source product: `product_b2d78ad2309b71a55f9bb54f`
- source clean roots: 5,000
- source examples: 65,000
- validation examples: 13,000
- classifier: class-balanced ridge least-squares
- config SHA-256: `sha256:c16b1940f0d4160b3525a2e3c751044c5325f718d4007dab310133933822a8fe`
- runner SHA-256: `sha256:43d726555ea1a745566564c856bd8d697df986d0fd06b1f5f2455ea9f5a77864`
- uploaded result ZIP SHA-256: `sha256:f8e8bc290d28f35c4207f5ab9a4d920fa3d3041a659234bf54cf7eb1b6e0b80c`

All seven result-file hashes recorded by `benchmark_complete.json` were independently verified against the uploaded bundle.

## Primary linear results

### Effect detection

- chance / majority balanced accuracy: `0.5000`
- `context_only`: BA `0.4977`
- `graph_stats_only`: BA `0.5859`, macro-F1 `0.5140`, AUROC `0.6258`
- `diag_local`: BA `0.5214`
- `diag_local_pairwise`: BA `0.5178`
- `diag_full`: BA `0.5139`, macro-F1 `0.4509`, AUROC `0.5264`
- `diag_full_graph`: BA `0.5931`
- `diag_full_context_graph`: BA `0.5951`, macro-F1 `0.5236`, effect/no-effect recalls `0.5952/0.5951`
- privileged `exact_diag_full_oracle`: BA `0.5847`
- privileged `family_oracle`: BA `0.6166`

The low linear `diag_full` effect score must not be interpreted as absence of effect information: `effect_present` is fundamentally magnitude-like, while this first classifier uses signed coordinates linearly. The exact-diagnostic linear oracle itself reaches only BA `0.5847`, motivating a pre-architecture nonlinear closure benchmark.

### Mechanism diagnosis

Population: validation examples with `mechanism_loss_mask=true`, `n=10,733`.

- chance BA: `0.3333`
- `context_only`: BA `0.3339`
- `graph_stats_only`: BA `0.3665`
- `diag_local`: BA `0.3842`
- `diag_local_pairwise`: BA `0.3910`
- `diag_full`: BA `0.3937`, macro-F1 `0.3931`, macro OVR AUROC `0.5639`
- `diag_full_graph`: BA `0.4037`
- `diag_full_context_graph`: BA `0.4026`, macro-F1 `0.4025`, macro OVR AUROC `0.5741`
- privileged `exact_diag_full_oracle`: BA `0.4855`, macro-F1 `0.4859`, macro OVR AUROC `0.6579`
- privileged `family_oracle`: BA `0.3769`

For `diag_full`, mechanism recalls are RX `0.3151`, RY `0.4550`, RZ `0.4109`. For `diag_full_context_graph`, they are RX `0.3575`, RY `0.4338`, RZ `0.4166`.

## Paired clean-root bootstrap findings

### Pairwise / parity

Under this linear model, pairwise and parity additions do **not** show clean balanced-accuracy gains:

- `diag_local_pairwise - diag_local`, mechanism BA: `+0.0068`, 95% CI `[-0.0006, +0.0143]`
- `diag_full - diag_local_pairwise`, mechanism BA: `+0.0026`, 95% CI `[-0.0031, +0.0082]`
- corresponding effect-detection BA differences are slightly negative with CIs crossing zero.

This does not overturn Step 4.1 identifiability. It says only that a globally linear finite-shot classifier does not exploit those additional correlation blocks reliably.

### Simple graph context

Adding simple graph statistics to finite-shot `diag_full` gives a reproducible mechanism gain:

- `diag_full_graph - diag_full`, mechanism BA: `+0.0101`, 95% CI `[+0.00295, +0.01678]`
- macro-F1: `+0.0105`, 95% CI `[+0.00330, +0.01721]`

For effect detection, the graph addition is much larger:

- BA: `+0.0789`, 95% CI `[+0.0665, +0.0906]`

This is evidence that circuit structure/context is useful for interpreting the diagnostic evidence, but it is not yet evidence that a graph neural network is necessary.

### Exact-vs-finite diagnostic gap

The privileged exact-diagnostic oracle substantially exceeds finite-shot `diag_full` for mechanism diagnosis:

- BA difference: `+0.0918`, 95% CI `[+0.0794, +0.1038]`
- macro-F1 difference: `+0.0928`, 95% CI `[+0.0802, +0.1050]`

This is strong development evidence that finite-shot noise is a major bottleneck.

## Strata

For finite-shot `diag_full` mechanism BA:

- family: Bell `0.349`, GHZ `0.364`, HEA `0.340`, phase-interference `0.333`, QAOA-like `0.470`, QFT-like `0.551`, random-shallow `0.362`
- qubits: 2q `0.420`, 3q `0.395`, 4q `0.397`, 6q `0.372`, 8q `0.377`
- shots: 512 `0.382`, 1024 `0.380`, 2048 `0.396`, 4096 `0.417`
- strength: 0.05 `0.371`, 0.15 `0.417`
- depth: early `0.394`, middle `0.385`, late `0.382`, terminal `0.414`

The shot and strength trends are consistent with a signal-to-shot-noise limitation. Family heterogeneity remains large.

## Scientific decision

Step 6A establishes that:

1. finite-shot B-delta contains above-chance mechanism information under a simple linear classifier;
2. simple graph statistics add reproducible mechanism value;
3. the exact-diagnostic gap is large, so shot noise remains a major limitation;
4. pairwise/parity blocks are not cleanly exploited by this linear classifier;
5. a linear benchmark alone is insufficient to decide what Step 7 architecture is needed.

Therefore Step 6 is **not closed after 6A**. A narrowly scoped Step 6B nonlinear sanity closure is required before any TriQTO GNN training, so later gains cannot be attributed merely to generic nonlinearity.

Boundaries: historical v0.1 test accessed **NO**; spent confirmatory cohort accessed **NO**; hardware executed **NO**; TriQTO architecture changed **NO**.