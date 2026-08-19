# Step 9H Representation-Contract Ablation — Frozen Result

Status: **FROZEN AFTER STEP-9H OUTCOME**

## Returned artifact integrity

- `representation_contract_ablation_v1.json`
  - SHA-256: `bf4a4bc4e26a280f1899c08e93f3bf37c6d974a68a24800cc5f36b6c5795ba53`
- `representation_contract_ablation_v1.log`
  - SHA-256: `4adeeb11ff1bc97cb8106ab624d8120214540eb8f4c76c256179cceb3b019e81`

The uploaded ZIP was checked after the run; both files match the terminal-reported hashes exactly.

## Frozen result

Bridge split: 192 train / 48 validation contexts.

### R0 — current Step-7 graph information

- feature dimension: 326
- Euclidean balanced accuracy: 0.9583333333333334
- cluster-bootstrap 95% CI: [0.9166666666666666, 0.9930555555555555]
- minimum mechanism recall: 0.9375
- inherited gate: PASS
- frozen Step-9D phase QPU: 3/3 correct
- predicted-frame median cosine to exact simulator frame: 1.0000
- full targeted pass: true

### R1 — R0 plus gate-level phasors

- feature dimension: 374
- Euclidean balanced accuracy: 0.9652777777777777
- cluster-bootstrap 95% CI: [0.9305555555555557, 0.9930555555555555]
- minimum mechanism recall: 0.9375
- inherited gate: PASS
- frozen Step-9D phase QPU: 3/3 correct
- predicted-frame median cosine to exact simulator frame: 1.0000
- full targeted pass: true

### R2 — R1 plus candidate local context

- feature dimension: 411
- Euclidean balanced accuracy: 0.9652777777777777
- cluster-bootstrap 95% CI: [0.9375, 0.9930555555555555]
- minimum mechanism recall: 0.9375
- inherited gate: PASS
- frozen Step-9D phase QPU: 3/3 correct
- predicted-frame median cosine to exact simulator frame: 1.0000
- full targeted pass: true

## Decision

`CURRENT_STEP7_GRAPH_INFORMATION_IS_SUFFICIENT__COVERAGE_IS_PRIMARY_GAP`

The richer R1/R2 contracts provide only a small bridge-validation improvement over R0 and are not required for the frozen gate or QPU 3/3 result. In this targeted audit, no graph-information contract change is demonstrated necessary.

The preferred next experiment is therefore a leakage-safe simulator training-distribution redesign that adds deployment-domain coverage while preserving the existing Step-7 input contract, followed by warm-start reuse of the frozen checkpoint as the first training run. A scratch run should be treated as a controlled comparison, not the default replacement.

## Scientific qualification

This result is targeted. It demonstrates sufficiency of the current Step-7 graph information for the audited phase-mechanism bridge and frozen three-case QPU pilot. It does **not** establish universal sufficiency across all circuit families, all distortion classes, or all hardware regimes.

Step 9G also showed that a naive single combined RBF fit did not preserve original Step-5 validation performance. Therefore the next dataset/training design must explicitly preserve legacy-domain validation and must not claim success from deployment-domain improvement alone.
