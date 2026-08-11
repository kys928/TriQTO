# Step 5 v3 — matched diagnostic training dataset

Step 5 v3 is the second regeneration of the 500-root training-data gate.

V1 was rejected for family/split aliasing and perfect depth/strength confounding. V2 repaired both and added 3-qubit circuits, but the full-artifact EDA found a remaining acquisition/context alias because the same `root_index + context_index` modular schedule controlled both shot count and hidden affected qubit.

## V3 changes only scheduling

All successful V2 contracts remain unchanged:

- within-family nested 80/20 train/validation split;
- 2q/3q/4q/6q/8q variable-size clean circuits;
- four early/middle/late/terminal contexts per root;
- deconfounded 0.05/0.15 strength schedule;
- matched RZ/RX/RY mechanisms;
- one clean control per root;
- Step 4.1 local + same-basis pairwise + global-parity finite-shot `B_delta` core;
- hidden perturbation absent from graph input;
- simulator state and phenomenology remain target/audit only.

V3 independently schedules:

1. affected qubits from a root-specific deterministic permutation cycle;
2. intervention shot counts from a separate root-specific deterministic permutation of 512/1024/2048/4096;
3. clean-control shot counts by within-family occurrence modulo four.

The schedule is deterministic and reproducible, but the independent namespaces prevent acquisition metadata from encoding hidden perturbation location.

## Frozen association gates

The 500-root product refuses publication if:

- shots/strength Cramer's V > 0.10;
- shots/depth Cramer's V > 0.10;
- shots/family or shots/split Cramer's V > 0.05;
- clean-control shots/family or shots/split Cramer's V > 0.10;
- within any qubit-count stratum, shots/affected-qubit, strength/affected-qubit or depth/affected-qubit Cramer's V > 0.25;
- any root lacks one of the four intervention shot levels.

These gates are in addition to all V2 family/split, depth/strength, 3q, leakage, graph, target and artifact-integrity gates.

## Run

```bash
PYTHONPATH=/workspace/triqto/src \
pytest -q tests/test_step5_matched_diagnostic_training_dataset_v3.py

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/generate_step5_matched_diagnostic_training_dataset_v3.py \
  --clean-circuit-roots 500

PYTHONPATH=/workspace/triqto/src \
python -u scripts/v0_2/audit_step5_training_dataset_eda_v3.py
```

The 1,000-root stage and Step 6 baselines remain locked until the v3 full-artifact audit returns `PROMOTION_READY`.
