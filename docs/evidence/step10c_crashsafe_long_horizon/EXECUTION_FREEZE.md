# Step 10C authoritative execution freeze

Status: **FROZEN AFTER FRESH-OUTER DATA-QUALITY ACCEPTANCE AND BEFORE ANY STEP-10C MODEL OUTCOME**

The scientific Step-10C protocol remains unchanged. No model has been trained or evaluated on the accepted fresh outer cohort while making this execution freeze.

## Authoritative entrypoint

The authoritative training/evaluation command must invoke:

`scripts/v0_2/run_step10c_crashsafe_long_horizon_strict.py`

The existing `run_step10c_crashsafe_long_horizon.py` remains the implementation/helper module containing the crash-safe per-seed training machinery. The strict entrypoint was added only to make the fresh-outer access boundary mechanically explicit and reviewer-auditable.

## Scientific settings unchanged

The strict entrypoint does not change:

- `late_concat` architecture;
- 453,829 trainable parameters;
- graph/input contract;
- Step-10 fit or selection data;
- warm-start or scratch initializations;
- seeds 1701/1702/1703;
- AdamW;
- learning rate 0.0003;
- weight decay 0.0001;
- root batch size 32;
- effect/mechanism loss weights 1.0/1.0;
- gradient clip norm 1.0;
- equal original/bridge optimizer-block schedule;
- early-stopping patience 4;
- early-stopping minimum delta 0.0005;
- maximum epoch ceiling 40;
- retention eligibility rule;
- best-checkpoint ordering;
- selection-only effect threshold choice;
- 2,000 bootstrap replicates and frozen gates.

## Strict outer boundary

The authoritative entrypoint verifies the fresh outer product and its manifests before training, but it does **not materialize fresh-outer NPZ artifacts or run a model forward pass on them during training/selection**.

The sequence is:

1. Materialize only original/bridge fit and selection partitions.
2. Train/select warm-start seeds 1701/1702/1703.
3. Train/select scratch seeds 1701/1702/1703.
4. Require all six selected records and atomically write `outer_evaluation_started.json`.
5. Only after that marker exists, materialize the accepted fresh original and bridge outer NPZ artifacts.
6. Evaluate the six already-selected states and untouched Step-9A baseline.
7. Select effect thresholds only from the already-stored original+bridge selection predictions.
8. Compute fresh-outer metrics/bootstrap and write final immutable outputs.

The final result contains `evaluation_boundary.json` and records `fresh_outer_materialized_after_all_six_selections=true`.

Human monitoring cannot alter checkpoint selection. Fresh outer data select no epoch, checkpoint, architecture, optimizer setting, or threshold.

## Crash safety

Per initialization/seed, the implementation atomically maintains:

- `best.pt` on each frozen eligible-checkpoint improvement;
- `resume.pt` after each completed epoch, including model, AdamW, early-stopping state, best states, history, and Python/NumPy/Torch RNG state;
- `progress.json` after each epoch for observation only.

Resume requires exact identity equality including training config hash, authoritative runner hash, mixture/fresh-outer/Step-9A identities, initialization, seed, architecture, parameter count, execution device, and runtime fingerprint. A mismatch refuses continuation.

## Tests required before merge/training

The pod must pass:

- Python compilation for the generator, implementation runner, strict entrypoint, and Step-10C tests;
- `tests/test_step10c_crashsafe_contract.py`;
- `tests/test_step10c_strict_outer_boundary.py`;
- explicit `late_concat` trainable-parameter count = 453,829.

No Step-10C training is authorized before those checks pass on the exact branch head merged to `main`.