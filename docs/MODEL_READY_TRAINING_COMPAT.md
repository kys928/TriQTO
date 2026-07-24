# Model-ready training compatibility

This compatibility layer consumes the immutable Phase 12 preprocessing product
and the split-safe Phase 11 topology attachment product directly. It does not
reinterpret the original native Phase 12 artifact schema.

## Source contract

The loader requires:

- `preprocessed_complete.json` with `complete=true`;
- `manifests/processed_item_manifest.parquet` with a matching SHA-256;
- `manifests/model_input_contract.json` declaring `x_*` inputs and `y_*` targets;
- `manifests/should_act_class_weights.json`;
- strict split-group and entity isolation;
- content-addressed NPZ artifacts loaded with `allow_pickle=False`.

Every scientific NPZ array must be either an `x_*` model input or a `y_*`
target. The only unprefixed arrays accepted are versioned identity and metadata
fields. Privileged/oracle names are rejected from `x_*` inputs.

## Canonical topology vector

The Phase 11 action-neighborhood artifact provides:

- 110 combined persistence features: 55 parameter + 55 Born;
- 11 cross-manifold alignment features;
- separate 55-wide parameter and Born vectors for controlled ablations.

The model input is therefore exactly:

```text
x_topology_features             110
x_topology_alignment_features    11
-----------------------------------
canonical topology input        121
```

The separate parameter and Born arrays are retained in the loaded example as
ablation inputs but are not concatenated into the main model input a second
time.

## Two-stage action head

The action head now emits one graph-level `should_act_logit` and
`should_act_probability` before candidate scoring. Model-ready action loss is:

1. weighted binary cross-entropy using `y_should_act_weight`;
2. listwise candidate-ranking loss only when `y_ranking_loss_mask=true`;
3. candidate reward regression only for the same ranking-active graphs.

No-action examples train the gate and contribute zero candidate-ranking and
candidate-reward loss.

## Leakage and topology boundaries

The model hard policy forbids topology from the action-ranking and
Born-prediction heads. Joint diagnosis may consume the attached topology input.
The topology head cannot consume topology directly. Hardware rows remain
without topology in the current campaign.

```text
lambda_top = 0
supervised topology targets = none
```

## Validation command

```bash
TRIQTO_MODEL_READY_ROOT=/path/to/phase12_topology_<id> \
PYTHONPATH=src \
python scripts/validate_model_ready_training.py
```

The command performs a strict manifest load, content-hash checks for selected
artifacts, an action forward/backward step, a topology-enabled joint forward
pass, and hard-policy assertions.
