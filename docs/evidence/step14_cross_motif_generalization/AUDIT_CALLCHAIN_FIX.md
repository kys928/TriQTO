# Step-14 deep-audit call-chain correction

The first deep-audit run produced a false hard failure in `static_model_input_key_audit` because the audit required `smoke_runner.load_example(...)` and `batch_from_step5_examples(...)` to appear directly in `run_step14_cross_motif_training.py`.

The actual frozen training path delegates materialization to `step7.materialize_blocks(...)`, implemented in `run_step7_full_development_benchmark.py`, where hashed NPZ artifacts are loaded and passed into `batch_from_step5_examples(...)`.

The corrected audit validates that delegated chain explicitly. No dataset, model, training, split, threshold, or scientific acceptance criterion is changed.
