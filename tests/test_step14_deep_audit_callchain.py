from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts/v0_2/audit_step14_development_cohort.py"


def test_static_input_audit_follows_real_delegated_materialization_chain() -> None:
    source = AUDIT.read_text(encoding="utf-8")

    assert 'materializer_text = (repo/"scripts/v0_2/run_step7_full_development_benchmark.py")' in source
    assert '"step14_delegates_to_step7_materializer": "step7.materialize_blocks(" in training_text' in source
    assert '"step7_materializer_loads_hashed_npz": "smoke_runner.load_example(product, row)" in materializer_text' in source
    assert '"step7_materializer_calls_graph_adapter": "batch_from_step5_examples(examples, device=\\"cpu\\")" in materializer_text' in source
    assert '"graph_adapter_exposes_expected_targets": expected_targets <= set(y_keys)' in source
    assert 'pipeline_contract = all(call_chain.values())' in source

    # Regression guard against the original false-negative implementation,
    # which incorrectly required delegated calls to appear directly in Step-14.
    assert '"batch_from_step5_examples(examples" in training_text' not in source
    assert '"smoke_runner.load_example(product, row)" in training_text' not in source
