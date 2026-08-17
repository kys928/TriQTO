from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v0_2/step8_untouched_confirmatory.json"
V2_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
GENERATOR = ROOT / "scripts/v0_2/generate_step8_untouched_confirmatory_cohort.py"
EVALUATOR = ROOT / "scripts/v0_2/run_step8_one_shot_confirmatory_evaluation.py"


def load_module(name: str, path: Path):
    scripts = str(ROOT / "scripts/v0_2")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_is_frozen_before_cohort_generation_and_architecture_search_is_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_CONFIRMATORY_COHORT_GENERATION"
    assert config["final_architecture"]["variant"] == "late_concat"
    assert config["final_architecture"]["architecture_search_closed"] is True
    assert config["final_architecture"]["no_step7_2"] is True
    assert config["confirmatory_cohort"]["clean_circuit_roots"] == 2000
    assert config["confirmatory_cohort"]["expected_examples"] == 26000


def test_confirmatory_namespace_preserves_family_cycle_but_changes_generation_indices() -> None:
    generator = load_module("triqto_step8_generator_test", GENERATOR)
    v2cfg = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    plan = generator.confirmatory_plan(40, 1_000_000, v2cfg)
    assert [row["root_index"] for row in plan] == list(range(40))
    assert [row["generation_root_index"] for row in plan] == list(range(1_000_000, 1_000_040))
    for row in plan:
        local_family = generator.BASE.family_for_root(int(row["root_index"]), v2cfg)
        generated_family = generator.BASE.family_for_root(int(row["generation_root_index"]), v2cfg)
        assert row["family"] == local_family == generated_family


def test_confirmatory_offset_must_preserve_family_cycle() -> None:
    generator = load_module("triqto_step8_generator_bad_offset_test", GENERATOR)
    v2cfg = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    try:
        generator.confirmatory_plan(2, 1_000_001, v2cfg)
    except RuntimeError as exc:
        assert "family cycle" in str(exc)
    else:
        raise AssertionError("non-cycle-aligned confirmatory offset was accepted")


def test_public_manifest_row_contains_no_forbidden_target_fields() -> None:
    generator = load_module("triqto_step8_public_row_test", GENERATOR)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    row = generator.public_example_row(
        example_id="sha256:abc", root_index=0, generation_root_index=1_000_000,
        group_id="sha256:def", family="ghz", n_qubits=3, clean_control=False,
        context_index=0, affected_qubit=1, boundary=2, depth_bin="middle",
        strength=0.05, shots=1024, graph_sha256="sha256:graph",
        artifact_path="artifacts/confirmatory/x.npz", artifact_sha256="sha256:file",
        layout="identity:0,1,2",
    )
    forbidden = set(config["blinding_and_sealing"]["human_visible_example_manifest_excludes"])
    assert forbidden.isdisjoint(row)
    assert row["split"] == "confirmatory"


def test_confirmatory_group_builder_accepts_shuffled_manifest_and_requires_13_derivatives() -> None:
    evaluator = load_module("triqto_step8_evaluator_group_test", EVALUATOR)
    rows = []
    for root in (1, 0):
        for derivative in range(13):
            rows.append({"root_index": str(root), "split": "confirmatory", "artifact_path": f"{root}-{derivative}.npz", "artifact_sha256": "x"})
    roots, by_root = evaluator.build_confirmatory_groups(rows, 2, 13)
    assert roots == [0, 1]
    assert len(by_root[0]) == 13 and len(by_root[1]) == 13


def test_primary_and_secondary_gates_are_absolute_and_predeclared() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    primary = config["evaluation"]["primary_gate"]
    secondary = config["evaluation"]["secondary_gates"]
    assert primary["balanced_accuracy_ci_low_strictly_greater_than"] == 0.45
    assert primary["minimum_each_mechanism_recall"] == 0.40
    assert secondary["effect_balanced_accuracy_ci_low_strictly_greater_than"] == 0.65
    assert secondary["integrated_balanced_accuracy_ci_low_strictly_greater_than"] == 0.40


def test_evaluator_marks_access_before_confirmatory_npz_materialization_and_fixes_late_concat() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    marker = text.index("write_access_marker(access_marker")
    unseal = text.index("confirm_blocks = step7.materialize_blocks")
    assert marker < unseal
    assert 'step7.instantiate_model("late_concat"' in text
    assert "late_concat_parity_residual" not in text
    assert "architecture_changed_after_confirmatory_access\": False" in text
