from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "v0_2"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
CONFIG = ROOT / "configs" / "v0_2" / "step14_cross_motif_generalization_training.json"


def load_module(name: str, filename: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cfg() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_all_four_step14_scripts_exist_and_compile() -> None:
    names = (
        "generate_step14_cross_motif_dataset.py",
        "run_step14_cross_motif_training.py",
        "generate_step14_fresh_legacy_retention_outer.py",
        "evaluate_step14_outer.py",
    )
    for name in names:
        path = SCRIPTS / name
        assert path.is_file(), name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_generator_materializes_only_fit_and_selection_before_training() -> None:
    gen = load_module("step14_generator_contract", "generate_step14_cross_motif_dataset.py")
    c = cfg(); gen.assert_contract(c)
    development = gen.family_indices_for_mode("development", c)
    outer = gen.family_indices_for_mode("simulator_outer", c)
    reserve = [i for i in range(1050) if gen.partition_for_family(i, c) == "future_hardware_reserve"]
    assert len(development) == 750
    assert sum(gen.partition_for_family(i, c) == "fit" for i in development) == 600
    assert sum(gen.partition_for_family(i, c) == "selection" for i in development) == 150
    assert len(outer) == 150 and all(gen.partition_for_family(i, c) == "simulator_outer" for i in outer)
    assert len(reserve) == 150
    assert not set(development) & set(outer)
    assert not set(development) & set(reserve)
    assert not set(outer) & set(reserve)


def test_generator_blueprint_is_deterministic_and_identifiability_is_model_blind() -> None:
    gen = load_module("step14_generator_determinism", "generate_step14_cross_motif_dataset.py")
    c = cfg()
    b1 = gen.family_blueprint(0, 0, c); b2 = gen.family_blueprint(0, 0, c)
    assert b1 == b2
    assert b1["n_qubits"] in {2, 3, 4, 5}
    assert b1["topology_class"] in set(c["cross_motif_dataset"]["circuit_grammar"]["topology_classes"])
    variants = [gen.build_variant(b1, 0, v, c) for v in range(4)]
    assert len(variants) == 4
    source = inspect.getsource(gen.identifiability).lower()
    assert "predict" not in source
    assert "model" not in source
    assert "inject_hidden_rotation" in source


def test_training_runner_is_warm_start_only_and_freezes_before_outer() -> None:
    train = load_module("step14_training_contract", "run_step14_cross_motif_training.py")
    c = cfg(); train.assert_contract(c)
    source = inspect.getsource(train)
    assert "current_development_product.json" in source
    assert "current_simulator_outer_product.json" not in source
    assert "FROZEN_BEFORE_ANY_STEP14_OUTER_MATERIALIZATION" in source
    assert "optimizer_state_reused\": False" in source
    assert "FIT/SELECTION ONLY; NO OUTER / NO QPU" in source


def test_legacy_outer_requires_selection_freeze_and_new_namespace() -> None:
    legacy = load_module("step14_legacy_outer_contract", "generate_step14_fresh_legacy_retention_outer.py")
    c = cfg()
    source = {
        "original_domain": {"global_root_index_start": 5000, "clean_root_count": 1000, "expected_examples": 13000},
        "bridge_domain": {"base_seed": 1, "parent_groups": 60, "variants_per_parent": 8, "expected_clean_roots": 480, "expected_examples": 6240},
    }
    derived = legacy.derived_generation_cfg(source, c)
    assert derived["original_domain"]["global_root_index_start"] == 6000
    assert derived["original_domain"]["clean_root_count"] == 500
    assert derived["original_domain"]["expected_examples"] == 6500
    assert derived["bridge_domain"]["base_seed"] == 2026082502
    assert derived["bridge_domain"]["parent_groups"] == 60
    assert derived["bridge_domain"]["variants_per_parent"] == 8
    assert "--selection-freeze" in inspect.getsource(legacy.parse_args)


def domain(candidate_mech: float, candidate_effect: float, base_mech: float, base_effect: float, min_recall: float = 0.8):
    return {
        "candidate": {"mechanism_balanced_accuracy": candidate_mech, "effect_balanced_accuracy": candidate_effect, "minimum_mechanism_recall": min_recall},
        "step10c": {"mechanism_balanced_accuracy": base_mech, "effect_balanced_accuracy": base_effect, "minimum_mechanism_recall": min_recall},
        "candidate_bootstrap": {"mechanism_ba_ci": [0.76, 0.90], "effect_ba_ci": [0.90, 0.98]},
        "step10c_bootstrap": {"mechanism_ba_ci": [0.60, 0.75], "effect_ba_ci": [0.90, 0.98]},
    }


def test_outer_evaluator_implements_exact_all_or_nothing_gate() -> None:
    ev = load_module("step14_outer_eval_contract", "evaluate_step14_outer.py")
    c = cfg()
    cross = domain(0.84, 0.94, 0.72, 0.93, 0.75)
    original = domain(0.83, 0.95, 0.84, 0.96)
    bridge = domain(0.82, 0.94, 0.83, 0.95)
    paired = {"ci": [0.04, 0.18], "mean": 0.12}
    result = ev.evaluate_support_gate(c, cross, original, bridge, paired)
    assert result["passed"] is True
    assert result["interpretation"] == "CROSS_MOTIF_GENERALIZATION_REPAIR_SUPPORTED_IN_SIMULATION"
    weak = json.loads(json.dumps(cross)); weak["candidate"]["mechanism_balanced_accuracy"] = 0.79
    result = ev.evaluate_support_gate(c, weak, original, bridge, paired)
    assert result["passed"] is False
    assert result["interpretation"] == "CROSS_MOTIF_GENERALIZATION_REPAIR_NOT_SUPPORTED_IN_SIMULATION"


def test_step14_implementation_has_no_qpu_execution_path() -> None:
    for filename in ("run_step14_cross_motif_training.py", "evaluate_step14_outer.py"):
        source = (SCRIPTS / filename).read_text(encoding="utf-8")
        assert "qiskit_ibm_runtime" not in source
        assert "SamplerV2" not in source
        assert "execute_physical_qpu" not in source
    assert "future_hardware_reserve_accessed\":False" in (SCRIPTS / "evaluate_step14_outer.py").read_text(encoding="utf-8")
