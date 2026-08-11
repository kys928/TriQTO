from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/v0_2/generate_step5_matched_diagnostic_training_dataset_v2.py"
)
SPEC = importlib.util.spec_from_file_location("step5_v2_dataset_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
)
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def plan500() -> list[dict]:
    return MODULE.root_plan(500, CONFIG)


def synthetic_manifests() -> tuple[list[dict], list[dict]]:
    roots: list[dict] = []
    examples: list[dict] = []
    depth_names = ("early", "middle", "late", "terminal")
    for row in plan500():
        root_index = int(row["root_index"])
        family = str(row["family"])
        split = str(row["split"])
        n_qubits = int(row["n_qubits"])
        occurrence = int(row["family_occurrence_index"])
        group = f"group-{root_index}"
        graph = f"graph-{root_index}"
        roots.append(
            {
                "root_index": root_index,
                "family_occurrence_index": occurrence,
                "clean_circuit_group_id": group,
                "split": split,
                "family": family,
                "n_qubits": n_qubits,
                "graph_sha256": graph,
            }
        )
        examples.append(
            {
                "clean_circuit_group_id": group,
                "split": split,
                "family": family,
                "n_qubits": n_qubits,
                "graph_sha256": graph,
                "clean_control": True,
                "effect_present": False,
                "mechanism_loss_mask": False,
                "mechanism": "clean_control",
                "insertion_depth_bin": "clean_control",
                "strength": 0.0,
            }
        )
        for depth, strength in zip(depth_names, row["strength_schedule"]):
            for mechanism in MODULE.MECHANISMS:
                examples.append(
                    {
                        "clean_circuit_group_id": group,
                        "split": split,
                        "family": family,
                        "n_qubits": n_qubits,
                        "graph_sha256": graph,
                        "clean_control": False,
                        "effect_present": True,
                        "mechanism_loss_mask": True,
                        "mechanism": mechanism,
                        "insertion_depth_bin": depth,
                        "strength": float(strength),
                    }
                )
    return roots, examples


def test_family_stratified_split_is_exact_80_20_at_500() -> None:
    plan = plan500()
    counts: dict[str, Counter[str]] = {}
    for row in plan:
        counts.setdefault(str(row["family"]), Counter())
        counts[str(row["family"])][str(row["split"])] += 1
    assert counts["bell_like"] == Counter({"train": 20, "validation": 5})
    assert counts["ghz"] == Counter({"train": 60, "validation": 15})
    assert counts["hardware_efficient_ansatz"] == Counter(
        {"train": 60, "validation": 15}
    )
    assert counts["phase_interference"] == Counter({"train": 60, "validation": 15})
    assert counts["qaoa_like"] == Counter({"train": 60, "validation": 15})
    assert counts["qft_like"] == Counter({"train": 60, "validation": 15})
    assert counts["random_shallow"] == Counter({"train": 80, "validation": 20})
    assert MODULE.cramers_v(plan, "family", "split") < 1e-12


def test_three_qubit_roots_exist_in_both_splits() -> None:
    plan = plan500()
    train = sum(int(row["n_qubits"]) == 3 and row["split"] == "train" for row in plan)
    validation = sum(
        int(row["n_qubits"]) == 3 and row["split"] == "validation" for row in plan
    )
    assert train >= CONFIG["stage_validation"]["minimum_three_qubit_train_roots_at_500_stage"]
    assert validation >= CONFIG["stage_validation"][
        "minimum_three_qubit_validation_roots_at_500_stage"
    ]
    # Freeze the current deterministic plan so a future seed/scheduling change is explicit.
    assert (train, validation) == (72, 16)


def test_strength_schedule_deconfounds_every_depth_in_both_splits_and_families() -> None:
    depth_names = ("early", "middle", "late", "terminal")
    rows: list[dict] = []
    for root in plan500():
        for depth, strength in zip(depth_names, root["strength_schedule"]):
            rows.append(
                {
                    "split": root["split"],
                    "family": root["family"],
                    "depth": depth,
                    "strength": f"{float(strength):.2f}",
                }
            )
    assert MODULE.cramers_v(rows, "depth", "strength") < 0.05
    for split in ("train", "validation"):
        for depth in depth_names:
            observed = {
                row["strength"]
                for row in rows
                if row["split"] == split and row["depth"] == depth
            }
            assert observed == {"0.05", "0.15"}
    for family in sorted({row["family"] for row in rows}):
        for depth in depth_names:
            observed = {
                row["strength"]
                for row in rows
                if row["family"] == family and row["depth"] == depth
            }
            assert observed == {"0.05", "0.15"}


def test_500_root_plan_is_nested_inside_1000_root_plan() -> None:
    first = MODULE.root_plan(500, CONFIG)
    larger = MODULE.root_plan(1000, CONFIG)
    assert first == larger[:500]


def test_v2_stage_validator_accepts_the_repaired_synthetic_500_plan() -> None:
    roots, examples = synthetic_manifests()
    result = MODULE.validate_stage_v2(roots, examples, 500, CONFIG)
    assert result["status"] == "PASS"
    assert result["family_split_cramers_v"] < 0.05
    assert result["depth_strength_cramers_v"] < 0.05
    assert result["three_qubit_train_root_count"] == 72
    assert result["three_qubit_validation_root_count"] == 16


def test_three_qubit_clean_circuit_families_are_simulatable() -> None:
    families = sorted(set(CONFIG["clean_circuit_generation"]["family_cycle"]))
    families.remove("bell_like")
    for index, family in enumerate(families):
        circuit = MODULE.BASE.build_clean_circuit(index + 700, family, 3, CONFIG)
        state = MODULE.BASE.normalized_state(circuit)
        assert circuit.num_qubits == 3
        assert state.shape == (8,)
        assert np.isclose(np.linalg.norm(state), 1.0)


def test_raw_reference_window_identifier_is_frozen_meta_only() -> None:
    acquisition = CONFIG["finite_shot_acquisition"]
    assert acquisition["raw_reference_window_identifier_role"] == (
        "META_AUDIT_ONLY_NEVER_MODEL_INPUT"
    )
    primary = set(CONFIG["example_artifact_contract"]["primary_deployable_arrays"])
    assert all("reference_window" not in key for key in primary)
