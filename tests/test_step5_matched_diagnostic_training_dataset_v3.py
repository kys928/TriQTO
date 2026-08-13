from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v0_2/generate_step5_matched_diagnostic_training_dataset_v3.py"
SPEC = importlib.util.spec_from_file_location("step5_v3_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
OVERLAY = json.loads((ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json").read_text())
V2CFG = json.loads((ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json").read_text())


def planned_rows():
    roots = M.V2.root_plan(500, V2CFG)
    contexts = []
    controls = []
    depths = ["early", "middle", "late", "terminal"]
    for r in roots:
        ri = int(r["root_index"]); nq = int(r["n_qubits"]); occ = int(r["family_occurrence_index"])
        qs = M.affected_qubit_schedule(ri, nq, OVERLAY)
        shots = M.intervention_shot_schedule(ri, OVERLAY)
        controls.append({"family":r["family"],"split":r["split"],"shots":M.clean_control_shots(occ,OVERLAY),"n_qubits":nq})
        for ci, depth in enumerate(depths):
            contexts.append({"family":r["family"],"split":r["split"],"n_qubits":nq,"depth":depth,"insertion_depth_bin":depth,"strength":float(r["strength_schedule"][ci]),"shots":shots[ci],"affected_qubit":qs[ci],"clean_circuit_group_id":str(ri),"clean_control":False})
    return roots, contexts, controls


def test_each_intervention_root_uses_all_four_shot_levels():
    roots, contexts, _ = planned_rows()
    by_root = {}
    for row in contexts:
        by_root.setdefault(row["clean_circuit_group_id"], set()).add(row["shots"])
    assert len(by_root) == 500
    assert all(values == {512,1024,2048,4096} for values in by_root.values())


def test_shot_schedule_is_not_confounded_with_strength_or_depth():
    _, contexts, _ = planned_rows()
    assert M.V2.cramers_v(contexts,"shots","strength") <= OVERLAY["association_gates"]["maximum_shot_strength_cramers_v"]
    assert M.V2.cramers_v(contexts,"shots","insertion_depth_bin") <= OVERLAY["association_gates"]["maximum_shot_depth_cramers_v"]


def test_shots_do_not_encode_family_or_split():
    _, contexts, controls = planned_rows()
    assert M.V2.cramers_v(contexts,"shots","family") <= OVERLAY["association_gates"]["maximum_shot_family_cramers_v"]
    assert M.V2.cramers_v(contexts,"shots","split") <= OVERLAY["association_gates"]["maximum_shot_split_cramers_v"]
    assert M.V2.cramers_v(controls,"shots","family") <= OVERLAY["association_gates"]["maximum_clean_shot_family_cramers_v"]
    assert M.V2.cramers_v(controls,"shots","split") <= OVERLAY["association_gates"]["maximum_clean_shot_split_cramers_v"]


def test_affected_qubit_context_associations_stay_below_frozen_limits():
    _, contexts, _ = planned_rows()
    gates = OVERLAY["association_gates"]
    for nq in sorted({row["n_qubits"] for row in contexts}):
        subset=[row for row in contexts if row["n_qubits"]==nq]
        assert M.V2.cramers_v(subset,"shots","affected_qubit") <= gates["maximum_per_qubit_count_shot_affected_cramers_v"]
        assert M.V2.cramers_v(subset,"strength","affected_qubit") <= gates["maximum_per_qubit_count_strength_affected_cramers_v"]
        assert M.V2.cramers_v(subset,"insertion_depth_bin","affected_qubit") <= gates["maximum_per_qubit_count_depth_affected_cramers_v"]


def test_v2_repairs_and_three_qubit_coverage_are_preserved():
    roots, contexts, _ = planned_rows()
    assert M.V2.cramers_v(roots,"family","split") == 0.0
    assert M.V2.cramers_v(contexts,"insertion_depth_bin","strength") <= V2CFG["stage_validation"]["maximum_depth_strength_cramers_v"]
    assert sum(int(r["n_qubits"])==3 and r["split"]=="train" for r in roots) == 72
    assert sum(int(r["n_qubits"])==3 and r["split"]=="validation" for r in roots) == 16
