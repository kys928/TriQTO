#!/usr/bin/env python3
"""Generate Step 5 v3 cohorts with independent acquisition/context schedules."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "generate_step5_matched_diagnostic_training_dataset_v2.py"
SPEC = importlib.util.spec_from_file_location("triqto_step5_v2_base", V2_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {V2_PATH}")
V2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V2
SPEC.loader.exec_module(V2)
BASE = V2.BASE

SCHEMA = "triqto.v0_2.step5_matched_diagnostic_training_dataset.v3"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v3.json"
DEFAULT_OUTPUT_PARENT = Path("/workspace/triqto-data/step5_matched_diagnostic_training_v3")
DEFAULT_V2_CONFIG = Path(__file__).resolve().parents[2] / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--clean-circuit-roots", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def stable_rng(base_seed: int, namespace: str, root_index: int, extra: int = 0) -> np.random.Generator:
    seed = BASE.stable_seed(base_seed, namespace, root_index, extra)
    return np.random.default_rng(seed)


def affected_qubit_schedule(root_index: int, n_qubits: int, overlay: Mapping[str, Any]) -> list[int]:
    cfg = overlay["scheduling"]
    rng = stable_rng(int(cfg["base_seed"]), str(cfg["affected_qubit_seed_namespace"]), root_index, n_qubits)
    output: list[int] = []
    while len(output) < int(cfg["contexts_per_root"]):
        output.extend(int(v) for v in rng.permutation(n_qubits).tolist())
    return output[: int(cfg["contexts_per_root"])]


def intervention_shot_schedule(root_index: int, overlay: Mapping[str, Any]) -> list[int]:
    cfg = overlay["scheduling"]
    shots = [int(v) for v in cfg["shots"]]
    rng = stable_rng(int(cfg["base_seed"]), str(cfg["intervention_shot_seed_namespace"]), root_index)
    return [int(v) for v in rng.permutation(shots).tolist()]


def clean_control_shots(family_occurrence: int, overlay: Mapping[str, Any]) -> int:
    shots = [int(v) for v in overlay["scheduling"]["shots"]]
    return shots[int(family_occurrence) % len(shots)]


def validate_v3_associations(roots: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]], overlay: Mapping[str, Any]) -> dict[str, Any]:
    gates = overlay["association_gates"]
    injected = [row for row in examples if not bool(row["clean_control"])]
    clean = [row for row in examples if bool(row["clean_control"])]

    def gate(name: str, value: float, limit_key: str) -> None:
        limit = float(gates[limit_key])
        if value > limit:
            raise RuntimeError(f"{name} association too large: {value:.6f} > {limit:.6f}")

    metrics = {
        "shot_strength_cramers_v": V2.cramers_v(injected, "shots", "strength"),
        "shot_depth_cramers_v": V2.cramers_v(injected, "shots", "insertion_depth_bin"),
        "shot_family_cramers_v": V2.cramers_v(injected, "shots", "family"),
        "shot_split_cramers_v": V2.cramers_v(injected, "shots", "split"),
        "clean_shot_family_cramers_v": V2.cramers_v(clean, "shots", "family"),
        "clean_shot_split_cramers_v": V2.cramers_v(clean, "shots", "split"),
    }
    gate("shot/strength", metrics["shot_strength_cramers_v"], "maximum_shot_strength_cramers_v")
    gate("shot/depth", metrics["shot_depth_cramers_v"], "maximum_shot_depth_cramers_v")
    gate("shot/family", metrics["shot_family_cramers_v"], "maximum_shot_family_cramers_v")
    gate("shot/split", metrics["shot_split_cramers_v"], "maximum_shot_split_cramers_v")
    gate("clean-shot/family", metrics["clean_shot_family_cramers_v"], "maximum_clean_shot_family_cramers_v")
    gate("clean-shot/split", metrics["clean_shot_split_cramers_v"], "maximum_clean_shot_split_cramers_v")

    per_n: dict[str, dict[str, float]] = {}
    for n_qubits in sorted({int(row["n_qubits"]) for row in injected}):
        subset = [row for row in injected if int(row["n_qubits"]) == n_qubits]
        values = {
            "shot_affected_cramers_v": V2.cramers_v(subset, "shots", "affected_qubit"),
            "strength_affected_cramers_v": V2.cramers_v(subset, "strength", "affected_qubit"),
            "depth_affected_cramers_v": V2.cramers_v(subset, "insertion_depth_bin", "affected_qubit"),
        }
        if values["shot_affected_cramers_v"] > float(gates["maximum_per_qubit_count_shot_affected_cramers_v"]):
            raise RuntimeError(f"{n_qubits}q shot/affected-qubit alias too large")
        if values["strength_affected_cramers_v"] > float(gates["maximum_per_qubit_count_strength_affected_cramers_v"]):
            raise RuntimeError(f"{n_qubits}q strength/affected-qubit alias too large")
        if values["depth_affected_cramers_v"] > float(gates["maximum_per_qubit_count_depth_affected_cramers_v"]):
            raise RuntimeError(f"{n_qubits}q depth/affected-qubit alias too large")
        per_n[str(n_qubits)] = values

    shots = {str(v) for v in overlay["scheduling"]["shots"]}
    if bool(gates["require_all_four_intervention_shots_per_root"]):
        by_root: dict[str, set[str]] = {}
        for row in injected:
            by_root.setdefault(str(row["clean_circuit_group_id"]), set()).add(str(row["shots"]))
        if any(values != shots for values in by_root.values()):
            raise RuntimeError("not every clean root uses all intervention shot levels")

    for key in ("family", "split"):
        for value in sorted({str(row[key]) for row in injected}):
            subset = {str(row["shots"]) for row in injected if str(row[key]) == value}
            if subset != shots:
                raise RuntimeError(f"intervention shots incomplete for {key}={value}")

    minimum_clean = int(gates["require_all_shot_levels_in_each_family_clean_controls_when_family_roots_at_least"])
    for family in sorted({str(row["family"]) for row in clean}):
        subset_rows = [row for row in clean if str(row["family"]) == family]
        if len(subset_rows) >= minimum_clean and {str(row["shots"]) for row in subset_rows} != shots:
            raise RuntimeError(f"clean-control shots incomplete for family={family}")

    metrics["per_qubit_count_context_associations"] = per_n
    return metrics


def main() -> None:
    args = parse_args()
    overlay = BASE.read_json(args.config.expanduser().resolve())
    if overlay.get("schema") != SCHEMA:
        raise RuntimeError("unexpected Step 5 v3 overlay schema")
    v2cfg = BASE.read_json(args.base_v2_config.expanduser().resolve())
    if v2cfg.get("schema") != V2.SCHEMA:
        raise RuntimeError("unexpected Step 5 v2 base config schema")

    root_count = int(args.clean_circuit_roots)
    allowed = [int(v) for v in v2cfg["stage_progression"]["allowed_root_counts"]]
    if root_count not in allowed:
        raise ValueError(f"--clean-circuit-roots must be one of {allowed}")

    identity = {
        "schema": SCHEMA,
        "overlay_config_sha256": BASE.sha256_file(args.config.expanduser().resolve()),
        "base_v2_config_sha256": BASE.sha256_file(args.base_v2_config.expanduser().resolve()),
        "runner_sha256": BASE.sha256_file(Path(__file__).resolve()),
        "base_v2_runner_sha256": BASE.sha256_file(V2_PATH),
        "clean_circuit_root_count": root_count,
        "rejected_v2_product_id": overlay["rejected_v2"]["product_id"],
    }
    product_id = "product_" + hashlib.sha256(BASE.canonical_json(identity).encode()).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    product_root = output_parent / product_id
    if product_root.exists():
        raise RuntimeError(f"refusing to overwrite existing Step 5 v3 product: {product_root}")
    staging = output_parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()

    roots: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    seen_graph_hashes: set[str] = set()
    plan = V2.root_plan(root_count, v2cfg)
    depths = [float(v) for v in v2cfg["matched_intervention_design"]["depth_target_fractions"]]
    p = v2cfg["privileged_supervision"]
    bound = float(v2cfg["stage_validation"]["diagnostic_delta_absolute_bound"])

    try:
        for planned in plan:
            ri = int(planned["root_index"]); family = str(planned["family"]); occ = int(planned["family_occurrence_index"])
            split = str(planned["split"]); nq = int(planned["n_qubits"]); strengths = [float(v) for v in planned["strength_schedule"]]
            q_schedule = affected_qubit_schedule(ri, nq, overlay)
            shot_schedule = intervention_shot_schedule(ri, overlay)

            circuit = BASE.build_clean_circuit(ri, family, nq, v2cfg)
            graph = BASE.serialize_graph(circuit); ghash = BASE.graph_hash(graph)
            if ghash in seen_graph_hashes: raise RuntimeError(f"duplicate generated clean graph at root {ri}")
            seen_graph_hashes.add(ghash)
            group_id = BASE.sha256_bytes(BASE.canonical_json({"root_index":ri,"family":family,"family_occurrence_index":occ,"n_qubits":nq,"graph_sha256":ghash}).encode())
            clean_state = BASE.normalized_state(circuit)
            clean_probs = {b: BASE.basis_probabilities(clean_state,nq,b) for b in BASE.BASIS_ORDER}
            layout = "identity:" + ",".join(str(i) for i in range(nq))
            roots.append({"root_index":ri,"family_occurrence_index":occ,"clean_circuit_group_id":group_id,"split":split,"family":family,"n_qubits":nq,"unitary_event_count":len(circuit.data),"graph_sha256":ghash,"physical_layout_identity":layout})

            clean_shots = clean_control_shots(occ, overlay)
            ref,pairs = BASE.make_reference_bundle(clean_probs,nq,clean_shots,ri,"clean_control")
            diagnostic,audit = BASE.diagnostic_arrays(clean_state,clean_probs,ref,nq,pairs,clean_shots,ri,"clean_control","clean_control")
            example_id = BASE.sha256_bytes(BASE.canonical_json([group_id,"clean_control"]).encode())
            rel = Path("artifacts")/split/f"{example_id.split(':',1)[1]}.npz"; path = staging/rel
            zero={"population_component":0.0,"phase_component":0.0,"dominance_log_ratio":0.0,"total_overlap_loss":0.0}
            BASE.save_example(path,graph=graph,n_qubits=nq,diagnostic=diagnostic,audit_diagnostic=audit,example_id=example_id,clean_group_id=group_id,clean_control=True,effect_present=False,mechanism_code=-1,mechanism_loss_mask=False,phenotype="clean_control",continuous=zero,affected_qubit=-1,boundary=-1,strength=0.0)
            with np.load(path,allow_pickle=False) as loaded: BASE.validate_array_contract(dict(loaded),bound)
            examples.append({"example_id":example_id,"root_index":ri,"family_occurrence_index":occ,"clean_circuit_group_id":group_id,"split":split,"family":family,"n_qubits":nq,"clean_control":True,"mechanism":"clean_control","effect_present":False,"mechanism_loss_mask":False,"phenomenology":"clean_control","affected_qubit":-1,"insertion_boundary_rank":-1,"insertion_depth_bin":"clean_control","strength":0.0,"shots":clean_shots,"reference_kind":v2cfg["finite_shot_acquisition"]["reference_kind"],"backend_identity":"step5_paired_reference_simulator_emulation_v3","physical_layout_identity":layout,"meta_reference_window_id":f"root{ri}:clean","graph_sha256":ghash,"artifact_path":rel.as_posix(),"artifact_sha256":BASE.sha256_file(path)})

            for ci,(fraction,strength) in enumerate(zip(depths,strengths)):
                boundary=BASE.depth_boundary(len(circuit.data),fraction); aq=int(q_schedule[ci]); shots=int(shot_schedule[ci])
                context=f"ctx{ci}:q{aq}:b{boundary}:s{strength:.12g}:shots{shots}"
                ref,pairs=BASE.make_reference_bundle(clean_probs,nq,shots,ri,context)
                for mechanism in BASE.MECHANISMS:
                    observed=BASE.inject_hidden_rotation(circuit,boundary,aq,mechanism,strength); observed_state=BASE.normalized_state(observed)
                    truth=BASE.state_diagnostics(clean_state,observed_state,epsilon=float(p["epsilon"]),negligible_floor=float(p["negligible_overlap_loss_floor"]),dominance_ratio=float(p["phenomenology_strong_dominance_ratio"]))
                    diagnostic,audit=BASE.diagnostic_arrays(observed_state,clean_probs,ref,nq,pairs,shots,ri,context,mechanism)
                    example_id=BASE.sha256_bytes(BASE.canonical_json([group_id,ci,aq,boundary,strength,shots,mechanism]).encode())
                    rel=Path("artifacts")/split/f"{example_id.split(':',1)[1]}.npz"; path=staging/rel
                    BASE.save_example(path,graph=graph,n_qubits=nq,diagnostic=diagnostic,audit_diagnostic=audit,example_id=example_id,clean_group_id=group_id,clean_control=False,effect_present=bool(truth["effect_present"]),mechanism_code=BASE.MECHANISM_CODES[mechanism],mechanism_loss_mask=bool(truth["effect_present"]),phenotype=str(truth["phenotype"]),continuous=truth,affected_qubit=aq,boundary=boundary,strength=strength)
                    with np.load(path,allow_pickle=False) as loaded: BASE.validate_array_contract(dict(loaded),bound)
                    examples.append({"example_id":example_id,"root_index":ri,"family_occurrence_index":occ,"clean_circuit_group_id":group_id,"split":split,"family":family,"n_qubits":nq,"clean_control":False,"mechanism":mechanism,"effect_present":bool(truth["effect_present"]),"mechanism_loss_mask":bool(truth["effect_present"]),"phenomenology":str(truth["phenotype"]),"affected_qubit":aq,"insertion_boundary_rank":boundary,"insertion_depth_bin":BASE.depth_bin(boundary,len(circuit.data)),"strength":strength,"shots":shots,"reference_kind":v2cfg["finite_shot_acquisition"]["reference_kind"],"backend_identity":"step5_paired_reference_simulator_emulation_v3","physical_layout_identity":layout,"meta_reference_window_id":f"root{ri}:ctx{ci}","graph_sha256":ghash,"artifact_path":rel.as_posix(),"artifact_sha256":BASE.sha256_file(path)})

            if args.progress_every and (ri+1)%args.progress_every==0:
                print(f"Generated {ri+1}/{root_count} clean roots ({len(examples)} examples)",flush=True)

        manifests=staging/"manifests"; BASE.write_csv(manifests/"clean_circuit_manifest.csv",roots); BASE.write_csv(manifests/"example_manifest.csv",examples)
        fam,splitrows,mech=BASE.summarize_rows(roots,examples); BASE.write_csv(manifests/"family_summary.csv",fam); BASE.write_csv(manifests/"split_summary.csv",splitrows); BASE.write_csv(manifests/"mechanism_summary.csv",mech); V2.write_additional_summaries(manifests,roots,examples)
        base_validation=V2.validate_stage_v2(roots,examples,root_count,v2cfg); v3metrics=validate_v3_associations(roots,examples,overlay)
        validation=dict(base_validation); validation["schema"]=SCHEMA; validation.update(v3metrics); BASE.atomic_json(staging/"stage_validation.json",validation)
        manifest_names=[p.name for p in manifests.iterdir() if p.is_file()]
        completion={"schema":SCHEMA,"status":"COMPLETE","product_id":product_id,"identity":identity,"clean_circuit_root_count":root_count,"example_count":len(examples),"train_clean_root_count":validation["train_clean_root_count"],"validation_clean_root_count":validation["validation_clean_root_count"],"train_example_count":next(int(r["example_count"]) for r in splitrows if r["split"]=="train"),"validation_example_count":next(int(r["example_count"]) for r in splitrows if r["split"]=="validation"),"mechanism_counts":validation["mechanism_counts"],"mechanism_supervised_example_count":validation["mechanism_supervised_example_count"],"negligible_injected_example_count":validation["negligible_injected_example_count"],"clean_control_count":validation["clean_control_count"],"selected_diagnostic_variant":v2cfg["deployable_diagnostic_input"]["selected_step4_1_variant"],"primary_input_is_empirical_finite_shot":True,"statevectors_persisted_in_example_artifacts":False,"raw_reference_window_identifier_persisted_as_model_input":False,"historical_v0_1_test_accessed":False,"spent_confirmatory_cohort_accessed":False,"classifier_trained":False,"model_architecture_changed":False,"manifest_hashes":{n:BASE.sha256_file(manifests/n) for n in manifest_names},"stage_validation_sha256":BASE.sha256_file(staging/"stage_validation.json"),"promotion_requires_full_artifact_eda":True}
        BASE.atomic_json(staging/"dataset_complete.json",completion); os.replace(staging,product_root)
        BASE.atomic_json(output_parent/"current_product.json",{"schema":"triqto.v0_2.step5_current_product.v3","product_dir":str(product_root),"product_id":product_id,"clean_circuit_root_count":root_count,"dataset_complete_sha256":BASE.sha256_file(product_root/"dataset_complete.json")})
        print("\nTRIQTO STEP 5 V3 MATCHED DIAGNOSTIC TRAINING DATASET COMPLETE\n")
        print(f"Stage clean roots: {root_count}\nExamples: {len(examples)}\nTrain/validation clean roots: {validation['train_clean_root_count']}/{validation['validation_clean_root_count']}")
        print(f"Family/split Cramer's V: {validation['family_split_cramers_v']:.6f}\nDepth/strength Cramer's V: {validation['depth_strength_cramers_v']:.6f}")
        print(f"Shot/strength Cramer's V: {validation['shot_strength_cramers_v']:.6f}\nShot/depth Cramer's V: {validation['shot_depth_cramers_v']:.6f}\nClean-shot/family Cramer's V: {validation['clean_shot_family_cramers_v']:.6f}")
        print(f"3q train/validation clean roots: {validation['three_qubit_train_root_count']}/{validation['three_qubit_validation_root_count']}")
        print(f"Product: {product_root}\nNext required gate: full-artifact EDA must return PROMOTION_READY before 1000 roots.")
    except Exception:
        import shutil
        if staging.exists(): shutil.rmtree(staging,ignore_errors=True)
        raise

if __name__ == "__main__":
    main()
