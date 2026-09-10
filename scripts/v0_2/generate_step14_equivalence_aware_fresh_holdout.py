#!/usr/bin/env python3
"""Generate the frozen fresh Step-14 equivalence-aware simulator holdout.

This is a generation/QC operation only. It performs no model inference and
cannot select or tune the already-frozen equivalence-aware method.

The only preprocessing is redaction/sealing:
- complete raw truth manifests are written under ``sealed_truth/``;
- model-ready manifests under ``manifests/`` omit mechanism class, strength,
  true affected qubit, and true injection boundary;
- scientific NPZ artifacts are never transformed, filtered, or rewritten.

Both pre- and post-preprocessing EDA are label-blind. Distributional surprises
are reported, never used to tune the method or drop valid examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

import benchmark_step6_cheap_baselines as baseline
import generate_step14_cross_motif_dataset as step14gen

ROOT = Path(__file__).resolve().parents[2]
GEN_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_generation.json"
FINAL_FREEZE = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json"
STEP14_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
V2_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
STEP12_CONFIG = ROOT / "configs/v0_2/step12_independent_phase_generalization.json"
SOURCE_DATASET_PARENT = Path("/workspace/triqto-data/step14_cross_motif_dataset")
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
SCHEMA = "triqto.v0_2.step14_equivalence_aware_fresh_holdout.v1"

FORBIDDEN_BLIND_FIELDS = {
    "mechanism",
    "affected_qubit",
    "injection_boundary_rank",
    "injection_boundary",
    "strength",
    "identifiability_min_delta_norm",
    "identifiability_min_pairwise_distance",
}
BLIND_FAMILY_FIELDS = (
    "family_id",
    "family_index",
    "step14_partition",
    "n_qubits",
    "topology_class",
    "injection_context_class",
    "family_signature_sha256",
)
BLIND_ROOT_FIELDS = (
    "root_index",
    "family_id",
    "step14_partition",
    "variant_index",
    "n_qubits",
    "topology_class",
    "injection_context_class",
    "graph_sha256",
    "operation_signature",
)
BLIND_EXAMPLE_FIELDS = (
    "root_index",
    "family_id",
    "step14_partition",
    "example_id",
    "artifact_path",
    "artifact_sha256",
    "evaluation_role",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--final-method-freeze-payload-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--progress-every", type=int, default=10)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def summarize(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def verify_freezes(
    args: argparse.Namespace,
    generation: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> None:
    if generation.get("schema") != "triqto.v0_2.step14_equivalence_aware_fresh_holdout_generation.v1":
        raise RuntimeError("unexpected fresh-holdout generation protocol schema")
    if generation.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT_GENERATION":
        raise RuntimeError("fresh-holdout generation protocol is not frozen")

    fcfg = generation["final_method_freeze"]
    if freeze.get("schema") != str(fcfg["required_schema"]):
        raise RuntimeError("unexpected final method freeze schema")
    if freeze.get("status") != str(fcfg["required_status"]):
        raise RuntimeError("final method is not frozen before fresh holdout")
    expected_payload = str(fcfg["freeze_payload_sha256"])
    if str(freeze.get("freeze_payload_sha256")) != expected_payload:
        raise RuntimeError("final method freeze payload identity drift")
    if args.final_method_freeze_payload_sha256 != expected_payload:
        raise RuntimeError("request final-method-freeze payload SHA differs from generation protocol")
    if str(freeze["method"]["method_id"]) != str(fcfg["method_id"]):
        raise RuntimeError("frozen method id drift")
    selected = freeze["method"]["selected_hyperparameters"]
    if float(selected["tau"]) != float(fcfg["selected_tau"]):
        raise RuntimeError("frozen tau drift")
    if float(selected["frame_temperature"]) != float(fcfg["selected_frame_temperature"]):
        raise RuntimeError("frozen frame temperature drift")
    if args.training_run_id != str(freeze["source_identity"]["training_run_id"]):
        raise RuntimeError("request training run differs from final method freeze")
    if args.selection_freeze_sha256 != str(freeze["source_identity"]["selection_freeze_sha256"]):
        raise RuntimeError("request selection freeze differs from final method freeze")
    if bool(freeze["integrity_boundaries"]["fresh_holdout_generated"]):
        raise RuntimeError("final method freeze unexpectedly claims the holdout already existed")


def safe_materialized_development_signatures() -> set[str]:
    """Read only family signatures from materialized fit/selection metadata.

    The family manifest contains no per-example mechanism class or true
    injection location. Simulator outer and future-reserve products are not
    opened or inspected.
    """
    pointer = SOURCE_DATASET_PARENT / "current_development_product.json"
    if not pointer.is_file():
        raise RuntimeError("current Step-14 development product pointer is missing")
    value = read_json(pointer)
    product = Path(str(value["product_dir"]))
    if not product.is_dir():
        raise RuntimeError("current Step-14 development product directory is missing")
    manifest = product / "manifests" / "family_manifest.csv"
    signatures: set[str] = set()
    partitions: Counter[str] = Counter()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step14_partition", "family_signature_sha256"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError("development family-manifest schema drift")
        for row in reader:
            partition = str(row["step14_partition"])
            if partition not in {"fit", "selection"}:
                raise RuntimeError(f"unexpected development partition {partition!r}")
            partitions[partition] += 1
            signatures.add(str(row["family_signature_sha256"]))
    if partitions != Counter({"fit": 600, "selection": 150}):
        raise RuntimeError(f"development family partition drift: {dict(partitions)}")
    if len(signatures) != 750:
        raise RuntimeError("development family signatures are not unique")
    return signatures


def blind_family(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in BLIND_FAMILY_FIELDS}


def blind_root(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in BLIND_ROOT_FIELDS}


def blind_examples_from_local(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen_roots: set[int] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        root_index = int(row["root_index"])
        role = "reference_control" if root_index not in seen_roots else "target"
        seen_roots.add(root_index)
        projected = {key: row[key] for key in BLIND_EXAMPLE_FIELDS if key != "evaluation_role"}
        projected["evaluation_role"] = role
        out.append(projected)
    counts = Counter(str(row["evaluation_role"]) for row in out)
    if counts != Counter({"target": 48, "reference_control": 4}):
        raise RuntimeError(f"local blind-role construction drift: {dict(counts)}")
    return out


def artifact_qc(base: Path, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    byte_sizes: list[float] = []
    schema_counts: Counter[str] = Counter()
    numeric_values = 0
    nonfinite_numeric_values = 0
    artifact_pairs: list[tuple[str, str]] = []
    for row in examples:
        rel = str(row["artifact_path"])
        expected = str(row["artifact_sha256"])
        path = base / rel
        if not path.is_file():
            raise RuntimeError(f"missing fresh-holdout artifact: {rel}")
        observed = step14gen.BASE.sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"fresh-holdout artifact hash mismatch: {rel}")
        artifact_pairs.append((rel, expected))
        byte_sizes.append(float(path.stat().st_size))
        with np.load(path, allow_pickle=False) as z:
            keys = tuple(sorted(str(k) for k in z.files))
            schema_counts[json.dumps(keys, separators=(",", ":"))] += 1
            for key in z.files:
                arr = np.asarray(z[key])
                if np.issubdtype(arr.dtype, np.number):
                    numeric_values += int(arr.size)
                    if np.issubdtype(arr.dtype, np.inexact):
                        nonfinite_numeric_values += int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))
    if nonfinite_numeric_values:
        raise RuntimeError(f"nonfinite numeric values found: {nonfinite_numeric_values}")
    return {
        "artifact_count": len(examples),
        "artifact_inventory_sha256": stable_hash(sorted(artifact_pairs)),
        "artifact_byte_size": summarize(byte_sizes),
        "numeric_value_count": numeric_values,
        "nonfinite_numeric_value_count": nonfinite_numeric_values,
        "artifact_key_schema_counts": dict(sorted(schema_counts.items())),
    }


def structural_eda(
    *,
    phase: str,
    base: Path,
    families: Sequence[Mapping[str, Any]],
    roots: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    generation_protocol_sha256: str,
    final_freeze_payload_sha256: str,
) -> dict[str, Any]:
    if any(FORBIDDEN_BLIND_FIELDS & set(row.keys()) for row in families):
        raise RuntimeError(f"{phase} EDA family view contains forbidden fields")
    if any(FORBIDDEN_BLIND_FIELDS & set(row.keys()) for row in roots):
        raise RuntimeError(f"{phase} EDA root view contains forbidden fields")
    if any(FORBIDDEN_BLIND_FIELDS & set(row.keys()) for row in examples):
        raise RuntimeError(f"{phase} EDA example view contains forbidden fields")

    root_groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in examples:
        root_groups[int(row["root_index"])].append(row)
    per_root = [len(v) for v in root_groups.values()]
    if set(per_root) != {13}:
        raise RuntimeError(f"{phase} EDA found non-13-example root")

    role_counts = Counter(str(row["evaluation_role"]) for row in examples)
    if role_counts != Counter({"target": 7200, "reference_control": 600}):
        raise RuntimeError(f"{phase} EDA role-count drift: {dict(role_counts)}")

    family_signatures = [str(row["family_signature_sha256"]) for row in families]
    if len(set(family_signatures)) != len(family_signatures):
        raise RuntimeError(f"{phase} EDA found duplicate fresh family signature")

    graph_hashes = [str(row["graph_sha256"]) for row in roots]
    if len(set(graph_hashes)) != len(graph_hashes):
        raise RuntimeError(f"{phase} EDA found duplicate fresh root graph")

    operation_counts: list[float] = []
    for row in roots:
        operations = json.loads(str(row["operation_signature"]))
        if not isinstance(operations, list):
            raise RuntimeError("operation signature is not a list")
        operation_counts.append(float(len(operations)))

    qc = artifact_qc(base, examples)
    return {
        "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_eda.v1",
        "status": "PASS",
        "phase": phase,
        "label_blind": True,
        "mechanism_class_accessed": False,
        "true_location_accessed": False,
        "strength_accessed": False,
        "model_inference": False,
        "qpu_access": False,
        "generation_protocol_sha256": generation_protocol_sha256,
        "final_method_freeze_payload_sha256": final_freeze_payload_sha256,
        "counts": {
            "families": len(families),
            "roots": len(roots),
            "examples": len(examples),
            "reference_controls": int(role_counts["reference_control"]),
            "targets": int(role_counts["target"]),
            "examples_per_root": summarize([float(v) for v in per_root]),
        },
        "support": {
            "n_qubits": dict(sorted(Counter(str(row["n_qubits"]) for row in families).items())),
            "topology_class": dict(sorted(Counter(str(row["topology_class"]) for row in families).items())),
            "injection_context_class": dict(sorted(Counter(str(row["injection_context_class"]) for row in families).items())),
            "operation_count": summarize(operation_counts),
        },
        "uniqueness": {
            "family_signature_count": len(set(family_signatures)),
            "root_graph_hash_count": len(set(graph_hashes)),
        },
        "artifact_qc": qc,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    step14gen.BASE.write_csv(path, list(rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def validate_blind_manifest_columns(path: Path, expected: Sequence[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
    if fields != list(expected):
        raise RuntimeError(f"blind manifest column drift at {path.name}: {fields}")
    overlap = FORBIDDEN_BLIND_FIELDS & set(fields)
    if overlap:
        raise RuntimeError(f"forbidden fields present in blind manifest {path.name}: {sorted(overlap)}")


def main() -> None:
    args = parse_args()
    if args.progress_every < 1 or args.progress_every > 1000:
        raise ValueError("progress_every must be in [1,1000]")

    generation = read_json(GEN_CONFIG)
    freeze = read_json(FINAL_FREEZE)
    verify_freezes(args, generation, freeze)
    step14_cfg = read_json(STEP14_CONFIG)
    step14gen.assert_contract(step14_cfg)
    v2cfg = read_json(V2_CONFIG)
    step12 = read_json(STEP12_CONFIG)

    hold = generation["holdout"]
    if int(hold["family_count"]) != 150 or int(hold["variants_per_family"]) != 4:
        raise RuntimeError("fresh-holdout family contract drift")
    if int(hold["examples_per_root"]) != 13 or int(hold["expected_target_examples"]) != 7200:
        raise RuntimeError("fresh-holdout example contract drift")
    partition = str(hold["partition_name"])
    if partition != "fresh_equivalence_holdout":
        raise RuntimeError("fresh-holdout partition name drift")

    generation_protocol_sha = baseline.sha256_file(GEN_CONFIG)
    final_freeze_file_sha = baseline.sha256_file(FINAL_FREEZE)
    final_freeze_payload_sha = str(freeze["freeze_payload_sha256"])
    step14_protocol_sha = baseline.sha256_file(STEP14_CONFIG)
    step12_sha = baseline.sha256_file(STEP12_CONFIG)

    cfg = copy.deepcopy(step14_cfg)
    cfg["cross_motif_dataset"]["base_seed"] = int(hold["base_seed"])
    family_offset = int(hold["family_index_namespace_offset"])
    family_indices = [family_offset + i for i in range(int(hold["family_count"]))]
    if min(family_indices) <= 1049:
        raise RuntimeError("fresh family-index namespace overlaps original Step-14 namespace")

    existing_signatures = safe_materialized_development_signatures()
    step12_signatures = {
        tuple(str(v) for v in raw["reference_operation_signature"])
        for raw in step12["generalization_design"]["motifs"]
    }

    identity = {
        "schema": SCHEMA,
        "base_seed": int(hold["base_seed"]),
        "family_indices_sha256": stable_hash(family_indices),
        "family_index_namespace_offset": family_offset,
        "family_count": len(family_indices),
        "generation_protocol_sha256": generation_protocol_sha,
        "final_method_freeze_payload_sha256": final_freeze_payload_sha,
        "final_method_freeze_file_sha256": final_freeze_file_sha,
        "step14_protocol_sha256": step14_protocol_sha,
        "step12_signature_source_sha256": step12_sha,
        "model_inference": False,
        "qpu_access": False,
        "simulator_outer_accessed": False,
        "future_hardware_reserve_accessed": False,
    }
    product_id = "fresh_holdout_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]

    parent = args.output_parent.expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    product = parent / product_id
    if product.exists():
        complete = read_json(product / "dataset_complete.json")
        if complete.get("identity") != identity:
            raise RuntimeError("existing fresh-holdout product identity mismatch")
        if complete.get("status") != "COMPLETE_FROZEN_FRESH_HOLDOUT_UNEVALUATED":
            raise RuntimeError("existing fresh-holdout product is not a complete unevaluated holdout")
        print("Fresh equivalence holdout already complete:", product)
        return

    staging = parent / f".{product_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)

    full_families: list[dict[str, Any]] = []
    full_roots: list[dict[str, Any]] = []
    full_examples: list[dict[str, Any]] = []
    blind_families: list[dict[str, Any]] = []
    blind_roots: list[dict[str, Any]] = []
    blind_examples: list[dict[str, Any]] = []
    fresh_signatures: set[str] = set()
    seen_graphs: set[str] = set()
    rejection_counts: Counter[str] = Counter()

    try:
        for pos, family_index in enumerate(family_indices, start=1):
            offset = 0
            while True:
                bp = step14gen.family_blueprint(family_index, offset, cfg)
                variants = [step14gen.build_variant(bp, family_index, v, cfg) for v in range(4)]
                signatures = [step14gen.op_signature(v) for v in variants]
                core = ["h:q0", "rz:q0", "h:q0", "cx:q0-q1"]
                signature_forbidden = (
                    any(tuple(sig) in step12_signatures for sig in signatures)
                    or any(any(sig[i:i + 4] == core for i in range(max(0, len(sig) - 3))) for sig in signatures)
                )
                if signature_forbidden:
                    rejection_counts["legacy_signature_exclusion"] += 1
                    offset += 1
                    continue

                family_signature = step14gen.stable_hash({
                    "events": [(n, list(q)) for n, q in bp["events"]],
                    "context": bp["injection_context_class"],
                })
                if family_signature in existing_signatures:
                    rejection_counts["materialized_fit_selection_family_signature_collision"] += 1
                    offset += 1
                    continue
                if family_signature in fresh_signatures:
                    rejection_counts["fresh_family_signature_collision"] += 1
                    offset += 1
                    continue

                audit = step14gen.identifiability(bp, variants, cfg)
                if audit["status"] != "PASS":
                    rejection_counts["identifiability_admission"] += 1
                    offset += 1
                    continue

                candidate_graphs = [
                    step14gen.BASE.graph_hash(step14gen.BASE.serialize_graph(circuit))
                    for circuit in variants
                ]
                if len(set(candidate_graphs)) != 4 or any(value in seen_graphs for value in candidate_graphs):
                    rejection_counts["fresh_root_graph_collision"] += 1
                    offset += 1
                    continue
                break

            family_id, observed_family_signature, local_roots, local_examples = step14gen.materialize_family(
                staging, family_index, partition, bp, variants, cfg, v2cfg
            )
            if observed_family_signature != family_signature:
                raise RuntimeError("family-signature construction drift")
            if [str(row["graph_sha256"]) for row in local_roots] != candidate_graphs:
                raise RuntimeError("root graph hash changed during materialization")

            fresh_signatures.add(family_signature)
            seen_graphs.update(candidate_graphs)
            full_roots.extend(local_roots)
            full_examples.extend(local_examples)
            blind_roots.extend(blind_root(row) for row in local_roots)
            blind_examples.extend(blind_examples_from_local(local_examples))

            family_row = {
                "family_index": family_index,
                "family_id": family_id,
                "step14_partition": partition,
                "candidate_seed_offset": offset,
                "n_qubits": bp["n_qubits"],
                "topology_class": bp["topology_class"],
                "injection_context_class": bp["injection_context_class"],
                "family_signature_sha256": family_signature,
                "identifiability_status": audit["status"],
                "identifiability_min_delta_norm": audit["minimum_observed_true_mechanism_delta_norm"],
                "identifiability_min_pairwise_distance": audit["minimum_observed_pairwise_mechanism_distance"],
            }
            full_families.append(family_row)
            blind_families.append(blind_family(family_row))

            if args.progress_every and pos % args.progress_every == 0:
                print(
                    f"fresh holdout generated {pos}/{len(family_indices)} families "
                    f"(candidate rejections={sum(rejection_counts.values())})",
                    flush=True,
                )

        if len(full_families) != 150 or len(full_roots) != 600 or len(full_examples) != 7800:
            raise RuntimeError("fresh-holdout materialized count mismatch")
        if len(blind_examples) != 7800:
            raise RuntimeError("fresh-holdout blind projection count mismatch")

        pre_eda = structural_eda(
            phase="PRE_PREPROCESSING",
            base=staging,
            families=blind_families,
            roots=blind_roots,
            examples=blind_examples,
            generation_protocol_sha256=generation_protocol_sha,
            final_freeze_payload_sha256=final_freeze_payload_sha,
        )
        atomic_json(staging / "eda_pre_preprocessing.json", pre_eda)

        sealed = staging / "sealed_truth"
        manifests = staging / "manifests"
        write_csv(sealed / "family_manifest.csv", full_families)
        write_csv(sealed / "root_manifest.csv", full_roots)
        write_csv(sealed / "example_manifest.csv", full_examples)
        write_csv(manifests / "family_manifest.csv", blind_families)
        write_csv(manifests / "root_manifest.csv", blind_roots)
        write_csv(manifests / "example_manifest.csv", blind_examples)

        validate_blind_manifest_columns(manifests / "family_manifest.csv", BLIND_FAMILY_FIELDS)
        validate_blind_manifest_columns(manifests / "root_manifest.csv", BLIND_ROOT_FIELDS)
        validate_blind_manifest_columns(manifests / "example_manifest.csv", BLIND_EXAMPLE_FIELDS)

        preprocessing = {
            "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_preprocessing.v1",
            "status": "PASS",
            "kind": "REDACTION_AND_SEALING_ONLY",
            "raw_scientific_artifacts_modified": False,
            "rows_filtered_or_dropped": False,
            "numeric_measurements_transformed": False,
            "mechanism_class_exposed_in_blind_view": False,
            "true_location_exposed_in_blind_view": False,
            "strength_exposed_in_blind_view": False,
            "sealed_truth_manifests": {
                name: baseline.sha256_file(sealed / name)
                for name in ("family_manifest.csv", "root_manifest.csv", "example_manifest.csv")
            },
            "blind_manifests": {
                name: baseline.sha256_file(manifests / name)
                for name in ("family_manifest.csv", "root_manifest.csv", "example_manifest.csv")
            },
            "raw_artifact_inventory_sha256": pre_eda["artifact_qc"]["artifact_inventory_sha256"],
            "generation_protocol_sha256": generation_protocol_sha,
            "final_method_freeze_payload_sha256": final_freeze_payload_sha,
        }
        atomic_json(staging / "preprocessing_audit.json", preprocessing)

        post_families = read_csv(manifests / "family_manifest.csv")
        post_roots = read_csv(manifests / "root_manifest.csv")
        post_examples = read_csv(manifests / "example_manifest.csv")
        post_eda = structural_eda(
            phase="POST_PREPROCESSING",
            base=staging,
            families=post_families,
            roots=post_roots,
            examples=post_examples,
            generation_protocol_sha256=generation_protocol_sha,
            final_freeze_payload_sha256=final_freeze_payload_sha,
        )
        post_eda["preprocessing_identity_checks"] = {
            "family_count_preserved": len(post_families) == len(blind_families),
            "root_count_preserved": len(post_roots) == len(blind_roots),
            "example_count_preserved": len(post_examples) == len(blind_examples),
            "artifact_inventory_preserved": (
                post_eda["artifact_qc"]["artifact_inventory_sha256"]
                == pre_eda["artifact_qc"]["artifact_inventory_sha256"]
            ),
            "raw_scientific_artifacts_modified": False,
        }
        if not all(bool(v) is True for v in post_eda["preprocessing_identity_checks"].values()):
            raise RuntimeError("post-preprocessing identity check failed")
        atomic_json(staging / "eda_post_preprocessing.json", post_eda)

        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE_FROZEN_FRESH_HOLDOUT_UNEVALUATED",
            "product_id": product_id,
            "identity": identity,
            "partition": partition,
            "family_count": 150,
            "root_count": 600,
            "example_count": 7800,
            "target_example_count": 7200,
            "reference_control_count": 600,
            "generation_rejection_counts": dict(sorted(rejection_counts.items())),
            "generation_rejection_count_total": int(sum(rejection_counts.values())),
            "preprocessing": "REDACTION_AND_SEALING_ONLY",
            "model_evaluated": False,
            "oracle_free_predictions_frozen": False,
            "selection_metrics_used": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "method_changed_after_freeze": False,
            "final_method_freeze_payload_sha256": final_freeze_payload_sha,
            "generation_protocol_sha256": generation_protocol_sha,
            "sealed_truth_manifest_sha256": preprocessing["sealed_truth_manifests"],
            "blind_manifest_sha256": preprocessing["blind_manifests"],
            "eda_pre_preprocessing_sha256": baseline.sha256_file(staging / "eda_pre_preprocessing.json"),
            "preprocessing_audit_sha256": baseline.sha256_file(staging / "preprocessing_audit.json"),
            "eda_post_preprocessing_sha256": baseline.sha256_file(staging / "eda_post_preprocessing.json"),
            "raw_artifact_inventory_sha256": pre_eda["artifact_qc"]["artifact_inventory_sha256"],
        }
        atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product)

        pointer = {
            "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_pointer.v1",
            "status": "FRESH_HOLDOUT_GENERATED_UNEVALUATED",
            "product_id": product_id,
            "product_dir": str(product),
            "dataset_complete_sha256": baseline.sha256_file(product / "dataset_complete.json"),
            "final_method_freeze_payload_sha256": final_freeze_payload_sha,
            "model_evaluated": False,
        }
        atomic_json(parent / "current_fresh_holdout.json", pointer)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nTRIQTO STEP-14 EQUIVALENCE-AWARE FRESH HOLDOUT COMPLETE")
    print("Product:", product_id)
    print("Families: 150 | Roots: 600 | Examples: 7800 | Targets: 7200")
    print("Preprocessing: REDACTION_AND_SEALING_ONLY")
    print("Pre-processing EDA: PASS")
    print("Post-processing EDA: PASS")
    print("Model evaluated: NO")
    print("Fresh holdout selects anything: NO")
    print("Simulator outer accessed: NO")
    print("Future hardware reserve accessed: NO")
    print("QPU executed: NO")
    print("Output:", product)


if __name__ == "__main__":
    main()
