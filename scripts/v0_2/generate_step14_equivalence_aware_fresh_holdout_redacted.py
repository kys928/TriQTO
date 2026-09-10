#!/usr/bin/env python3
"""Generate the frozen fresh Step-14 equivalence-aware holdout with x-only blind NPZs.

Generation and EDA are model-free. Raw truth-bearing NPZs are byte-preserved
under sealed_truth/raw_artifacts/. Separate blind_artifacts/*.npz contain every
and only the frozen deployable x__ arrays. No valid row is filtered or changed.
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
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
import generate_step14_equivalence_aware_fresh_holdout as legacy

ROOT = Path(__file__).resolve().parents[2]
GEN_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_generation.json"
REDACTION_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_artifact_redaction.json"
FINAL_FREEZE = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json"
STEP14_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
V2_CONFIG = ROOT / "configs/v0_2/step5_matched_diagnostic_training_dataset_v2.json"
STEP12_CONFIG = ROOT / "configs/v0_2/step12_independent_phase_generalization.json"
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
SCHEMA = "triqto.v0_2.step14_equivalence_aware_fresh_holdout.v1"

FORBIDDEN_PREFIXES = ("y__", "audit__", "meta__")
DELTA_KEYS = (
    "x__delta_local_expectations",
    "x__delta_pairwise_correlations",
    "x__delta_global_parity",
)
SHOT_KEYS = ("x__observed_shots", "x__reference_shots")


def parse_args():
    return legacy.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: Any) -> str:
    return legacy.stable_hash(value)


def array_payload_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in sorted(arrays):
        arr = np.asarray(arrays[key])
        h.update(key.encode("utf-8")); h.update(b"\0")
        h.update(arr.dtype.str.encode("ascii")); h.update(b"\0")
        h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii")); h.update(b"\0")
        h.update(np.ascontiguousarray(arr).tobytes()); h.update(b"\0")
    return "sha256:" + h.hexdigest()


def verify_redaction_contract(generation: Mapping[str, Any], redaction: Mapping[str, Any], freeze: Mapping[str, Any]) -> set[str]:
    if redaction.get("schema") != "triqto.v0_2.step14_equivalence_aware_fresh_holdout_artifact_redaction.v1":
        raise RuntimeError("unexpected artifact-redaction protocol schema")
    if redaction.get("status") != "FROZEN_BEFORE_FRESH_HOLDOUT_GENERATION":
        raise RuntimeError("artifact-redaction protocol is not frozen")
    base = redaction["base_generation_protocol"]
    if generation.get("schema") != str(base["required_schema"]) or generation.get("status") != str(base["required_status"]):
        raise RuntimeError("base generation protocol identity drift")
    if str(redaction["final_method_freeze_payload_sha256"]) != str(freeze["freeze_payload_sha256"]):
        raise RuntimeError("redaction protocol/final method freeze mismatch")
    policy = redaction["blind_artifact_policy"]
    if list(policy["allowed_array_prefixes"]) != ["x__"]:
        raise RuntimeError("blind artifact prefix contract drift")
    if tuple(policy["forbidden_array_prefixes"]) != FORBIDDEN_PREFIXES:
        raise RuntimeError("blind forbidden-prefix contract drift")
    required = {str(v) for v in redaction["required_x_arrays"]}
    if len(required) != len(redaction["required_x_arrays"]):
        raise RuntimeError("duplicate required x-array names")
    return required


def project_local_examples(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen_roots: set[int] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        root = int(row["root_index"])
        role = "reference_control" if root not in seen_roots else "target"
        seen_roots.add(root)
        out.append({
            "root_index": row["root_index"],
            "family_id": row["family_id"],
            "step14_partition": row["step14_partition"],
            "example_id": row["example_id"],
            "artifact_path": row["artifact_path"],
            "artifact_sha256": row["artifact_sha256"],
            "evaluation_role": role,
        })
    if Counter(str(v["evaluation_role"]) for v in out) != Counter({"target": 48, "reference_control": 4}):
        raise RuntimeError("local blind-role construction drift")
    return out


def _prefix(name: str) -> str:
    if name.startswith("x__"): return "x__"
    if name.startswith("y__"): return "y__"
    if name.startswith("audit__"): return "audit__"
    if name.startswith("meta__"): return "meta__"
    return "other"


def artifact_qc(base: Path, examples: Sequence[Mapping[str, Any]], required_x: set[str], *, blind: bool) -> dict[str, Any]:
    artifact_pairs: list[tuple[str, str]] = []
    x_payload_pairs: list[tuple[str, str]] = []
    byte_sizes: list[float] = []
    x_schema_counts: Counter[str] = Counter()
    prefix_array_counts: Counter[str] = Counter()
    numeric_value_count = 0
    nonfinite = 0
    values: dict[str, list[float]] = {key: [] for key in (*DELTA_KEYS, *SHOT_KEYS)}

    for row in examples:
        example_id = str(row["example_id"])
        rel = str(row["artifact_path"])
        expected = str(row["artifact_sha256"])
        path = base / rel
        if not path.is_file():
            raise RuntimeError(f"missing artifact {rel}")
        observed = baseline.sha256_file(path)
        if observed != expected:
            raise RuntimeError(f"artifact hash mismatch for {example_id}")
        artifact_pairs.append((example_id, expected))
        byte_sizes.append(float(path.stat().st_size))

        with np.load(path, allow_pickle=False) as z:
            names = [str(k) for k in z.files]
            for name in names:
                prefix_array_counts[_prefix(name)] += 1
            if blind and any(not name.startswith("x__") for name in names):
                raise RuntimeError(f"blind artifact contains non-x array: {example_id}")
            x_names = sorted(name for name in names if name.startswith("x__"))
            if set(x_names) != required_x:
                raise RuntimeError(f"x-array schema drift for {example_id}: {x_names}")
            x_schema_counts[json.dumps(x_names, separators=(",", ":"))] += 1
            arrays = {name: np.asarray(z[name]) for name in x_names}

        step14gen.BASE.validate_array_contract(arrays, 2.000001)
        x_payload_pairs.append((example_id, array_payload_sha256(arrays)))
        for name, arr in arrays.items():
            if np.issubdtype(arr.dtype, np.number):
                numeric_value_count += int(arr.size)
                if np.issubdtype(arr.dtype, np.inexact):
                    nonfinite += int(arr.size - np.count_nonzero(np.isfinite(arr)))
            if name in values:
                values[name].extend(np.asarray(arr, dtype=np.float64).ravel().tolist())

    if nonfinite:
        raise RuntimeError(f"nonfinite x-array numeric values found: {nonfinite}")
    summaries = {key: legacy.summarize(val) for key, val in values.items()}
    return {
        "artifact_count": len(examples),
        "artifact_inventory_sha256": stable_hash(sorted(artifact_pairs)),
        "x_payload_inventory_sha256": stable_hash(sorted(x_payload_pairs)),
        "artifact_byte_size": legacy.summarize(byte_sizes),
        "numeric_x_value_count": numeric_value_count,
        "nonfinite_x_numeric_value_count": nonfinite,
        "x_array_key_schema_counts": dict(sorted(x_schema_counts.items())),
        "array_prefix_counts": dict(sorted(prefix_array_counts.items())),
        "x_numeric_summaries": summaries,
    }


def structural_eda(*, phase: str, base: Path, families: Sequence[Mapping[str, Any]], roots: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]], required_x: set[str], blind_artifacts: bool, generation_sha: str, redaction_sha: str, final_freeze_sha: str) -> dict[str, Any]:
    forbidden = legacy.FORBIDDEN_BLIND_FIELDS
    if any(forbidden & set(row.keys()) for row in families):
        raise RuntimeError(f"{phase} family view contains forbidden fields")
    if any(forbidden & set(row.keys()) for row in roots):
        raise RuntimeError(f"{phase} root view contains forbidden fields")
    if any(forbidden & set(row.keys()) for row in examples):
        raise RuntimeError(f"{phase} example view contains forbidden fields")

    by_root: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in examples:
        by_root[int(row["root_index"])].append(row)
    per_root = [len(v) for v in by_root.values()]
    if set(per_root) != {13}:
        raise RuntimeError(f"{phase}: root example-count drift")
    roles = Counter(str(row["evaluation_role"]) for row in examples)
    if roles != Counter({"target": 7200, "reference_control": 600}):
        raise RuntimeError(f"{phase}: role-count drift {dict(roles)}")

    fam_sigs = [str(row["family_signature_sha256"]) for row in families]
    graph_hashes = [str(row["graph_sha256"]) for row in roots]
    if len(set(fam_sigs)) != 150 or len(set(graph_hashes)) != 600:
        raise RuntimeError(f"{phase}: uniqueness drift")
    op_counts: list[float] = []
    for row in roots:
        sig = json.loads(str(row["operation_signature"]))
        if not isinstance(sig, list):
            raise RuntimeError("operation signature schema drift")
        op_counts.append(float(len(sig)))

    qc = artifact_qc(base, examples, required_x, blind=blind_artifacts)
    return {
        "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_eda.v2",
        "status": "PASS",
        "phase": phase,
        "label_blind": True,
        "mechanism_class_value_accessed": False,
        "true_location_value_accessed": False,
        "strength_value_accessed": False,
        "non_x_raw_array_values_accessed": False,
        "model_inference": False,
        "qpu_access": False,
        "generation_protocol_sha256": generation_sha,
        "artifact_redaction_protocol_sha256": redaction_sha,
        "final_method_freeze_payload_sha256": final_freeze_sha,
        "counts": {
            "families": len(families), "roots": len(roots), "examples": len(examples),
            "reference_controls": int(roles["reference_control"]), "targets": int(roles["target"]),
            "examples_per_root": legacy.summarize([float(v) for v in per_root]),
        },
        "support": {
            "n_qubits": dict(sorted(Counter(str(row["n_qubits"]) for row in families).items())),
            "topology_class": dict(sorted(Counter(str(row["topology_class"]) for row in families).items())),
            "injection_context_class": dict(sorted(Counter(str(row["injection_context_class"]) for row in families).items())),
            "operation_count": legacy.summarize(op_counts),
        },
        "uniqueness": {"family_signature_count": len(set(fam_sigs)), "root_graph_hash_count": len(set(graph_hashes))},
        "artifact_qc": qc,
    }


def write_ordered_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def redact_artifacts(staging: Path, full_examples: list[dict[str, Any]], pre_blind_examples: Sequence[Mapping[str, Any]], required_x: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(full_examples) != len(pre_blind_examples):
        raise RuntimeError("preprocessing row alignment drift")
    blind_rows: list[dict[str, Any]] = []
    raw_pairs: list[tuple[str, str]] = []
    blind_pairs: list[tuple[str, str]] = []
    x_pairs: list[tuple[str, str]] = []
    removed: Counter[str] = Counter()

    for raw_row, projected in zip(full_examples, pre_blind_examples):
        example_id = str(raw_row["example_id"])
        if example_id != str(projected["example_id"]):
            raise RuntimeError("preprocessing example alignment drift")
        original_rel = Path(str(raw_row["artifact_path"]))
        original = staging / original_rel
        expected_raw_sha = str(raw_row["artifact_sha256"])
        if baseline.sha256_file(original) != expected_raw_sha:
            raise RuntimeError(f"raw pre-move hash mismatch {example_id}")

        sealed_rel = Path("sealed_truth") / "raw_artifacts" / original_rel
        sealed = staging / sealed_rel
        sealed.parent.mkdir(parents=True, exist_ok=True)
        os.replace(original, sealed)
        if baseline.sha256_file(sealed) != expected_raw_sha:
            raise RuntimeError(f"raw bytes changed while sealing {example_id}")
        raw_row["artifact_path"] = sealed_rel.as_posix()
        raw_pairs.append((example_id, expected_raw_sha))

        with np.load(sealed, allow_pickle=False) as source:
            names = [str(k) for k in source.files]
            x_names = sorted(name for name in names if name.startswith("x__"))
            if set(x_names) != required_x:
                raise RuntimeError(f"raw x-array schema drift {example_id}")
            arrays = {name: np.array(source[name], copy=True) for name in x_names}
            for name in names:
                pfx = _prefix(name)
                if pfx in FORBIDDEN_PREFIXES:
                    removed[pfx] += 1

        x_sha = array_payload_sha256(arrays)
        x_pairs.append((example_id, x_sha))
        blind_rel = Path("blind_artifacts") / f"{example_id.split(':', 1)[-1]}.npz"
        blind = staging / blind_rel
        blind.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(blind, **arrays)

        with np.load(blind, allow_pickle=False) as check:
            blind_names = [str(k) for k in check.files]
            if any(not name.startswith("x__") for name in blind_names):
                raise RuntimeError(f"forbidden array survived redaction {example_id}")
            if set(blind_names) != required_x:
                raise RuntimeError(f"blind x-array schema drift {example_id}")
            for name, source_arr in arrays.items():
                target = np.asarray(check[name])
                if source_arr.dtype != target.dtype or source_arr.shape != target.shape or not np.array_equal(source_arr, target):
                    raise RuntimeError(f"x-array value drift during redaction {example_id}:{name}")

        blind_sha = baseline.sha256_file(blind)
        blind_pairs.append((example_id, blind_sha))
        row = dict(projected)
        row["artifact_path"] = blind_rel.as_posix()
        row["artifact_sha256"] = blind_sha
        blind_rows.append(row)

    return blind_rows, {
        "raw_artifact_inventory_sha256": stable_hash(sorted(raw_pairs)),
        "blind_artifact_inventory_sha256": stable_hash(sorted(blind_pairs)),
        "x_payload_inventory_sha256": stable_hash(sorted(x_pairs)),
        "forbidden_prefix_array_count_removed": dict(sorted(removed.items())),
        "raw_bytes_preserved": True,
        "x_array_names_dtypes_shapes_values_preserved": True,
        "rows_filtered_or_dropped": False,
        "numeric_measurements_transformed": False,
    }


def main() -> None:
    args = parse_args()
    if args.progress_every < 1 or args.progress_every > 1000:
        raise ValueError("progress_every must be in [1,1000]")

    generation = read_json(GEN_CONFIG); redaction = read_json(REDACTION_CONFIG); freeze = read_json(FINAL_FREEZE)
    legacy.verify_freezes(args, generation, freeze)
    required_x = verify_redaction_contract(generation, redaction, freeze)
    step14_cfg = read_json(STEP14_CONFIG); step14gen.assert_contract(step14_cfg)
    v2cfg = read_json(V2_CONFIG); step12 = read_json(STEP12_CONFIG)

    hold = generation["holdout"]
    if (int(hold["family_count"]), int(hold["variants_per_family"]), int(hold["examples_per_root"])) != (150, 4, 13):
        raise RuntimeError("fresh holdout count contract drift")
    if int(hold["expected_target_examples"]) != 7200:
        raise RuntimeError("fresh holdout target-count contract drift")
    partition = str(hold["partition_name"])
    if partition != "fresh_equivalence_holdout":
        raise RuntimeError("fresh holdout partition drift")

    generation_sha = baseline.sha256_file(GEN_CONFIG)
    redaction_sha = baseline.sha256_file(REDACTION_CONFIG)
    final_freeze_file_sha = baseline.sha256_file(FINAL_FREEZE)
    final_freeze_payload_sha = str(freeze["freeze_payload_sha256"])
    cfg = copy.deepcopy(step14_cfg)
    cfg["cross_motif_dataset"]["base_seed"] = int(hold["base_seed"])
    family_offset = int(hold["family_index_namespace_offset"])
    family_indices = [family_offset + i for i in range(150)]
    if min(family_indices) <= 1049:
        raise RuntimeError("fresh family namespace overlaps original Step-14 namespace")

    existing_signatures = legacy.safe_materialized_development_signatures()
    step12_signatures = {tuple(str(v) for v in raw["reference_operation_signature"]) for raw in step12["generalization_design"]["motifs"]}
    identity = {
        "schema": SCHEMA,
        "base_seed": int(hold["base_seed"]),
        "family_indices_sha256": stable_hash(family_indices),
        "family_index_namespace_offset": family_offset,
        "family_count": 150,
        "generation_protocol_sha256": generation_sha,
        "artifact_redaction_protocol_sha256": redaction_sha,
        "final_method_freeze_payload_sha256": final_freeze_payload_sha,
        "final_method_freeze_file_sha256": final_freeze_file_sha,
        "step14_protocol_sha256": baseline.sha256_file(STEP14_CONFIG),
        "step12_signature_source_sha256": baseline.sha256_file(STEP12_CONFIG),
        "model_inference": False, "qpu_access": False,
        "simulator_outer_accessed": False, "future_hardware_reserve_accessed": False,
    }
    product_id = "fresh_holdout_" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    parent = args.output_parent.expanduser().resolve(); parent.mkdir(parents=True, exist_ok=True)
    product = parent / product_id
    if product.exists():
        complete = read_json(product / "dataset_complete.json")
        if complete.get("identity") != identity or complete.get("status") != "COMPLETE_FROZEN_FRESH_HOLDOUT_UNEVALUATED":
            raise RuntimeError("existing fresh holdout product identity/status mismatch")
        print("Fresh equivalence holdout already complete:", product)
        return

    staging = parent / f".{product_id}.staging-{uuid.uuid4().hex}"; staging.mkdir(parents=True, exist_ok=False)
    full_families: list[dict[str, Any]] = []; full_roots: list[dict[str, Any]] = []; full_examples: list[dict[str, Any]] = []
    blind_families: list[dict[str, Any]] = []; blind_roots: list[dict[str, Any]] = []; pre_blind_examples: list[dict[str, Any]] = []
    fresh_signatures: set[str] = set(); seen_graphs: set[str] = set(); rejection_counts: Counter[str] = Counter()

    try:
        for pos, family_index in enumerate(family_indices, start=1):
            offset = 0
            while True:
                bp = step14gen.family_blueprint(family_index, offset, cfg)
                variants = [step14gen.build_variant(bp, family_index, v, cfg) for v in range(4)]
                signatures = [step14gen.op_signature(v) for v in variants]
                core = ["h:q0", "rz:q0", "h:q0", "cx:q0-q1"]
                forbidden = any(tuple(sig) in step12_signatures for sig in signatures) or any(any(sig[i:i+4] == core for i in range(max(0, len(sig)-3))) for sig in signatures)
                if forbidden:
                    rejection_counts["legacy_signature_exclusion"] += 1; offset += 1; continue
                family_signature = step14gen.stable_hash({"events": [(n, list(q)) for n, q in bp["events"]], "context": bp["injection_context_class"]})
                if family_signature in existing_signatures:
                    rejection_counts["materialized_fit_selection_family_signature_collision"] += 1; offset += 1; continue
                if family_signature in fresh_signatures:
                    rejection_counts["fresh_family_signature_collision"] += 1; offset += 1; continue
                audit = step14gen.identifiability(bp, variants, cfg)
                if audit["status"] != "PASS":
                    rejection_counts["identifiability_admission"] += 1; offset += 1; continue
                candidate_graphs = [step14gen.BASE.graph_hash(step14gen.BASE.serialize_graph(c)) for c in variants]
                if len(set(candidate_graphs)) != 4 or any(v in seen_graphs for v in candidate_graphs):
                    rejection_counts["fresh_root_graph_collision"] += 1; offset += 1; continue
                break

            family_id, observed_sig, local_roots, local_examples = step14gen.materialize_family(staging, family_index, partition, bp, variants, cfg, v2cfg)
            if observed_sig != family_signature or [str(r["graph_sha256"]) for r in local_roots] != candidate_graphs:
                raise RuntimeError("materialization identity drift")
            fresh_signatures.add(family_signature); seen_graphs.update(candidate_graphs)
            full_roots.extend(local_roots); full_examples.extend(local_examples)
            blind_roots.extend(legacy.blind_root(r) for r in local_roots)
            pre_blind_examples.extend(project_local_examples(local_examples))
            family_row = {
                "family_index": family_index, "family_id": family_id, "step14_partition": partition,
                "candidate_seed_offset": offset, "n_qubits": bp["n_qubits"], "topology_class": bp["topology_class"],
                "injection_context_class": bp["injection_context_class"], "family_signature_sha256": family_signature,
                "identifiability_status": audit["status"],
                "identifiability_min_delta_norm": audit["minimum_observed_true_mechanism_delta_norm"],
                "identifiability_min_pairwise_distance": audit["minimum_observed_pairwise_mechanism_distance"],
            }
            full_families.append(family_row); blind_families.append(legacy.blind_family(family_row))
            if pos % args.progress_every == 0:
                print(f"fresh holdout generated {pos}/150 families (candidate rejections={sum(rejection_counts.values())})", flush=True)

        if (len(full_families), len(full_roots), len(full_examples), len(pre_blind_examples)) != (150, 600, 7800, 7800):
            raise RuntimeError("fresh holdout materialized count mismatch")

        pre_eda = structural_eda(
            phase="PRE_PREPROCESSING", base=staging, families=blind_families, roots=blind_roots,
            examples=pre_blind_examples, required_x=required_x, blind_artifacts=False,
            generation_sha=generation_sha, redaction_sha=redaction_sha, final_freeze_sha=final_freeze_payload_sha,
        )
        legacy.atomic_json(staging / "eda_pre_preprocessing.json", pre_eda)

        blind_examples, prep = redact_artifacts(staging, full_examples, pre_blind_examples, required_x)
        sealed = staging / "sealed_truth"; manifests = staging / "manifests"
        step14gen.BASE.write_csv(sealed / "family_manifest.csv", full_families)
        step14gen.BASE.write_csv(sealed / "root_manifest.csv", full_roots)
        step14gen.BASE.write_csv(sealed / "example_manifest.csv", full_examples)
        write_ordered_csv(manifests / "family_manifest.csv", blind_families, legacy.BLIND_FAMILY_FIELDS)
        write_ordered_csv(manifests / "root_manifest.csv", blind_roots, legacy.BLIND_ROOT_FIELDS)
        write_ordered_csv(manifests / "example_manifest.csv", blind_examples, legacy.BLIND_EXAMPLE_FIELDS)

        preprocessing = {
            "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_preprocessing.v2",
            "status": "PASS", "kind": "RAW_SEALING_PLUS_X_ONLY_ARTIFACT_REDACTION",
            **prep,
            "sealed_truth_manifest_sha256": {name: baseline.sha256_file(sealed / name) for name in ("family_manifest.csv", "root_manifest.csv", "example_manifest.csv")},
            "blind_manifest_sha256": {name: baseline.sha256_file(manifests / name) for name in ("family_manifest.csv", "root_manifest.csv", "example_manifest.csv")},
            "generation_protocol_sha256": generation_sha,
            "artifact_redaction_protocol_sha256": redaction_sha,
            "final_method_freeze_payload_sha256": final_freeze_payload_sha,
            "model_inference": False,
        }
        if prep["raw_artifact_inventory_sha256"] != pre_eda["artifact_qc"]["artifact_inventory_sha256"]:
            raise RuntimeError("raw artifact byte inventory changed during sealing")
        if prep["x_payload_inventory_sha256"] != pre_eda["artifact_qc"]["x_payload_inventory_sha256"]:
            raise RuntimeError("x payload inventory changed during preprocessing")
        legacy.atomic_json(staging / "preprocessing_audit.json", preprocessing)

        post_families = read_csv(manifests / "family_manifest.csv"); post_roots = read_csv(manifests / "root_manifest.csv"); post_examples = read_csv(manifests / "example_manifest.csv")
        post_eda = structural_eda(
            phase="POST_PREPROCESSING", base=staging, families=post_families, roots=post_roots,
            examples=post_examples, required_x=required_x, blind_artifacts=True,
            generation_sha=generation_sha, redaction_sha=redaction_sha, final_freeze_sha=final_freeze_payload_sha,
        )
        checks = {
            "family_count_preserved": len(post_families) == 150,
            "root_count_preserved": len(post_roots) == 600,
            "example_count_preserved": len(post_examples) == 7800,
            "x_payload_inventory_preserved": post_eda["artifact_qc"]["x_payload_inventory_sha256"] == pre_eda["artifact_qc"]["x_payload_inventory_sha256"],
            "x_numeric_summaries_preserved": post_eda["artifact_qc"]["x_numeric_summaries"] == pre_eda["artifact_qc"]["x_numeric_summaries"],
            "blind_artifacts_contain_only_x_prefix": set(post_eda["artifact_qc"]["array_prefix_counts"]) <= {"x__"},
            "raw_bytes_preserved": bool(prep["raw_bytes_preserved"]),
            "numeric_measurements_transformed": bool(prep["numeric_measurements_transformed"]),
        }
        if not all(bool(v) for k, v in checks.items() if k != "numeric_measurements_transformed") or checks["numeric_measurements_transformed"]:
            raise RuntimeError(f"post-preprocessing identity check failed: {checks}")
        post_eda["preprocessing_identity_checks"] = checks
        legacy.atomic_json(staging / "eda_post_preprocessing.json", post_eda)

        completion = {
            "schema": SCHEMA, "status": "COMPLETE_FROZEN_FRESH_HOLDOUT_UNEVALUATED",
            "product_id": product_id, "identity": identity, "partition": partition,
            "family_count": 150, "root_count": 600, "example_count": 7800,
            "target_example_count": 7200, "reference_control_count": 600,
            "generation_rejection_counts": dict(sorted(rejection_counts.items())),
            "generation_rejection_count_total": int(sum(rejection_counts.values())),
            "preprocessing": "RAW_SEALING_PLUS_X_ONLY_ARTIFACT_REDACTION",
            "model_evaluated": False, "oracle_free_predictions_frozen": False,
            "selection_metrics_used": False, "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False, "qpu_executed": False,
            "method_changed_after_freeze": False,
            "final_method_freeze_payload_sha256": final_freeze_payload_sha,
            "generation_protocol_sha256": generation_sha,
            "artifact_redaction_protocol_sha256": redaction_sha,
            "sealed_truth_manifest_sha256": preprocessing["sealed_truth_manifest_sha256"],
            "blind_manifest_sha256": preprocessing["blind_manifest_sha256"],
            "sealed_raw_artifact_inventory_sha256": prep["raw_artifact_inventory_sha256"],
            "blind_artifact_inventory_sha256": prep["blind_artifact_inventory_sha256"],
            "x_payload_inventory_sha256": prep["x_payload_inventory_sha256"],
            "eda_pre_preprocessing_sha256": baseline.sha256_file(staging / "eda_pre_preprocessing.json"),
            "preprocessing_audit_sha256": baseline.sha256_file(staging / "preprocessing_audit.json"),
            "eda_post_preprocessing_sha256": baseline.sha256_file(staging / "eda_post_preprocessing.json"),
        }
        legacy.atomic_json(staging / "dataset_complete.json", completion)
        os.replace(staging, product)
        legacy.atomic_json(parent / "current_fresh_holdout.json", {
            "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_pointer.v1",
            "status": "FRESH_HOLDOUT_GENERATED_UNEVALUATED", "product_id": product_id,
            "product_dir": str(product), "dataset_complete_sha256": baseline.sha256_file(product / "dataset_complete.json"),
            "final_method_freeze_payload_sha256": final_freeze_payload_sha, "model_evaluated": False,
        })
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise

    print("\nTRIQTO STEP-14 EQUIVALENCE-AWARE FRESH HOLDOUT COMPLETE")
    print("Product:", product_id)
    print("Families: 150 | Roots: 600 | Examples: 7800 | Targets: 7200")
    print("Preprocessing: RAW_SEALING_PLUS_X_ONLY_ARTIFACT_REDACTION")
    print("Pre-processing EDA: PASS | Post-processing EDA: PASS")
    print("Model evaluated: NO | Simulator outer accessed: NO | Future reserve accessed: NO | QPU: NO")
    print("Output:", product)


if __name__ == "__main__":
    main()
