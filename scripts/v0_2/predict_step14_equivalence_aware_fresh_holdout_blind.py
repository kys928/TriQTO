#!/usr/bin/env python3
"""Freeze Step-14 equivalence-aware predictions on the blind fresh holdout.

This is Phase A of the confirmatory protocol. It is deliberately restricted to
blind manifests and x-only NPZ artifacts. It must finish and hash the 7,200
oracle-free predictions before any process is allowed to read sealed truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_step14_latent_frame_inference as latent
import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import run_step14_oracle_raw_evidence_ceiling as oracle_compat
import step14_equivalence_aware_common as equiv

ROOT = Path(__file__).resolve().parents[2]
FINAL_FREEZE = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json"
CONFIRM_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json"
HOLDOUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
METHOD_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
OUTPUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout_confirmation")
SCHEMA = "triqto.v0_2.step14_equivalence_aware_oracle_free_prediction_freeze.v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--final-method-freeze-payload-sha256", required=True)
    p.add_argument("--fresh-holdout-product-id", required=True)
    p.add_argument("--fresh-holdout-dataset-complete-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8") as h:
        json.dump(value, h, indent=2, sort_keys=True, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def path_is_blind_artifact(product: Path, path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(product.resolve())
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] == "blind_artifacts" and "sealed_truth" not in rel.parts


def load_x_only_artifact(product: Path, row: Mapping[str, str]) -> dict[str, np.ndarray]:
    rel = Path(str(row["artifact_path"]))
    path = product / rel
    if not path_is_blind_artifact(product, path):
        raise RuntimeError(f"Phase A refused non-blind artifact path: {rel}")
    if baseline.sha256_file(path) != str(row["artifact_sha256"]):
        raise RuntimeError(f"blind artifact hash mismatch for {row['example_id']}")
    with np.load(path, allow_pickle=False) as source:
        names = [str(k) for k in source.files]
        if not names or any(not name.startswith("x__") for name in names):
            raise RuntimeError(f"Phase A found non-x array in {row['example_id']}: {names}")
        return {name: np.asarray(source[name]) for name in names}


def pad_profiles(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.full((len(values), max_candidates, 3), -np.inf, dtype=np.float64)
    for i, value in enumerate(values):
        out[i, : value.shape[0]] = value
    return out


def pad_pair_float(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.ones((len(values), max_candidates, max_candidates), dtype=np.float32)
    for i, value in enumerate(values):
        n = value.shape[0]
        out[i, :n, :n] = value
    return out


def pad_pair_bool(values: Sequence[np.ndarray], max_candidates: int) -> np.ndarray:
    out = np.zeros((len(values), max_candidates, max_candidates), dtype=np.bool_)
    for i, value in enumerate(values):
        n = value.shape[0]
        out[i, :n, :n] = value
    return out


def extract_blind_table(
    product: Path,
    target_rows: Sequence[Mapping[str, str]],
    safe_roots: Mapping[int, Mapping[str, str]],
    latent_cfg: Mapping[str, Any],
    progress_every: int,
) -> dict[str, Any]:
    cache: dict[int, dict[str, Any]] = {}
    candidate_features: list[np.ndarray] = []
    profile_scores: list[np.ndarray] = []
    pair_distances: list[np.ndarray] = []
    pair_gates: list[np.ndarray] = []
    ids: list[str] = []
    families: list[str] = []
    variants: list[str] = []
    roots: list[int] = []

    for position, row in enumerate(target_rows, start=1):
        if str(row.get("evaluation_role")) != "target":
            raise RuntimeError("non-target row reached Phase A")
        loaded = load_x_only_artifact(product, row)
        root_index = int(row["root_index"])
        root = safe_roots.get(root_index)
        if root is None:
            raise RuntimeError(f"blind root missing: {root_index}")
        delta, weights, pairs = frame.measured_delta_and_weights(loaded)
        if root_index not in cache:
            clean = frame.circuit_from_serialized(loaded)
            signature = oracle.parse_operation_signature(str(root["operation_signature"]))
            if latent.operation_qubits(clean) != signature:
                raise RuntimeError(f"blind circuit reconstruction mismatch at root {root_index}")
            candidates = latent.plausible_candidates(clean)
            finite_jacs = latent.finite_shot_candidate_jacobians(
                clean, pairs, candidates, root_index, latent_cfg
            )
            distance, gate = equiv.finite_frame_pair_geometry(finite_jacs)
            cache[root_index] = {
                "finite_jacs": finite_jacs,
                "distance": distance,
                "gate": gate,
            }
        root_cache = cache[root_index]
        finite_jacs = root_cache["finite_jacs"]
        per_candidate = np.stack(
            [frame.canonicalize_evidence(delta, jac, weights)[0] for jac in finite_jacs]
        ).astype(np.float32)
        profiles = latent.profile_log_likelihoods(delta, weights, finite_jacs).astype(np.float64)
        candidate_features.append(per_candidate)
        profile_scores.append(profiles)
        pair_distances.append(root_cache["distance"])
        pair_gates.append(root_cache["gate"])
        ids.append(str(row["example_id"]))
        families.append(str(row["family_id"]))
        variants.append(str(root["variant_index"]))
        roots.append(root_index)
        if progress_every and position % progress_every == 0:
            print(
                f"blind confirmatory extraction {position}/{len(target_rows)} roots_cached={len(cache)}",
                flush=True,
            )

    x, mask = latent.pad_candidate_features(candidate_features)
    max_candidates = int(x.shape[1])
    return {
        "x": x,
        "mask": mask,
        "profiles": pad_profiles(profile_scores, max_candidates),
        "distance": pad_pair_float(pair_distances, max_candidates),
        "gate": pad_pair_bool(pair_gates, max_candidates),
        "example_id": ids,
        "family_id": families,
        "variant_index": variants,
        "root_index": roots,
        "root_count": len(cache),
    }


def write_predictions(path: Path, table: Mapping[str, Any], logits: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    fields = [
        "example_id", "family_id", "variant_index", "root_index", "candidate_count",
        "predicted_index", "logit_0", "logit_1", "logit_2",
    ]
    with temp.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fields)
        writer.writeheader()
        for i in range(len(logits)):
            writer.writerow({
                "example_id": table["example_id"][i],
                "family_id": table["family_id"][i],
                "variant_index": table["variant_index"][i],
                "root_index": int(table["root_index"][i]),
                "candidate_count": int(np.asarray(table["mask"])[i].sum()),
                "predicted_index": int(np.argmax(logits[i])),
                "logit_0": format(float(logits[i, 0]), ".17g"),
                "logit_1": format(float(logits[i, 1]), ".17g"),
                "logit_2": format(float(logits[i, 2]), ".17g"),
            })
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    oracle_compat.apply_frozen_support_bound()
    confirm = read_json(CONFIRM_CONFIG)
    freeze = read_json(FINAL_FREEZE)
    if confirm.get("status") != "FROZEN_AFTER_HOLDOUT_GENERATION_BEFORE_MODEL_INFERENCE":
        raise RuntimeError("confirmatory protocol is not frozen for Phase A")
    method = confirm["method_identity"]
    hold = confirm["holdout_identity"]
    source = freeze["source_identity"]
    if args.training_run_id != str(source["training_run_id"]):
        raise RuntimeError("training run identity drift")
    if args.selection_freeze_sha256 != str(source["selection_freeze_sha256"]):
        raise RuntimeError("selection freeze identity drift")
    if args.final_method_freeze_payload_sha256 != str(method["final_method_freeze_payload_sha256"]):
        raise RuntimeError("final method freeze identity drift")
    if args.fresh_holdout_product_id != str(hold["product_id"]):
        raise RuntimeError("fresh holdout product identity drift")
    if args.fresh_holdout_dataset_complete_sha256 != str(hold["dataset_complete_sha256"]):
        raise RuntimeError("fresh holdout dataset hash drift")
    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)
    latent_cfg = latent.load_frozen_config()

    pointer = read_json(HOLDOUT_PARENT / "current_fresh_holdout.json")
    product = Path(str(pointer["product_dir"])).resolve()
    if pointer.get("status") != "FRESH_HOLDOUT_GENERATED_UNEVALUATED" or bool(pointer.get("model_evaluated", True)):
        raise RuntimeError("fresh holdout pointer is not unevaluated")
    if str(pointer["product_id"]) != args.fresh_holdout_product_id:
        raise RuntimeError("fresh holdout pointer product drift")
    if baseline.sha256_file(product / "dataset_complete.json") != args.fresh_holdout_dataset_complete_sha256:
        raise RuntimeError("fresh holdout completion hash mismatch")
    complete = read_json(product / "dataset_complete.json")
    if complete.get("status") != "COMPLETE_FROZEN_FRESH_HOLDOUT_UNEVALUATED" or bool(complete.get("model_evaluated", True)):
        raise RuntimeError("fresh holdout completion boundary is not unevaluated")
    for key in ("blind_artifact_inventory_sha256", "x_payload_inventory_sha256"):
        if str(complete[key]) != str(hold[key]):
            raise RuntimeError(f"fresh holdout {key} drift")

    manifests = product / "manifests"
    examples = baseline.read_csv(manifests / "example_manifest.csv")
    roots = baseline.read_csv(manifests / "root_manifest.csv")
    forbidden_fields = {
        "mechanism", "strength", "affected_qubit", "injection_boundary_rank", "injection_boundary",
    }
    if any(forbidden_fields & set(row) for row in examples + roots):
        raise RuntimeError("privileged fields found in blind manifests")
    target_rows = [row for row in examples if str(row.get("evaluation_role")) == "target"]
    if len(target_rows) != int(hold["target_example_count"]):
        raise RuntimeError(f"expected {hold['target_example_count']} target rows, got {len(target_rows)}")
    safe_roots = {
        int(row["root_index"]): {
            "operation_signature": str(row["operation_signature"]),
            "variant_index": str(row["variant_index"]),
        }
        for row in roots
    }
    if len(safe_roots) != int(hold["root_count"]):
        raise RuntimeError("blind root count drift")

    table = extract_blind_table(product, target_rows, safe_roots, latent_cfg, args.progress_every)
    if table["root_count"] != int(hold["root_count"]):
        raise RuntimeError("Phase A did not cover all fresh roots")

    method_pointer = read_json(METHOD_PARENT / "current_equivalence_aware_fit.json")
    method_dir = Path(str(method_pointer["method_dir"])).resolve()
    if str(method_pointer["method_id"]) != str(method["method_id"]):
        raise RuntimeError("method pointer id drift")
    if str(method_pointer["method_bundle_sha256"]) != str(method["method_bundle_sha256"]):
        raise RuntimeError("method pointer bundle hash drift")
    if str(method_pointer["fit_development_result_sha256"]) != str(method["fit_development_result_sha256"]):
        raise RuntimeError("method pointer fit-result hash drift")
    if str(method_pointer["fit_complete_sha256"]) != str(method["fit_complete_sha256"]):
        raise RuntimeError("method pointer fit-complete hash drift")
    if baseline.sha256_file(method_dir / "method_bundle.json") != str(method["method_bundle_sha256"]):
        raise RuntimeError("method bundle bytes drift")

    device = resolve_device(args.device)
    candidate_logits: list[np.ndarray] = []
    for seed in [int(v) for v in method["candidate_network_seeds"]]:
        name = f"candidate_network_seed{seed}.pt"
        path = method_dir / name
        if baseline.sha256_file(path) != str(method["checkpoint_sha256"][name]):
            raise RuntimeError(f"checkpoint hash drift: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if str(payload.get("method_id")) != str(method["method_id"]) or int(payload.get("seed")) != seed:
            raise RuntimeError(f"checkpoint identity drift: {name}")
        if (
            float(payload.get("selected_tau")) != float(method["tau"])
            or float(payload.get("selected_frame_temperature")) != float(method["frame_temperature"])
        ):
            raise RuntimeError(f"checkpoint hyperparameter drift: {name}")
        _uniform, per_candidate = equiv.infer_candidate_network(
            table["x"], table["mask"], state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg, device=device,
        )
        candidate_logits.append(per_candidate)
        print(f"blind inference checkpoint seed={seed} complete", flush=True)

    ensemble = np.mean(np.stack(candidate_logits), axis=0)
    logits = equiv.equivalence_aware_pool_batch(
        ensemble, table["profiles"], table["distance"], table["gate"], table["mask"],
        tau=float(method["tau"]), frame_temperature=float(method["frame_temperature"]),
    )
    if logits.shape != (int(hold["target_example_count"]), 3) or not np.all(np.isfinite(logits)):
        raise RuntimeError("oracle-free prediction shape/finiteness failure")

    confirm_sha = baseline.sha256_file(CONFIRM_CONFIG)
    prediction_key = hashlib.sha256(
        (confirm_sha + args.fresh_holdout_dataset_complete_sha256 + str(method["method_id"])).encode()
    ).hexdigest()[:24]
    prediction_id = f"predictions_{prediction_key}"
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    output_dir = output_parent / prediction_id
    if output_dir.exists() and any(output_dir.iterdir()):
        existing = read_json(output_dir / "oracle_free_predictions_complete.json")
        if existing.get("status") == "PREDICTIONS_FROZEN_BEFORE_TRUTH_ACCESS":
            print(json.dumps(existing, indent=2, sort_keys=True))
            return
        raise RuntimeError(f"refusing to overwrite prediction product: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "oracle_free_predictions.csv"
    write_predictions(pred_path, table, logits)
    pred_sha = baseline.sha256_file(pred_path)
    completion = {
        "schema": SCHEMA,
        "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH_ACCESS",
        "prediction_id": prediction_id,
        "prediction_dir": str(output_dir),
        "confirmatory_protocol_sha256": confirm_sha,
        "final_method_freeze_payload_sha256": str(method["final_method_freeze_payload_sha256"]),
        "fresh_holdout_product_id": args.fresh_holdout_product_id,
        "fresh_holdout_dataset_complete_sha256": args.fresh_holdout_dataset_complete_sha256,
        "method_id": str(method["method_id"]),
        "tau": float(method["tau"]),
        "frame_temperature": float(method["frame_temperature"]),
        "checkpoint_sha256": dict(method["checkpoint_sha256"]),
        "prediction_count": int(len(logits)),
        "oracle_free_predictions_sha256": pred_sha,
        "sealed_truth_accessed": False,
        "true_location_accessed": False,
        "true_mechanism_accessed": False,
        "selection_metrics_used": False,
        "simulator_outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
        "method_changed_after_freeze": False,
    }
    complete_path = output_dir / "oracle_free_predictions_complete.json"
    atomic_json(complete_path, completion)
    complete_sha = baseline.sha256_file(complete_path)
    atomic_json(output_parent / "current_oracle_free_predictions.json", {
        "schema": "triqto.v0_2.step14_equivalence_aware_prediction_pointer.v1",
        "status": "PREDICTIONS_FROZEN_BEFORE_TRUTH_ACCESS",
        "prediction_id": prediction_id,
        "prediction_dir": str(output_dir),
        "oracle_free_predictions_sha256": pred_sha,
        "oracle_free_predictions_complete_sha256": complete_sha,
        "fresh_holdout_product_id": args.fresh_holdout_product_id,
        "sealed_truth_accessed": False,
    })
    print(
        json.dumps({**completion, "oracle_free_predictions_complete_sha256": complete_sha}, indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
