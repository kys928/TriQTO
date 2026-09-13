#!/usr/bin/env python3
"""One-shot scoring of frozen Step-14 equivalence-aware fresh-holdout predictions.

Phase B refuses to open sealed truth until the Phase-A prediction CSV and
completion object hash-verify.  The frozen oracle-free logits are never changed.
Raw and exact-local-frame references are trained on original FIT only and are
analysis comparators; the fresh holdout selects nothing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_step14_local_frame_canonicalization as frame
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import analyze_step14_representation_fusion_head as rep14
import benchmark_step6_cheap_baselines as baseline
import run_step14_cross_motif_training as step14
import run_step14_oracle_raw_evidence_ceiling as oracle_compat

ROOT = Path(__file__).resolve().parents[2]
FINAL_FREEZE = ROOT / "configs/v0_2/step14_equivalence_aware_latent_frame_final_method_freeze.json"
CONFIRM_CONFIG = ROOT / "configs/v0_2/step14_equivalence_aware_fresh_holdout_confirmatory.json"
STEP14_CONFIG = ROOT / "configs/v0_2/step14_cross_motif_generalization_training.json"
HOLDOUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
PREDICTION_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout_confirmation")
OUTPUT_PARENT = PREDICTION_PARENT
SCHEMA = "triqto.v0_2.step14_equivalence_aware_fresh_holdout_confirmatory_result.v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run-id", required=True)
    p.add_argument("--selection-freeze-sha256", required=True)
    p.add_argument("--final-method-freeze-payload-sha256", required=True)
    p.add_argument("--fresh-holdout-product-id", required=True)
    p.add_argument("--fresh-holdout-dataset-complete-sha256", required=True)
    p.add_argument("--oracle-free-prediction-complete-sha256", required=True)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    p.add_argument("--progress-every", type=int, default=5000)
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
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def load_x_only(path: Path, expected_sha: str, example_id: str) -> dict[str, np.ndarray]:
    if baseline.sha256_file(path) != expected_sha:
        raise RuntimeError(f"x-only artifact hash mismatch for {example_id}")
    with np.load(path, allow_pickle=False) as source:
        names = [str(k) for k in source.files]
        if not names or any(not name.startswith("x__") for name in names):
            raise RuntimeError(f"non-x data reached confirmatory feature extractor: {example_id}")
        return {name: np.asarray(source[name]) for name in names}


def prediction_rows_and_logits(pointer: Mapping[str, Any], expected_complete_sha: str) -> tuple[list[dict[str, str]], np.ndarray, dict[str, Any]]:
    if pointer.get("status") != "PREDICTIONS_FROZEN_BEFORE_TRUTH_ACCESS":
        raise RuntimeError("Phase-A prediction pointer is not frozen")
    prediction_dir = Path(str(pointer["prediction_dir"])).resolve()
    complete_path = prediction_dir / "oracle_free_predictions_complete.json"
    if baseline.sha256_file(complete_path) != expected_complete_sha:
        raise RuntimeError("Phase-A completion hash mismatch; truth access refused")
    complete = read_json(complete_path)
    if complete.get("status") != "PREDICTIONS_FROZEN_BEFORE_TRUTH_ACCESS":
        raise RuntimeError("Phase-A completion status mismatch; truth access refused")
    if bool(complete.get("sealed_truth_accessed", True)) or bool(complete.get("true_mechanism_accessed", True)) or bool(complete.get("true_location_accessed", True)):
        raise RuntimeError("Phase-A privilege boundary violated; truth access refused")
    pred_path = prediction_dir / "oracle_free_predictions.csv"
    if baseline.sha256_file(pred_path) != str(complete["oracle_free_predictions_sha256"]):
        raise RuntimeError("frozen prediction CSV hash mismatch; truth access refused")
    rows = baseline.read_csv(pred_path)
    if len(rows) != int(complete["prediction_count"]):
        raise RuntimeError("frozen prediction row-count mismatch; truth access refused")
    if len({str(row["example_id"]) for row in rows}) != len(rows):
        raise RuntimeError("duplicate frozen prediction id; truth access refused")
    logits = np.asarray(
        [[float(row["logit_0"]), float(row["logit_1"]), float(row["logit_2"])] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("non-finite frozen oracle-free logits; truth access refused")
    return rows, logits, complete


def exact_feature(loaded: Mapping[str, np.ndarray], root: Mapping[str, str]) -> np.ndarray:
    delta, weights, pairs = frame.measured_delta_and_weights(loaded)
    clean = frame.circuit_from_serialized(loaded)
    boundary = int(root["injection_boundary_rank"])
    affected = int(root["affected_qubit"])
    jac = frame.frame_response_jacobian(clean, boundary, affected, pairs)
    feature = frame.canonicalize_evidence(delta, jac, weights)[0]
    if feature.shape != (24,) or not np.all(np.isfinite(feature)):
        raise RuntimeError("exact canonical frame feature contract drift")
    return feature.astype(np.float32)


def build_fit_reference_table(product: Path, progress_every: int) -> dict[str, np.ndarray]:
    cfg = step14.read_json(STEP14_CONFIG)
    step14.assert_contract(cfg)
    cross_rows, _by, _fit_roots, _selection_roots = step14.verify_cross_product(product, cfg)
    manifests = product / "manifests"
    root_rows = baseline.read_csv(manifests / "root_manifest.csv")
    roots = {
        int(row["root_index"]): row
        for row in root_rows
        if str(row.get("step14_partition")) == "fit"
    }
    selected = [
        row for row in cross_rows
        if str(row.get("step14_partition")) == "fit" and str(row.get("mechanism")) in frame.TARGET
    ]
    raw: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    truth: list[int] = []
    families: list[str] = []
    for pos, row in enumerate(selected, start=1):
        root_index = int(row["root_index"])
        root = roots.get(root_index)
        if root is None or str(root.get("step14_partition")) != "fit":
            raise RuntimeError("non-fit root reached FIT-only reference extractor")
        path = product / str(row["artifact_path"])
        if baseline.sha256_file(path) != str(row["artifact_sha256"]):
            raise RuntimeError(f"FIT artifact hash mismatch for {row['example_id']}")
        with np.load(path, allow_pickle=False) as source:
            loaded = {str(k): np.asarray(source[k]) for k in source.files if str(k).startswith("x__")}
        raw.append(oracle.raw_diagnostic_features(loaded)[0].astype(np.float32))
        exact.append(exact_feature(loaded, root))
        truth.append(int(frame.TARGET[str(row["mechanism"])]))
        families.append(str(row["family_id"]))
        if progress_every and pos % progress_every == 0:
            print(f"FIT reference extraction {pos}/{len(selected)}", flush=True)
    y = np.asarray(truth, dtype=np.int64)
    counts = Counter(int(v) for v in y.tolist())
    if len(y) != 28800 or counts != Counter({0: 9600, 1: 9600, 2: 9600}):
        raise RuntimeError(f"FIT reference balance/count drift: n={len(y)} counts={counts}")
    if len(set(families)) != 600:
        raise RuntimeError("FIT family count drift")
    return {
        "raw": np.stack(raw).astype(np.float32),
        "exact": np.stack(exact).astype(np.float32),
        "truth": y,
    }


def build_fresh_reference_table(
    product: Path,
    prediction_rows: Sequence[Mapping[str, str]],
    progress_every: int,
) -> dict[str, Any]:
    # This function is called only after Phase-A hashes have been verified.
    blind_examples = {str(row["example_id"]): row for row in baseline.read_csv(product / "manifests" / "example_manifest.csv")}
    sealed_examples = {str(row["example_id"]): row for row in baseline.read_csv(product / "sealed_truth" / "example_manifest.csv")}
    sealed_roots = {int(row["root_index"]): row for row in baseline.read_csv(product / "sealed_truth" / "root_manifest.csv")}
    raw: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    truth: list[int] = []
    families: list[str] = []
    ids: list[str] = []
    for pos, pred in enumerate(prediction_rows, start=1):
        example_id = str(pred["example_id"])
        blind = blind_examples.get(example_id)
        sealed = sealed_examples.get(example_id)
        if blind is None or sealed is None:
            raise RuntimeError(f"fresh truth/blind join missing for {example_id}")
        if str(blind.get("evaluation_role")) != "target":
            raise RuntimeError("frozen prediction joins to non-target blind row")
        mechanism = str(sealed.get("mechanism"))
        if mechanism not in frame.TARGET:
            raise RuntimeError(f"frozen prediction joins to unexpected mechanism {mechanism}")
        root_index = int(pred["root_index"])
        if int(blind["root_index"]) != root_index or int(sealed["root_index"]) != root_index:
            raise RuntimeError("fresh root-index join drift")
        root = sealed_roots.get(root_index)
        if root is None:
            raise RuntimeError(f"sealed root missing for {root_index}")
        blind_path = product / str(blind["artifact_path"])
        try:
            rel = blind_path.resolve().relative_to(product.resolve())
        except ValueError as exc:
            raise RuntimeError("blind artifact escaped fresh product") from exc
        if not rel.parts or rel.parts[0] != "blind_artifacts" or "sealed_truth" in rel.parts:
            raise RuntimeError("confirmatory measurements must come from blind x-only artifacts")
        loaded = load_x_only(blind_path, str(blind["artifact_sha256"]), example_id)
        raw.append(oracle.raw_diagnostic_features(loaded)[0].astype(np.float32))
        exact.append(exact_feature(loaded, root))
        truth.append(int(frame.TARGET[mechanism]))
        families.append(str(sealed["family_id"]))
        ids.append(example_id)
        if progress_every and pos % progress_every == 0:
            print(f"fresh reference extraction {pos}/{len(prediction_rows)}", flush=True)
    y = np.asarray(truth, dtype=np.int64)
    counts = Counter(int(v) for v in y.tolist())
    if len(y) != 7200 or counts != Counter({0: 2400, 1: 2400, 2: 2400}):
        raise RuntimeError(f"fresh mechanism balance/count drift: n={len(y)} counts={counts}")
    if len(set(families)) != 150:
        raise RuntimeError("fresh family count drift")
    return {
        "raw": np.stack(raw).astype(np.float32),
        "exact": np.stack(exact).astype(np.float32),
        "truth": y,
        "family": np.asarray(families, dtype=object),
        "example_id": ids,
    }


def ensemble_probe(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_fresh: np.ndarray,
    y_fresh: np.ndarray,
    seeds: Sequence[int],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    parts: list[np.ndarray] = []
    records: dict[str, Any] = {}
    for seed in seeds:
        record = oracle.fit_probe(
            x_fit, y_fit, x_fresh, y_fresh,
            seed=int(seed), high_capacity=False, device=device,
        )
        parts.append(np.asarray(record.pop("selection_logits"), dtype=np.float64))
        records[str(seed)] = record
        print(
            f"reference probe seed={seed} fit_BA={record['fit_balanced_accuracy']:.6f} fresh_BA={record['selection_balanced_accuracy']:.6f}",
            flush=True,
        )
    return np.mean(np.stack(parts), axis=0), records


def main() -> None:
    args = parse_args()
    oracle_compat.apply_frozen_support_bound()
    confirm = read_json(CONFIRM_CONFIG)
    freeze = read_json(FINAL_FREEZE)
    if confirm.get("status") != "FROZEN_AFTER_HOLDOUT_GENERATION_BEFORE_MODEL_INFERENCE":
        raise RuntimeError("confirmatory protocol drift")
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
        raise RuntimeError("fresh holdout dataset identity drift")
    rep14.verify_training_freeze(args.training_run_id, args.selection_freeze_sha256)

    # HARD BARRIER: every Phase-A object is verified before the first sealed-truth read below.
    prediction_pointer = read_json(PREDICTION_PARENT / "current_oracle_free_predictions.json")
    if str(prediction_pointer.get("oracle_free_predictions_complete_sha256")) != args.oracle_free_prediction_complete_sha256:
        raise RuntimeError("requested Phase-A completion does not match current frozen pointer; truth access refused")
    pred_rows, equiv_logits, prediction_complete = prediction_rows_and_logits(
        prediction_pointer, args.oracle_free_prediction_complete_sha256
    )
    if str(prediction_complete["fresh_holdout_product_id"]) != args.fresh_holdout_product_id:
        raise RuntimeError("Phase-A holdout identity mismatch; truth access refused")
    if str(prediction_complete["fresh_holdout_dataset_complete_sha256"]) != args.fresh_holdout_dataset_complete_sha256:
        raise RuntimeError("Phase-A holdout hash mismatch; truth access refused")
    if str(prediction_complete["method_id"]) != str(method["method_id"]):
        raise RuntimeError("Phase-A method identity mismatch; truth access refused")
    if len(pred_rows) != int(hold["target_example_count"]):
        raise RuntimeError("Phase-A target count mismatch; truth access refused")
    print("PHASE-A PREDICTION FREEZE VERIFIED — SEALED TRUTH ACCESS NOW PERMITTED", flush=True)

    holdout_pointer = read_json(HOLDOUT_PARENT / "current_fresh_holdout.json")
    product = Path(str(holdout_pointer["product_dir"])).resolve()
    if baseline.sha256_file(product / "dataset_complete.json") != args.fresh_holdout_dataset_complete_sha256:
        raise RuntimeError("fresh dataset completion bytes drift")

    fresh = build_fresh_reference_table(product, pred_rows, args.progress_every)
    if fresh["example_id"] != [str(row["example_id"]) for row in pred_rows]:
        raise RuntimeError("frozen prediction / truth alignment drift")
    y_fresh = fresh["truth"]
    groups = fresh["family"]
    equiv_metric = oracle.metric_record(y_fresh, equiv_logits)

    cfg = step14.read_json(STEP14_CONFIG)
    step14.assert_contract(cfg)
    development = step14.resolve_cross_product(None)
    dev_complete = read_json(development / "dataset_complete.json")
    if str(dev_complete["product_id"]) != str(source["development_product_id"]):
        raise RuntimeError("development product identity drift")
    if baseline.sha256_file(development / "dataset_complete.json") != str(source["development_dataset_sha256"]):
        raise RuntimeError("development dataset completion hash drift")
    fit = build_fit_reference_table(development, args.progress_every)
    device = resolve_device(args.device)
    seeds = [int(v) for v in confirm["phase_b_confirmatory_scoring"]["raw_baseline"]["probe_seeds"]]
    if seeds != [6101, 6102, 6103]:
        raise RuntimeError("frozen reference probe seed drift")

    raw_logits, raw_training = ensemble_probe(fit["raw"], fit["truth"], fresh["raw"], y_fresh, seeds, device)
    exact_logits, exact_training = ensemble_probe(fit["exact"], fit["truth"], fresh["exact"], y_fresh, seeds, device)
    raw_metric = oracle.metric_record(y_fresh, raw_logits)
    exact_metric = oracle.metric_record(y_fresh, exact_logits)

    stats = confirm["statistics_and_support_gates"]
    if int(oracle.PROBE_SPEC["bootstrap_replicates"]) != int(stats["bootstrap_replicates"]):
        raise RuntimeError("bootstrap replicate drift")
    paired = oracle.bootstrap_delta(
        y_fresh, equiv_logits, raw_logits, groups, seed=int(stats["bootstrap_seed"])
    )
    equiv_ba = float(equiv_metric["mechanism_balanced_accuracy"])
    raw_ba = float(raw_metric["mechanism_balanced_accuracy"])
    exact_ba = float(exact_metric["mechanism_balanced_accuracy"])
    denom = exact_ba - raw_ba
    gain_recovery = None if denom <= 0.0 else float((equiv_ba - raw_ba) / denom)
    gates = {
        "mechanism_balanced_accuracy": equiv_ba >= float(stats["minimum_mechanism_balanced_accuracy"]),
        "minimum_mechanism_recall": float(equiv_metric["minimum_mechanism_recall"]) >= float(stats["minimum_minimum_mechanism_recall"]),
        "paired_ba_gain_over_raw": float(paired["mean_delta"]) >= float(stats["minimum_paired_ba_gain_over_raw"]),
        "paired_gain_bootstrap_ci_lower_positive": float(paired["bootstrap_ci"][0]) > 0.0,
        "fraction_oracle_canonicalization_gain_recovered": gain_recovery is not None and gain_recovery >= float(stats["minimum_fraction_of_oracle_canonicalization_gain_recovered"]),
    }
    full = all(bool(v) for v in gates.values())
    interpretation = confirm["interpretation"]
    if full:
        verdict = str(interpretation["full_support"])
    elif float(paired["bootstrap_ci"][0]) > 0.0 and equiv_ba >= 0.60:
        verdict = str(interpretation["partial_support"])
    else:
        verdict = str(interpretation["failure"])

    confirm_sha = baseline.sha256_file(CONFIRM_CONFIG)
    result_id = "confirmatory_" + hashlib.sha256(
        (confirm_sha + args.oracle_free_prediction_complete_sha256 + args.fresh_holdout_dataset_complete_sha256).encode()
    ).hexdigest()[:24]
    output_parent = args.output_parent.expanduser().resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    out_dir = output_parent / result_id
    if out_dir.exists() and any(out_dir.iterdir()):
        existing = read_json(out_dir / "confirmatory_complete.json")
        print(json.dumps(existing, indent=2, sort_keys=True)); return
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_ONE_SHOT_FRESH_HOLDOUT_CONFIRMATORY_EVALUATION",
        "verdict": verdict,
        "full_support_gates_all_passed": bool(full),
        "identity": {
            "confirmatory_protocol_sha256": confirm_sha,
            "final_method_freeze_payload_sha256": str(method["final_method_freeze_payload_sha256"]),
            "fresh_holdout_product_id": args.fresh_holdout_product_id,
            "fresh_holdout_dataset_complete_sha256": args.fresh_holdout_dataset_complete_sha256,
            "oracle_free_prediction_id": str(prediction_complete["prediction_id"]),
            "oracle_free_predictions_sha256": str(prediction_complete["oracle_free_predictions_sha256"]),
            "oracle_free_predictions_complete_sha256": args.oracle_free_prediction_complete_sha256,
            "method_id": str(method["method_id"]),
            "tau": float(method["tau"]),
            "frame_temperature": float(method["frame_temperature"]),
        },
        "oracle_free_equivalence_aware": equiv_metric,
        "fit_only_raw_baseline": raw_metric,
        "fit_only_privileged_exact_local_frame_reference": exact_metric,
        "paired_equivalence_aware_minus_raw": paired,
        "fraction_oracle_canonicalization_gain_recovered": gain_recovery,
        "support_gates": gates,
        "reference_probe_training": {
            "raw": raw_training,
            "privileged_exact_local_frame": exact_training,
            "training_partition": "fit",
            "fresh_holdout_used_for_training_or_tuning": False,
        },
        "scientific_boundaries": {
            "oracle_free_predictions_frozen_before_truth_access": True,
            "fresh_truth_accessed_only_after_prediction_hash_verification": True,
            "fresh_measurement_features_read_from_blind_x_only_artifacts": True,
            "true_mechanism_used_only_for_scoring": True,
            "true_location_used_only_for_privileged_reference": True,
            "method_changed_after_freeze": False,
            "selection_metrics_used": False,
            "selection_artifact_contents_loaded_for_reference_training": False,
            "simulator_outer_accessed": False,
            "future_hardware_reserve_accessed": False,
            "qpu_executed": False,
            "physical_hardware_claim_allowed": False,
            "causal_claim_that_boundary_ambiguity_is_the_only_failure_source_allowed": False,
        },
    }
    result_path = out_dir / "confirmatory_result.json"
    atomic_json(result_path, result)
    result_sha = baseline.sha256_file(result_path)
    complete = {
        "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_confirmatory_complete.v1",
        "status": "COMPLETE_ONE_SHOT_FRESH_HOLDOUT_CONFIRMATORY_EVALUATION",
        "result_id": result_id,
        "result_dir": str(out_dir),
        "verdict": verdict,
        "confirmatory_result_sha256": result_sha,
        "oracle_free_predictions_complete_sha256": args.oracle_free_prediction_complete_sha256,
        "fresh_holdout_dataset_complete_sha256": args.fresh_holdout_dataset_complete_sha256,
        "full_support_gates_all_passed": bool(full),
        "simulator_outer_accessed": False,
        "future_hardware_reserve_accessed": False,
        "qpu_executed": False,
    }
    complete_path = out_dir / "confirmatory_complete.json"
    atomic_json(complete_path, complete)
    complete_sha = baseline.sha256_file(complete_path)
    atomic_json(output_parent / "current_confirmatory_result.json", {
        "schema": "triqto.v0_2.step14_equivalence_aware_fresh_holdout_confirmatory_pointer.v1",
        "status": "COMPLETE_ONE_SHOT_FRESH_HOLDOUT_CONFIRMATORY_EVALUATION",
        "result_id": result_id,
        "result_dir": str(out_dir),
        "verdict": verdict,
        "confirmatory_result_sha256": result_sha,
        "confirmatory_complete_sha256": complete_sha,
        "oracle_free_predictions_complete_sha256": args.oracle_free_prediction_complete_sha256,
    })
    print(json.dumps({**complete, "confirmatory_complete_sha256": complete_sha, "metrics": {
        "equivalence_aware_ba": equiv_ba,
        "raw_ba": raw_ba,
        "exact_oracle_ba": exact_ba,
        "equivalence_minus_raw_ba": float(paired["mean_delta"]),
        "paired_ci": paired["bootstrap_ci"],
        "gain_recovery": gain_recovery,
        "minimum_recall": float(equiv_metric["minimum_mechanism_recall"]),
        "gates": gates,
    }}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
