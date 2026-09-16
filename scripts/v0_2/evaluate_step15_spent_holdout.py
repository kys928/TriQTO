#!/usr/bin/env python3
"""Step-7 audit: evaluate the frozen Step-15 rank adapter on the spent holdout.

This is an audit of an already-spent Step-14 holdout, not reconfirmation. The
adapter must already be immutable and hash-verified. No parameter, threshold,
feature, or hyperparameter is selected or modified by this executable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Mapping

import numpy as np
import torch

import analyze_step14_fresh_holdout_residual_bottleneck as residual
import analyze_step14_latent_frame_inference as latent
import analyze_step14_oracle_raw_evidence_ceiling as oracle
import benchmark_step6_cheap_baselines as baseline
import score_step14_equivalence_aware_fresh_holdout_confirmatory as confirm_scorer
import step14_equivalence_aware_common as equiv
import step15_frame_ranker_common as ranker

ROOT = Path(__file__).resolve().parents[2]
STEP15_CONFIG = ROOT / "configs/v0_2/step15_frame_ranking_development.json"
STEP14_POSTHOC_CONFIG = ROOT / "configs/v0_2/step14_fresh_holdout_residual_bottleneck_posthoc.json"
STEP15_PARENT = Path("/workspace/triqto-data/step15_frame_ranking")
HOLDOUT_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout")
METHOD_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_development")
CONFIRM_PARENT = Path("/workspace/triqto-data/step14_equivalence_aware_fresh_holdout_confirmation")
OUTPUT_PARENT = Path("/workspace/triqto-data/step15_spent_holdout_audit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step15-parent", type=Path, default=STEP15_PARENT)
    p.add_argument("--output-parent", type=Path, default=OUTPUT_PARENT)
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as h:
        json.dump(value, h, indent=2, sort_keys=True, allow_nan=False)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write empty Step-15 audit CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def verify_step15_freeze(parent: Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    pointer = read_json(parent / "current_step15_frame_ranker.json")
    method_dir = Path(str(pointer["method_dir"])).resolve()
    files = {
        "frame_rank_adapter.json": str(pointer["adapter_sha256"]),
        "fit_development_result.json": str(pointer["fit_development_result_sha256"]),
        "adapter_freeze.json": str(pointer["adapter_freeze_sha256"]),
        "step15_complete.json": str(pointer["step15_complete_sha256"]),
    }
    for name, expected in files.items():
        got = baseline.sha256_file(method_dir / name)
        if got != expected:
            raise RuntimeError(f"Step-15 pre-audit hash mismatch for {name}")
    adapter_payload = read_json(method_dir / "frame_rank_adapter.json")
    freeze = read_json(method_dir / "adapter_freeze.json")
    complete = read_json(method_dir / "step15_complete.json")
    if freeze.get("status") != "IMMUTABLY_FROZEN_BEFORE_SPENT_HOLDOUT_EVALUATION":
        raise RuntimeError("Step-15 adapter was not frozen before audit")
    if complete.get("status") != "HASH_VERIFIED_READY_FOR_SPENT_HOLDOUT_AUDIT":
        raise RuntimeError("Step-15 completion manifest is not audit-ready")
    if bool(freeze.get("spent_holdout_accessed")) or bool(complete.get("spent_holdout_accessed")):
        raise RuntimeError("Step-15 freeze indicates premature spent-holdout access")
    frozen_copy = dict(freeze)
    recorded = str(frozen_copy.pop("freeze_payload_sha256"))
    if canonical_sha(frozen_copy) != recorded:
        raise RuntimeError("Step-15 freeze payload canonical hash mismatch")
    return method_dir, pointer, adapter_payload, freeze


def build_rank_feature_batch(table: Mapping[str, Any], tau: float, temperature: float) -> np.ndarray:
    n, max_candidates, _ = table["x"].shape
    out = np.zeros((n, max_candidates, len(ranker.FEATURE_NAMES)), dtype=np.float32)
    for i in range(n):
        count = int(table["mask"][i].sum())
        out[i, :count] = ranker.build_rank_features_one(
            table["x"][i, :count],
            table["profiles"][i, :count],
            table["distance"][i, :count, :count],
            table["gate"][i, :count, :count],
            table["candidate_sets"][i],
            qubit_count=int(table["qubit_count"][i]),
            gate_count=int(table["gate_count"][i]),
            tau=tau,
            frame_temperature=temperature,
        )
    return out


def mechanism_calibration(y: np.ndarray, logits: np.ndarray, bins: int = 10) -> dict[str, float]:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1.0e-300)
    yy = np.asarray(y, dtype=np.int64)
    nll = float(np.mean(-np.log(np.maximum(p[np.arange(len(yy)), yy], 1.0e-300))))
    target = np.eye(p.shape[1], dtype=np.float64)[yy]
    brier = float(np.mean(np.sum(np.square(p - target), axis=1)))
    confidence = np.max(p, axis=1)
    pred = np.argmax(p, axis=1)
    correct = pred == yy
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        member = (confidence >= lo) & ((confidence < hi) if b < bins - 1 else (confidence <= hi))
        if np.any(member):
            ece += float(np.mean(member)) * abs(float(np.mean(correct[member])) - float(np.mean(confidence[member])))
    return {"nll": nll, "brier": brier, "ece": float(ece)}


def metric(y: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    out = oracle.metric_record(y, logits)
    out["calibration"] = mechanism_calibration(y, logits)
    return out


def main() -> None:
    args = parse_args()
    cfg = read_json(STEP15_CONFIG)
    posthoc_cfg = read_json(STEP14_POSTHOC_CONFIG)
    _method_dir, step15_pointer, adapter_payload, freeze = verify_step15_freeze(args.step15_parent)
    adapter = ranker.adapter_from_json(adapter_payload["adapter"])
    method_id = str(step15_pointer["method_id"])

    src = posthoc_cfg["source_identity"]
    pred_pointer = read_json(CONFIRM_PARENT / "current_oracle_free_predictions.json")
    pred_rows, frozen_logits, pred_complete = confirm_scorer.prediction_rows_and_logits(
        pred_pointer, str(src["oracle_free_predictions_complete_sha256"])
    )
    if str(pred_complete["oracle_free_predictions_sha256"]) != str(src["oracle_free_predictions_sha256"]):
        raise RuntimeError("Step-14 frozen prediction CSV identity drift")

    hold_pointer = read_json(HOLDOUT_PARENT / "current_fresh_holdout.json")
    product = Path(str(hold_pointer["product_dir"])).resolve()
    if str(hold_pointer["product_id"]) != str(src["fresh_holdout_product_id"]):
        raise RuntimeError("spent holdout product identity drift")
    if baseline.sha256_file(product / "dataset_complete.json") != str(src["fresh_holdout_dataset_complete_sha256"]):
        raise RuntimeError("spent holdout dataset-complete hash drift")

    latent_cfg = latent.load_frozen_config()
    table = residual.extract_fresh_table(
        product,
        pred_rows,
        latent_cfg,
        posthoc_cfg["response_equivalence"],
        args.progress_every,
    )
    y = table["truth"]
    families = table["family"]
    device = resolve_device(args.device)

    source15 = cfg["source_identity"]
    tau = float(source15["step14_tau"])
    temperature = float(source15["step14_frame_temperature"])
    if abs(tau - float(adapter_payload["step14_tau"])) > 0.0 or abs(temperature - float(adapter_payload["step14_frame_temperature"])) > 0.0:
        raise RuntimeError("Step-15 adapter/frozen Step-14 pooling hyperparameter drift")

    method_pointer = read_json(METHOD_PARENT / "current_equivalence_aware_fit.json")
    if str(method_pointer["method_id"]) != str(source15["step14_method_id"]):
        raise RuntimeError("Step-14 method pointer drift at Step-7 audit")
    step14_method_dir = Path(str(method_pointer["method_dir"])).resolve()
    if baseline.sha256_file(step14_method_dir / "method_bundle.json") != str(source15["step14_method_bundle_sha256"]):
        raise RuntimeError("Step-14 method bundle hash drift at Step-7 audit")

    seed_candidate_logits: list[np.ndarray] = []
    for seed in [int(v) for v in source15["candidate_network_seeds"]]:
        name = f"candidate_network_seed{seed}.pt"
        path = step14_method_dir / name
        if baseline.sha256_file(path) != str(source15["candidate_network_checkpoint_sha256"][name]):
            raise RuntimeError(f"Step-14 checkpoint hash drift at Step-7 audit: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _uniform, candidates = equiv.infer_candidate_network(
            table["x"],
            table["mask"],
            state_dict=payload["state_dict"],
            mean=np.asarray(payload["normalization_mean"], dtype=np.float32),
            std=np.asarray(payload["normalization_std"], dtype=np.float32),
            latent_cfg=latent_cfg,
            device=device,
        )
        seed_candidate_logits.append(candidates)
        print(f"Step-7 frozen Step-14 candidate network seed={seed} replay complete", flush=True)
    candidate_logits = np.mean(np.stack(seed_candidate_logits), axis=0)

    reproduced_step14 = equiv.equivalence_aware_pool_batch(
        candidate_logits,
        table["profiles"],
        table["distance"],
        table["gate"],
        table["mask"],
        tau=tau,
        frame_temperature=temperature,
    )
    reproduction = residual.audit_frozen_logit_reproduction(frozen_logits, reproduced_step14)

    rank_features = build_rank_feature_batch(table, tau, temperature)
    baseline_scores = np.asarray(rank_features[:, :, 0], dtype=np.float64)
    baseline_scores[~table["mask"]] = -np.inf
    step15_scores = ranker.adapter_scores(rank_features, table["mask"], adapter)
    baseline_ranking = ranker.ranking_metrics(
        baseline_scores, table["mask"], table["equivalence_mask"]
    )
    step15_ranking = ranker.ranking_metrics(
        step15_scores, table["mask"], table["equivalence_mask"]
    )

    step15_logits = ranker.equivalence_aware_pool_batch_with_adapter(
        candidate_logits,
        table["profiles"],
        table["distance"],
        table["gate"],
        table["mask"],
        rank_features,
        adapter,
        tau=tau,
        frame_temperature=temperature,
    )

    original_metric = metric(y, frozen_logits)
    step15_metric = metric(y, step15_logits)
    original_ba = float(original_metric["mechanism_balanced_accuracy"])
    if abs(original_ba - float(src["confirmatory_oracle_free_ba"])) > 1.0e-12:
        raise RuntimeError("original frozen Step-14 BA no longer reproduces the confirmatory source")
    step15_ba = float(step15_metric["mechanism_balanced_accuracy"])
    ba_delta = step15_ba - original_ba
    paired = oracle.bootstrap_delta(
        y,
        step15_logits,
        frozen_logits,
        families,
        seed=2026091601,
    )

    baseline_order = np.argsort(-baseline_scores, axis=1, kind="stable")
    step15_order = np.argsort(-step15_scores, axis=1, kind="stable")
    rows: list[dict[str, Any]] = []
    for i, example_id in enumerate(table["example_id"]):
        count = int(table["mask"][i].sum())
        eq = np.asarray(table["equivalence_mask"][i, :count], dtype=np.bool_)
        base_order = baseline_order[i, :count]
        new_order = step15_order[i, :count]
        base_rank = int(np.flatnonzero(eq[base_order])[0]) + 1
        new_rank = int(np.flatnonzero(eq[new_order])[0]) + 1
        rows.append({
            "example_id": str(example_id),
            "family_id": str(families[i]),
            "truth": int(y[i]),
            "step14_pred": int(np.argmax(frozen_logits[i])),
            "step15_pred": int(np.argmax(step15_logits[i])),
            "step14_correct": int(np.argmax(frozen_logits[i]) == y[i]),
            "step15_correct": int(np.argmax(step15_logits[i]) == y[i]),
            "best_equivalent_rank_step14": base_rank,
            "best_equivalent_rank_step15": new_rank,
            "candidate_count": count,
            "equivalence_size": int(np.sum(eq)),
        })

    result = {
        "schema": "triqto.v0_2.step15_spent_holdout_audit_result.v1",
        "status": "COMPLETE_SPENT_HOLDOUT_AUDIT_NOT_RECONFIRMATION",
        "interpretation": {
            "holdout_status": "spent Step-14 fresh holdout",
            "confirmatory_claim": False,
            "adapter_or_hyperparameter_selection_performed": False,
            "result_may_be_used_as_posthoc_causal_test_of_frame_ranking_intervention": True,
        },
        "identity": {
            "step15_method_id": method_id,
            "step15_adapter_sha256": str(step15_pointer["adapter_sha256"]),
            "step15_freeze_sha256": str(step15_pointer["adapter_freeze_sha256"]),
            "step14_method_id": str(source15["step14_method_id"]),
            "spent_holdout_product_id": str(src["fresh_holdout_product_id"]),
            "spent_holdout_dataset_complete_sha256": str(src["fresh_holdout_dataset_complete_sha256"]),
            "frozen_step14_predictions_sha256": str(src["oracle_free_predictions_sha256"]),
        },
        "pre_audit_hash_verification": {
            "freeze_payload_sha256": str(freeze["freeze_payload_sha256"]),
            "adapter_hash_verified": True,
            "freeze_hash_verified": True,
            "step14_checkpoint_hashes_verified": True,
            "step14_frozen_logit_reproduction": reproduction,
        },
        "ranking": {
            "step14_frozen": baseline_ranking,
            "step15_rank_adapter": step15_ranking,
            "delta": {key: float(step15_ranking[key] - baseline_ranking[key]) for key in baseline_ranking},
        },
        "downstream_mechanism": {
            "step14_original_frozen": original_metric,
            "step15_rank_adapter": step15_metric,
            "balanced_accuracy_delta": float(ba_delta),
            "paired_family_bootstrap_step15_minus_step14": paired,
        },
        "example_count": int(len(y)),
        "family_count": int(len(set(families.tolist()))),
    }

    audit_key = hashlib.sha256(
        (method_id + str(src["fresh_holdout_dataset_complete_sha256"]) + str(step15_pointer["adapter_sha256"])).encode("utf-8")
    ).hexdigest()[:24]
    audit_id = f"step15_spent_{audit_key}"
    output_dir = args.output_parent / audit_id
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise RuntimeError(f"refusing to overwrite immutable Step-15 spent-holdout audit: {output_dir}")
        output_dir.rmdir()
    output_dir.mkdir(parents=True, exist_ok=False)

    result_path = output_dir / "spent_holdout_audit_result.json"
    atomic_json(result_path, result)
    per_path = output_dir / "per_example_spent_holdout_audit.csv"
    atomic_csv(per_path, rows)
    result_sha = baseline.sha256_file(result_path)
    per_sha = baseline.sha256_file(per_path)
    complete = {
        "schema": "triqto.v0_2.step15_spent_holdout_audit_complete.v1",
        "status": "COMPLETE_SPENT_HOLDOUT_AUDIT_NOT_RECONFIRMATION",
        "audit_id": audit_id,
        "audit_dir": str(output_dir),
        "step15_method_id": method_id,
        "spent_holdout_audit_result_sha256": result_sha,
        "per_example_spent_holdout_audit_sha256": per_sha,
        "adapter_sha256": str(step15_pointer["adapter_sha256"]),
        "adapter_freeze_sha256": str(step15_pointer["adapter_freeze_sha256"]),
        "balanced_accuracy_step14": original_ba,
        "balanced_accuracy_step15": step15_ba,
        "balanced_accuracy_delta": float(ba_delta),
        "confirmatory_claim": False,
    }
    complete_path = output_dir / "spent_holdout_audit_complete.json"
    atomic_json(complete_path, complete)
    complete_sha = baseline.sha256_file(complete_path)
    atomic_json(
        args.output_parent / "current_step15_spent_holdout_audit.json",
        {
            "schema": "triqto.v0_2.step15_spent_holdout_audit_pointer.v1",
            "audit_id": audit_id,
            "audit_dir": str(output_dir),
            "spent_holdout_audit_result_sha256": result_sha,
            "per_example_spent_holdout_audit_sha256": per_sha,
            "spent_holdout_audit_complete_sha256": complete_sha,
            "confirmatory_claim": False,
        },
    )

    print(json.dumps({
        "status": complete["status"],
        "audit_id": audit_id,
        "ranking_step14": baseline_ranking,
        "ranking_step15": step15_ranking,
        "balanced_accuracy_step14": original_ba,
        "balanced_accuracy_step15": step15_ba,
        "balanced_accuracy_delta": ba_delta,
        "paired_family_bootstrap": paired,
        "spent_holdout_audit_result_sha256": result_sha,
        "spent_holdout_audit_complete_sha256": complete_sha,
        "confirmatory_claim": False,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
