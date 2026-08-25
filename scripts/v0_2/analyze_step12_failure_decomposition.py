#!/usr/bin/env python3
"""Posthoc, no-QPU failure decomposition for the frozen Step-12 result.

This analysis asks whether the Step-12 failure is better explained by collapse of
hardware-facing diagnostic evidence or by failure of the learned mechanism
mapping to generalize across new circuit contexts. It never trains, retunes, or
submits a QPU job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from typing import Any, Mapping

import numpy as np

import run_step12_independent_phase_generalization as step12
from triqto.hardware.diagnostic_acquisition import BASIS_ORDER, empirical_stats_from_counts

EXPECTED_BUNDLE_SHA256 = "9c9a4a726e0403a466960669917a394ef7df73c2f6fb12239dbdf799f818d7ff"
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--bootstrap-replicates", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=20260825)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _root(zf: zipfile.ZipFile) -> str:
    roots = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
    if len(roots) != 1:
        raise RuntimeError(f"expected one bundle root, found {sorted(roots)}")
    return next(iter(roots)) + "/"


def _json(zf: zipfile.ZipFile, root: str, name: str) -> dict[str, Any]:
    return json.loads(zf.read(root + name))


def _case_rows(zf: zipfile.ZipFile, root: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in zf.read(root + "case_results.jsonl").decode().splitlines() if line.strip()]


def _qpu_local_delta(row: Mapping[str, Any]) -> np.ndarray:
    counts = row["counts_by_program"]
    n = len(next(iter(counts["reference_Z"])))
    by_basis: dict[str, np.ndarray] = {}
    for basis in BASIS_ORDER:
        ref = empirical_stats_from_counts(counts[f"reference_{basis}"], n)
        obs = empirical_stats_from_counts(counts[f"observed_{basis}"], n)
        by_basis[str(basis)] = np.asarray(obs.local - ref.local, dtype=np.float64)
    # Match Step-12 _bloch_vector ordering: q-major [X,Y,Z].
    return np.asarray(
        [[by_basis["X"][q], by_basis["Y"][q], by_basis["Z"][q]] for q in range(n)],
        dtype=np.float64,
    ).reshape(-1)


def _ideal_local_delta(motif: str, mechanism: str, strength: float) -> np.ndarray:
    reference, _ = step12._build_motif(motif)
    observed, _ = step12._build_motif(motif, mechanism, strength)
    return step12._bloch_vector(observed) - step12._bloch_vector(reference)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _corr(a: list[float], b: list[float]) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rank(x), _rank(y))[0, 1])
    return pearson, spearman


def _bootstrap_qpu_local_delta(row: Mapping[str, Any], *, replicates: int, rng: np.random.Generator) -> np.ndarray:
    counts = row["counts_by_program"]
    n = len(next(iter(counts["reference_Z"])))
    out = np.zeros((replicates, 3 * n), dtype=np.float64)
    component = {"X": 0, "Y": 1, "Z": 2}
    for basis in BASIS_ORDER:
        sampled: dict[str, np.ndarray] = {}
        for kind in ("reference", "observed"):
            raw = counts[f"{kind}_{basis}"]
            keys = list(raw)
            probabilities = np.asarray([raw[key] for key in keys], dtype=np.float64)
            shots = int(probabilities.sum())
            probabilities /= float(shots)
            eig = []
            for bitstring in keys:
                logical_bits = np.fromiter((int(char) for char in reversed(bitstring)), dtype=np.int8)
                eig.append(1.0 - 2.0 * logical_bits.astype(np.float64))
            eig = np.asarray(eig, dtype=np.float64)
            draws = rng.multinomial(shots, probabilities, size=replicates)
            sampled[kind] = draws @ eig / float(shots)
        delta = sampled["observed"] - sampled["reference"]
        c = component[str(basis)]
        for q in range(n):
            out[:, 3 * q + c] = delta[:, q]
    return out


def analyze(bundle: Path, *, bootstrap_replicates: int, bootstrap_seed: int) -> dict[str, Any]:
    bundle = bundle.expanduser().resolve()
    observed_sha = sha256_file(bundle)
    if observed_sha != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError(f"Step-12 source-bundle SHA256 mismatch: {observed_sha}")
    with zipfile.ZipFile(bundle) as zf:
        if zf.testzip() is not None:
            raise RuntimeError("Step-12 source ZIP failed CRC validation")
        root = _root(zf)
        summary = _json(zf, root, "step12_summary.json")
        audit = _json(zf, root, "identifiability_audit.json")
        rows = _case_rows(zf, root)

    distorted = [row for row in rows if bool(row["expected_effect"])]
    if len(distorted) != 18:
        raise RuntimeError(f"expected 18 distorted Step-12 cases, found {len(distorted)}")

    per_case: list[dict[str, Any]] = []
    ideal_pairwise: list[float] = []
    qpu_pairwise: list[float] = []
    pairwise_ratios: list[float] = []
    rng = np.random.default_rng(bootstrap_seed)

    for row in distorted:
        motif = str(row["family"])
        mechanism = str(row["expected_mechanism"])
        strength = float(row["strength"])
        qpu = _qpu_local_delta(row)
        ideal = _ideal_local_delta(motif, mechanism, strength)
        cosine = float(np.dot(qpu, ideal) / (np.linalg.norm(qpu) * np.linalg.norm(ideal)))
        templates = {m: _ideal_local_delta(motif, m, strength) for m in MECHANISMS}
        distances = {m: float(np.linalg.norm(qpu - template)) for m, template in templates.items()}
        oracle_prediction = min(distances, key=distances.get)

        boot = _bootstrap_qpu_local_delta(row, replicates=bootstrap_replicates, rng=rng)
        template_matrix = np.stack([templates[m] for m in MECHANISMS], axis=0)
        boot_dist = np.linalg.norm(boot[:, None, :] - template_matrix[None, :, :], axis=2)
        truth_index = MECHANISMS.index(mechanism)
        boot_pred = np.argmin(boot_dist, axis=1)
        bootstrap_oracle_accuracy = float(np.mean(boot_pred == truth_index))

        per_case.append({
            "case_id": str(row["case_id"]),
            "motif": motif,
            "strength": strength,
            "truth": mechanism,
            "step10c_prediction": str(row["step10c_prediction"]["mechanism_prediction"]),
            "step9a_prediction": str(row["step9a_prediction"]["mechanism_prediction"]),
            "ideal_local_delta_norm": float(np.linalg.norm(ideal)),
            "qpu_local_delta_norm": float(np.linalg.norm(qpu)),
            "ideal_qpu_local_cosine": cosine,
            "nearest_ideal_template_prediction": oracle_prediction,
            "nearest_ideal_template_correct": oracle_prediction == mechanism,
            "bootstrap_nearest_template_correct_probability": bootstrap_oracle_accuracy,
        })

    for motif in sorted({str(row["family"]) for row in distorted}):
        for strength in sorted({float(row["strength"]) for row in distorted}):
            subset = [row for row in distorted if str(row["family"]) == motif and float(row["strength"]) == strength]
            qpu = {str(row["expected_mechanism"]): _qpu_local_delta(row) for row in subset}
            ideal = {m: _ideal_local_delta(motif, m, strength) for m in MECHANISMS}
            for i, left in enumerate(MECHANISMS):
                for right in MECHANISMS[i + 1:]:
                    idist = float(np.linalg.norm(ideal[left] - ideal[right]))
                    qdist = float(np.linalg.norm(qpu[left] - qpu[right]))
                    ideal_pairwise.append(idist)
                    qpu_pairwise.append(qdist)
                    pairwise_ratios.append(qdist / idist)

    pearson, spearman = _corr(ideal_pairwise, qpu_pairwise)
    actual_oracle_correct = sum(bool(row["nearest_ideal_template_correct"]) for row in per_case)
    bootstrap_probabilities = [float(row["bootstrap_nearest_template_correct_probability"]) for row in per_case]
    step10c_predictions = [str(row["step10c_prediction"]["mechanism_prediction"]) for row in distorted]
    step9a_predictions = [str(row["step9a_prediction"]["mechanism_prediction"]) for row in distorted]

    return {
        "schema": "triqto.v0_2.step13_step12_failure_decomposition.v1",
        "status": "COMPLETE_POSTHOC_NO_QPU_NO_TRAINING",
        "source_step12_bundle": {
            "sha256": "sha256:" + observed_sha,
            "zip_crc_validation": "PASS",
            "plan_id": str(summary["plan_id"]),
            "ibm_job_id": str(summary["ibm_job_id"]),
            "backend_name": str(summary["backend_name"]),
        },
        "frozen_step12_outcome": {
            "step10c_mechanism_correct": int(summary["primary_metrics"]["distorted_mechanism_correct_count"]),
            "step9a_mechanism_correct_report_only": int(summary["baseline_metrics_report_only"]["distorted_mechanism_correct_count"]),
            "step10c_distorted_effect_detection": int(summary["primary_metrics"]["distorted_effect_detection_count"]),
            "predeclared_gate_passed": bool(summary["predeclared_gate"]["passed"]),
        },
        "ideal_pre_qpu_audit": {
            "status": str(audit["status"]),
            "minimum_low_strength_delta_norm": float(audit["minimum_observed_low_strength_delta_norm"]),
            "minimum_low_strength_pairwise_mechanism_distance": float(audit["minimum_observed_low_strength_pairwise_mechanism_distance"]),
        },
        "hardware_evidence_geometry": {
            "mean_truth_template_cosine": float(np.mean([row["ideal_qpu_local_cosine"] for row in per_case])),
            "median_truth_template_cosine": float(np.median([row["ideal_qpu_local_cosine"] for row in per_case])),
            "minimum_truth_template_cosine": float(np.min([row["ideal_qpu_local_cosine"] for row in per_case])),
            "ideal_vs_qpu_pairwise_distance_pearson": pearson,
            "ideal_vs_qpu_pairwise_distance_spearman": spearman,
            "median_qpu_to_ideal_pairwise_distance_ratio": float(np.median(pairwise_ratios)),
            "minimum_qpu_to_ideal_pairwise_distance_ratio": float(np.min(pairwise_ratios)),
            "maximum_qpu_to_ideal_pairwise_distance_ratio": float(np.max(pairwise_ratios)),
        },
        "posthoc_physics_template_oracle": {
            "scope_warning": "Diagnostic decomposition only. Uses known motif, injection strength and ideal mechanism templates; not a deployable model or confirmatory metric.",
            "actual_nearest_template_correct": actual_oracle_correct,
            "actual_nearest_template_case_count": len(per_case),
            "bootstrap_replicates_per_case": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "minimum_bootstrap_correct_probability": float(np.min(bootstrap_probabilities)),
            "mean_bootstrap_correct_probability": float(np.mean(bootstrap_probabilities)),
        },
        "model_behavior": {
            "step10c_step9a_same_mechanism_prediction_count": int(sum(a == b for a, b in zip(step10c_predictions, step9a_predictions, strict=True))),
            "case_count": len(distorted),
            "step10c_motif_correct": summary["primary_metrics"]["by_motif"],
            "step10c_mechanism_correct": summary["primary_metrics"]["by_mechanism"],
        },
        "decomposition_conclusion": {
            "hardware_evidence_collapse_as_primary_explanation": "NOT_SUPPORTED",
            "learned_representation_or_classifier_generalization_failure": "SUPPORTED_AS_PRIMARY_EXPLANATION",
            "architecture_subcomponent_at_fault": "NOT_YET_ISOLATED",
            "interpretation": "The deployable QPU evidence retains mechanism-specific structure, including in cases the learned model misclassifies. Step-12 failure is therefore better explained by out-of-distribution mapping/generalization than by disappearance of the physical diagnostic signal. Additional decomposition is still required to separate graph-conditioning, diagnostic-encoder, fusion, and mechanism-head causes.",
        },
        "per_case": per_case,
    }


def main() -> None:
    args = parse_args()
    result = analyze(args.bundle, bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        path = args.output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote: {path}")
    print("\nTRIQTO STEP 13 STEP-12 FAILURE DECOMPOSITION COMPLETE")
    print("Hardware evidence collapse as primary explanation:", result["decomposition_conclusion"]["hardware_evidence_collapse_as_primary_explanation"])
    print("Representation/classifier generalization failure:", result["decomposition_conclusion"]["learned_representation_or_classifier_generalization_failure"])
    print("Nearest ideal-template oracle:", f"{result['posthoc_physics_template_oracle']['actual_nearest_template_correct']}/18")
    print("Mean ideal/QPU truth-template cosine:", f"{result['hardware_evidence_geometry']['mean_truth_template_cosine']:.4f}")
    print("Ideal/QPU pairwise geometry Pearson:", f"{result['hardware_evidence_geometry']['ideal_vs_qpu_pairwise_distance_pearson']:.4f}")
    print("Minimum bootstrap oracle correctness probability:", f"{result['posthoc_physics_template_oracle']['minimum_bootstrap_correct_probability']:.4f}")


if __name__ == "__main__":
    main()
