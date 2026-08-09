#!/usr/bin/env python3
"""Audit the physical semantics of TriQTO phase_like/amplitude_like labels.

This is a read-only development audit. It does not train a classifier, change
labels, touch the historical v0.1 test, or modify the source product.

For pure clean/distorted states psi and phi, define

    BC = sum_i sqrt(p_i q_i),     Q = |<psi|phi>|,

where p_i=|psi_i|^2 and q_i=|phi_i|^2. Then

    1 - Q = (1 - BC) + (BC - Q).

The first non-negative term measures computational-basis population/amplitude-
magnitude change. The second measures overlap loss that remains after magnitude
agreement and is therefore attributable to relative-phase/interference change.
The decomposition is invariant to global phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


SCHEMA = "triqto.v0_2.phase_amplitude_label_semantics_audit_result.v1"
DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_label_semantics_audit"
)

H = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
SDG = np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product-dir", type=Path)
    parser.add_argument("--product-parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            / "configs/v0_2/phase_amplitude_label_semantics_audit.json"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_product(args: argparse.Namespace) -> Path:
    if args.product_dir is not None:
        root = args.product_dir.expanduser().resolve()
    else:
        parent = args.product_parent.expanduser().resolve()
        pointer = parent / "current_product.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"missing product pointer: {pointer}")
        root = Path(str(read_json(pointer)["product_dir"])).expanduser().resolve()
    marker = root / "generation_complete.json"
    manifest = root / "manifests" / "item_manifest.parquet"
    if not marker.is_file():
        raise RuntimeError(f"source generation is incomplete: {marker}")
    if not manifest.is_file():
        raise FileNotFoundError(f"missing item manifest: {manifest}")
    return root


def scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected scalar")
    item = array.reshape(-1)[0]
    return item.item() if isinstance(item, np.generic) else item


def normalized_state(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    state = np.asarray(real, dtype=np.float64).reshape(-1) + 1j * np.asarray(
        imag, dtype=np.float64
    ).reshape(-1)
    if state.size == 0 or state.size & (state.size - 1):
        raise ValueError(f"statevector dimension is not a power of two: {state.size}")
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid statevector norm")
    state = state / norm
    if not np.all(np.isfinite(state.real)) or not np.all(np.isfinite(state.imag)):
        raise ValueError("non-finite statevector")
    return state


def apply_single_qubit_matrix(
    state: np.ndarray, qubit: int, matrix: np.ndarray
) -> np.ndarray:
    output = np.asarray(state, dtype=np.complex128).copy()
    mask = 1 << qubit
    for low in range(output.size):
        if low & mask:
            continue
        high = low | mask
        a = output[low]
        b = output[high]
        output[low] = matrix[0, 0] * a + matrix[0, 1] * b
        output[high] = matrix[1, 0] * a + matrix[1, 1] * b
    return output


def measurement_probabilities(state: np.ndarray, basis: str) -> np.ndarray:
    n_qubits = int(round(math.log2(state.size)))
    rotated = np.asarray(state, dtype=np.complex128)
    if basis == "X":
        for qubit in range(n_qubits):
            rotated = apply_single_qubit_matrix(rotated, qubit, H)
    elif basis == "Y":
        for qubit in range(n_qubits):
            rotated = apply_single_qubit_matrix(rotated, qubit, SDG)
            rotated = apply_single_qubit_matrix(rotated, qubit, H)
    elif basis != "Z":
        raise ValueError(f"unsupported basis: {basis}")
    probabilities = np.abs(rotated) ** 2
    probabilities /= np.sum(probabilities)
    return probabilities.astype(np.float64)


def pauli_expectations_from_basis_probabilities(
    probabilities: np.ndarray, n_qubits: int
) -> np.ndarray:
    result = np.zeros(n_qubits, dtype=np.float64)
    indices = np.arange(probabilities.size, dtype=np.int64)
    for qubit in range(n_qubits):
        signs = np.where((indices & (1 << qubit)) == 0, 1.0, -1.0)
        result[qubit] = float(np.sum(signs * probabilities))
    return result


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def overlap_decomposition(
    clean: np.ndarray,
    distorted: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, float]:
    p = np.abs(clean) ** 2
    q = np.abs(distorted) ** 2
    bc = float(np.sum(np.sqrt(np.maximum(0.0, p * q))))
    overlap = float(abs(np.vdot(clean, distorted)))
    bc = float(np.clip(bc, 0.0, 1.0))
    overlap = float(np.clip(overlap, 0.0, 1.0))
    population = max(0.0, 1.0 - bc)
    phase = max(0.0, bc - overlap)
    total = max(0.0, 1.0 - overlap)
    closure_error = abs(total - (population + phase))
    return {
        "bhattacharyya_coefficient": bc,
        "overlap_abs": overlap,
        "population_component": population,
        "phase_component": phase,
        "total_overlap_loss": total,
        "decomposition_closure_error": closure_error,
        "dominance_log_ratio": float(
            math.log((phase + epsilon) / (population + epsilon))
        ),
        "state_infidelity": max(0.0, 1.0 - overlap * overlap),
    }


def expected_sign_match(coarse_label: str, log_ratio: float) -> bool:
    if coarse_label == "phase_like":
        return log_ratio > 0.0
    if coarse_label == "amplitude_like":
        return log_ratio < 0.0
    raise ValueError(f"unexpected coarse label: {coarse_label}")


def expected_median_direction(coarse_label: str, median: float) -> bool:
    return expected_sign_match(coarse_label, median)


def strong_phenotype(
    population: float,
    phase: float,
    total: float,
    *,
    negligible_floor: float,
    dominance_ratio: float,
) -> str:
    if total <= negligible_floor:
        return "negligible"
    if phase >= dominance_ratio * population:
        return "phase_dominant"
    if population >= dominance_ratio * phase:
        return "population_dominant"
    return "mixed"


def strong_expected_match(coarse_label: str, phenotype: str) -> bool:
    return (
        coarse_label == "phase_like" and phenotype == "phase_dominant"
    ) or (
        coarse_label == "amplitude_like" and phenotype == "population_dominant"
    )


def load_manifest(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "manifests" / "item_manifest.parquet"
    rows = [dict(row) for row in pq.read_table(manifest_path).to_pylist()]
    if not rows:
        raise RuntimeError("empty source manifest")
    entity_ids: set[str] = set()
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "validation"}:
            raise RuntimeError(
                "label-semantics audit refuses non-development split "
                f"{split!r} for {row.get('entity_id')}"
            )
        coarse = str(row["coarse_label"])
        if coarse not in {"phase_like", "amplitude_like"}:
            raise RuntimeError(f"unexpected coarse label: {coarse}")
        entity_id = str(row["entity_id"])
        if entity_id in entity_ids:
            raise RuntimeError(f"duplicate entity_id in manifest: {entity_id}")
        entity_ids.add(entity_id)
    return rows


def audit_example(
    root: Path,
    row: Mapping[str, Any],
    *,
    epsilon: float,
    negligible_floor: float,
    dominance_ratio: float,
) -> dict[str, Any]:
    artifact = (root / str(row["artifact_ref"])).resolve()
    if root not in artifact.parents:
        raise RuntimeError(f"artifact escapes source product: {artifact}")
    with np.load(artifact, allow_pickle=False) as archive:
        if "entity_id" in archive.files:
            found = str(scalar(archive["entity_id"]))
            if found != str(row["entity_id"]):
                raise RuntimeError(
                    f"entity mismatch: manifest={row['entity_id']} artifact={found}"
                )
        required = {
            "c__clean_statevector_real",
            "c__clean_statevector_imag",
            "c__distorted_statevector_real",
            "c__distorted_statevector_imag",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{artifact} lacks required state evidence: {missing}")
        clean = normalized_state(
            archive["c__clean_statevector_real"],
            archive["c__clean_statevector_imag"],
        )
        distorted = normalized_state(
            archive["c__distorted_statevector_real"],
            archive["c__distorted_statevector_imag"],
        )

    if clean.shape != distorted.shape:
        raise RuntimeError(f"clean/distorted state shape mismatch: {artifact}")
    n_qubits = int(round(math.log2(clean.size)))
    if int(row["n_qubits"]) != n_qubits:
        raise RuntimeError(
            f"n_qubits mismatch for {row['entity_id']}: "
            f"manifest={row['n_qubits']} state={n_qubits}"
        )

    decomposition = overlap_decomposition(clean, distorted, epsilon=epsilon)
    basis_clean = {
        basis: measurement_probabilities(clean, basis) for basis in ("X", "Y", "Z")
    }
    basis_distorted = {
        basis: measurement_probabilities(distorted, basis)
        for basis in ("X", "Y", "Z")
    }
    expectations_clean = {
        basis: pauli_expectations_from_basis_probabilities(
            basis_clean[basis], n_qubits
        )
        for basis in ("X", "Y", "Z")
    }
    expectations_distorted = {
        basis: pauli_expectations_from_basis_probabilities(
            basis_distorted[basis], n_qubits
        )
        for basis in ("X", "Y", "Z")
    }

    phenotype = strong_phenotype(
        decomposition["population_component"],
        decomposition["phase_component"],
        decomposition["total_overlap_loss"],
        negligible_floor=negligible_floor,
        dominance_ratio=dominance_ratio,
    )
    coarse = str(row["coarse_label"])
    resolved = phenotype != "negligible"
    sign_match = (
        expected_sign_match(coarse, decomposition["dominance_log_ratio"])
        if resolved
        else None
    )
    strong_match = strong_expected_match(coarse, phenotype) if resolved else None

    output: dict[str, Any] = {
        "entity_id": str(row["entity_id"]),
        "split_group_id": str(row["split_group_id"]),
        "split": str(row["split"]),
        "coarse_label": coarse,
        "raw_label": str(row["raw_label"]),
        "family": str(row["family"]),
        "n_qubits": n_qubits,
        "strength_key": str(row["strength_key"]),
        "phase_sensitive_family": bool(row["phase_sensitive_family"]),
        "affected_qubit_signature": str(row["affected_qubit_signature"]),
        "phenotype": phenotype,
        "resolved_effect": resolved,
        "expected_sign_match": sign_match,
        "strong_expected_match": strong_match,
        **decomposition,
    }
    for basis in ("X", "Y", "Z"):
        output[f"tv_{basis.lower()}"] = total_variation(
            basis_clean[basis], basis_distorted[basis]
        )
        delta = expectations_distorted[basis] - expectations_clean[basis]
        output[f"expectation_rms_delta_{basis.lower()}"] = float(
            np.sqrt(np.mean(delta * delta))
        )
        output[f"expectation_max_abs_delta_{basis.lower()}"] = float(
            np.max(np.abs(delta))
        )
    return output


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    stratum_type: str,
    stratum_value: str,
    coarse_label: str,
) -> dict[str, Any]:
    resolved = [row for row in rows if bool(row["resolved_effect"])]
    log_ratios = np.asarray(
        [float(row["dominance_log_ratio"]) for row in resolved], dtype=np.float64
    )
    sign_values = [bool(row["expected_sign_match"]) for row in resolved]
    strong_values = [bool(row["strong_expected_match"]) for row in resolved]
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "coarse_label": coarse_label,
        "n": len(rows),
        "n_resolved": len(resolved),
        "negligible_fraction": 1.0 - (len(resolved) / len(rows)),
        "expected_sign_concordance": (
            finite_mean(float(value) for value in sign_values)
            if sign_values
            else float("nan")
        ),
        "strong_dominance_concordance": (
            finite_mean(float(value) for value in strong_values)
            if strong_values
            else float("nan")
        ),
        "median_dominance_log_ratio": (
            float(np.median(log_ratios)) if log_ratios.size else float("nan")
        ),
        "mean_population_component": finite_mean(
            float(row["population_component"]) for row in rows
        ),
        "mean_phase_component": finite_mean(
            float(row["phase_component"]) for row in rows
        ),
        "mean_total_overlap_loss": finite_mean(
            float(row["total_overlap_loss"]) for row in rows
        ),
        "mean_tv_x": finite_mean(float(row["tv_x"]) for row in rows),
        "mean_tv_y": finite_mean(float(row["tv_y"]) for row in rows),
        "mean_tv_z": finite_mean(float(row["tv_z"]) for row in rows),
    }


def stratified_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for coarse in ("phase_like", "amplitude_like"):
        selected = [row for row in rows if row["coarse_label"] == coarse]
        output.append(
            summarize_rows(
                selected,
                stratum_type="overall",
                stratum_value="ALL",
                coarse_label=coarse,
            )
        )

    dimensions = (
        "raw_label",
        "family",
        "n_qubits",
        "strength_key",
        "phase_sensitive_family",
        "affected_qubit_signature",
    )
    for dimension in dimensions:
        groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(str(row["coarse_label"]), str(row[dimension]))].append(row)
        for (coarse, value), members in sorted(groups.items()):
            output.append(
                summarize_rows(
                    members,
                    stratum_type=dimension,
                    stratum_value=value,
                    coarse_label=coarse,
                )
            )
    return output


def group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["split_group_id"])].append(row)
    group_ids = sorted(groups)
    if not group_ids:
        raise RuntimeError("cannot bootstrap empty group set")
    rng = np.random.default_rng(seed)
    sign_values: list[float] = []
    median_values: list[float] = []
    for _ in range(repeats):
        sampled_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
        sample: list[Mapping[str, Any]] = []
        for group_id in sampled_ids:
            sample.extend(groups[str(group_id)])
        resolved = [row for row in sample if bool(row["resolved_effect"])]
        if not resolved:
            continue
        sign_values.append(
            float(np.mean([bool(row["expected_sign_match"]) for row in resolved]))
        )
        median_values.append(
            float(np.median([float(row["dominance_log_ratio"]) for row in resolved]))
        )
    alpha = (1.0 - confidence) / 2.0

    def interval(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "low": float(np.quantile(array, alpha)),
            "high": float(np.quantile(array, 1.0 - alpha)),
        }

    return {
        "expected_sign_concordance": interval(sign_values),
        "median_dominance_log_ratio": interval(median_values),
    }


def decide(
    summaries: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    stable_overall = float(thresholds["stable_overall_sign_concordance_min"])
    context_floor = float(
        thresholds["context_dependent_overall_sign_concordance_min"]
    )
    stable_stratum = float(thresholds["stable_stratum_sign_concordance_min"])
    min_count = int(thresholds["minimum_stratum_count"])
    max_negligible = float(thresholds["maximum_negligible_fraction_for_stable"])

    overall = {
        str(row["coarse_label"]): row
        for row in summaries
        if row["stratum_type"] == "overall"
    }
    required = {"phase_like", "amplitude_like"}
    if set(overall) != required:
        raise RuntimeError(f"missing overall summaries: {set(overall)}")

    overall_concordance_ok_stable = all(
        float(overall[label]["expected_sign_concordance"]) >= stable_overall
        for label in required
    )
    overall_concordance_ok_context = all(
        float(overall[label]["expected_sign_concordance"]) >= context_floor
        for label in required
    )
    medians_correct = all(
        expected_median_direction(
            label, float(overall[label]["median_dominance_log_ratio"])
        )
        for label in required
    )
    negligible_ok = all(
        float(overall[label]["negligible_fraction"]) <= max_negligible
        for label in required
    )

    decision_dimensions = {"raw_label", "family", "n_qubits", "strength_key"}
    eligible = [
        row
        for row in summaries
        if row["stratum_type"] in decision_dimensions and int(row["n"]) >= min_count
    ]
    bad_strata = [
        row
        for row in eligible
        if (
            not np.isfinite(float(row["expected_sign_concordance"]))
            or float(row["expected_sign_concordance"]) < stable_stratum
            or not expected_median_direction(
                str(row["coarse_label"]),
                float(row["median_dominance_log_ratio"]),
            )
        )
    ]

    if not negligible_ok:
        status = "INSUFFICIENT_EFFECT"
    elif overall_concordance_ok_stable and medians_correct and not bad_strata:
        status = "PHENOMENOLOGICALLY_STABLE"
    elif overall_concordance_ok_context and medians_correct:
        status = "CONTEXT_DEPENDENT"
    else:
        status = "SEMANTICALLY_UNSTABLE"

    return {
        "schema": SCHEMA,
        "status": status,
        "coarse_labels_are_not_rewritten_by_this_audit": True,
        "overall": overall,
        "eligible_strata_count": len(eligible),
        "bad_strata_count": len(bad_strata),
        "bad_strata": bad_strata,
        "interpretation": {
            "PHENOMENOLOGICALLY_STABLE": (
                "The coarse phenotype names track the observed final-state effect "
                "consistently enough to remain plausible supervised targets."
            ),
            "CONTEXT_DEPENDENT": (
                "The coarse phenotype names are directionally meaningful overall "
                "but depend materially on circuit/state context; mechanism and "
                "phenomenology should not be conflated."
            ),
            "SEMANTICALLY_UNSTABLE": (
                "The coarse phenotype names do not reliably describe the actual "
                "final-state effect. Prefer mechanism labels and/or continuous "
                "effect coordinates for future development."
            ),
            "INSUFFICIENT_EFFECT": (
                "Too many injected distortions produce negligible state change for "
                "a stable phenotype target under this cohort."
            ),
        }[status],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("status") != "FROZEN_DEVELOPMENT_AUDIT":
        raise RuntimeError("audit config is not frozen")

    source_root = resolve_product(args)
    manifest_path = source_root / "manifests" / "item_manifest.parquet"
    source_generation = source_root / "generation_complete.json"
    rows = load_manifest(source_root)

    config_hash = sha256_file(config_path)
    manifest_hash = sha256_file(manifest_path)
    generation_hash = sha256_file(source_generation)
    audit_identity = hashlib.sha256(
        canonical_json(
            {
                "schema": SCHEMA,
                "config_sha256": config_hash,
                "manifest_sha256": manifest_hash,
                "generation_complete_sha256": generation_hash,
            }
        ).encode("utf-8")
    ).hexdigest()
    output_root = (
        args.output_parent.expanduser().resolve() / f"audit_{audit_identity[:24]}"
    )
    complete_path = output_root / "audit_complete.json"
    if complete_path.exists():
        print(f"Audit already complete: {complete_path}")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing non-empty incomplete audit directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = config["thresholds"]
    epsilon = float(thresholds["epsilon"])
    negligible_floor = float(thresholds["negligible_total_overlap_loss"])
    dominance_ratio = float(thresholds["strong_dominance_ratio"])

    example_metrics: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        example_metrics.append(
            audit_example(
                source_root,
                row,
                epsilon=epsilon,
                negligible_floor=negligible_floor,
                dominance_ratio=dominance_ratio,
            )
        )
        if index % 25 == 0 or index == len(rows):
            print(f"Audited {index}/{len(rows)} examples", flush=True)

    max_closure_error = max(
        float(row["decomposition_closure_error"]) for row in example_metrics
    )
    if max_closure_error > 1e-10:
        raise RuntimeError(
            f"overlap decomposition closure error too large: {max_closure_error}"
        )

    summaries = stratified_summaries(example_metrics)
    decision = decide(summaries, config)

    bootstrap_config = config["bootstrap"]
    bootstrap: dict[str, Any] = {}
    for offset, coarse in enumerate(("phase_like", "amplitude_like")):
        selected = [row for row in example_metrics if row["coarse_label"] == coarse]
        bootstrap[coarse] = group_bootstrap(
            selected,
            repeats=int(bootstrap_config["group_repeats"]),
            seed=int(bootstrap_config["seed"]) + offset,
            confidence=float(bootstrap_config["confidence"]),
        )

    example_path = output_root / "example_metrics.csv"
    stratified_path = output_root / "stratified_metrics.csv"
    decision_path = output_root / "decision.json"
    atomic_csv(example_path, example_metrics)
    atomic_csv(stratified_path, summaries)
    atomic_json(
        decision_path,
        {
            **decision,
            "bootstrap_95_ci": bootstrap,
            "development_only": True,
            "historical_v0_1_test_accessed": False,
            "source_product": str(source_root),
        },
    )

    complete = {
        "schema": SCHEMA,
        "status": "AUDIT_COMPLETE",
        "audit_id": f"audit_{audit_identity[:24]}",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_product": str(source_root),
        "source_manifest_count": len(rows),
        "source_manifest_sha256": manifest_hash,
        "source_generation_complete_sha256": generation_hash,
        "config_sha256": config_hash,
        "example_metrics_sha256": sha256_file(example_path),
        "stratified_metrics_sha256": sha256_file(stratified_path),
        "decision_sha256": sha256_file(decision_path),
        "decision_status": decision["status"],
        "maximum_decomposition_closure_error": max_closure_error,
        "historical_v0_1_test_accessed": False,
        "classifier_trained": False,
        "labels_changed": False,
        "source_product_modified": False,
    }
    atomic_json(complete_path, complete)

    print()
    print("=" * 78)
    print("TRIQTO PHASE/AMPLITUDE LABEL-SEMANTICS AUDIT COMPLETE")
    print("=" * 78)
    print(f"Decision: {decision['status']}")
    for coarse in ("phase_like", "amplitude_like"):
        overall = decision["overall"][coarse]
        ci = bootstrap[coarse]["expected_sign_concordance"]
        print(
            f"{coarse}: sign concordance="
            f"{float(overall['expected_sign_concordance']):.4f} "
            f"(95% group-bootstrap CI {ci['low']:.4f}..{ci['high']:.4f}), "
            f"median log-ratio="
            f"{float(overall['median_dominance_log_ratio']):.4f}, "
            f"negligible={float(overall['negligible_fraction']):.4f}"
        )
    print(f"Bad eligible context strata: {decision['bad_strata_count']}")
    print("Historical v0.1 test accessed: NO")
    print("Classifier trained: NO")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
