#!/usr/bin/env python3
"""Matched B_delta identifiability audit for TriQTO Step 3.

This is a frozen development audit. It does not train a classifier, modify the
model, access the historical v0.1 test, or reuse the spent confirmatory cohort.

The audit first verifies that the original pilot distortion gate is a single
terminal RX/RY/RZ rotation whose axis matches the manifest raw label. Only under
that condition is applying a matched rotation directly to the stored clean final
state exactly equivalent to the tested generator's intervention protocol.

For each deduplicated physical context (clean final state, family, affected
qubit, and strength), the audit generates matched RZ, RX, and RY
counterfactuals and computes hardware-facing relational Z/X/Y evidence. Pairwise
separation is then measured without fitting a classifier. Privileged statevector
access is used only for counterfactual generation and phenomenology ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


SCHEMA = "triqto.v0_2.matched_bdelta_identifiability_audit_result.v1"
DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_matched_bdelta_identifiability"
)

H = np.asarray([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
SDG = np.asarray([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
PAIR_TYPES = (
    ("rz_drift", "rx_overrotation", "rz_vs_rx"),
    ("rz_drift", "ry_overrotation", "rz_vs_ry"),
    ("rx_overrotation", "ry_overrotation", "rx_vs_ry"),
)


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
            / "configs/v0_2/matched_bdelta_identifiability_audit.json"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def resolve_product(arguments: argparse.Namespace) -> Path:
    if arguments.product_dir is not None:
        root = arguments.product_dir.expanduser().resolve()
    else:
        pointer = arguments.product_parent.expanduser().resolve() / "current_product.json"
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


def canonical_state_hash(state: np.ndarray) -> str:
    state = np.asarray(state, dtype=np.complex128).copy()
    index = int(np.argmax(np.abs(state)))
    anchor = state[index]
    if abs(anchor) > 0.0:
        state *= np.exp(-1j * np.angle(anchor))
    if state[index].real < 0.0:
        state *= -1.0
    packed = np.stack(
        (np.round(state.real, 12), np.round(state.imag, 12)), axis=1
    ).astype("<f8", copy=False)
    return sha256_bytes(packed.tobytes(order="C"))


def apply_single_qubit_matrix(
    state: np.ndarray, qubit: int, matrix: np.ndarray
) -> np.ndarray:
    output = np.asarray(state, dtype=np.complex128).copy()
    if qubit < 0 or (1 << qubit) >= output.size:
        raise ValueError(f"qubit {qubit} outside state dimension {output.size}")
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


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    c = math.cos(half)
    s = math.sin(half)
    if axis == "rx":
        return np.asarray([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if axis == "ry":
        return np.asarray([[c, -s], [s, c]], dtype=np.complex128)
    if axis == "rz":
        return np.asarray(
            [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
            dtype=np.complex128,
        )
    raise ValueError(f"unsupported rotation axis: {axis}")


def axis_for_mechanism(mechanism: str) -> str:
    value = mechanism.lower()
    if "rz" in value:
        return "rz"
    if "rx" in value:
        return "rx"
    if "ry" in value:
        return "ry"
    raise ValueError(f"cannot infer rotation axis from mechanism {mechanism!r}")


def normalized_gate_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def parse_affected_qubit(signature: str, n_qubits: int) -> int:
    values = {int(value) for value in re.findall(r"\d+", str(signature))}
    if len(values) != 1:
        raise ValueError(f"expected one affected qubit in signature {signature!r}")
    qubit = next(iter(values))
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(
            f"affected qubit {qubit} outside n_qubits={n_qubits}: {signature!r}"
        )
    return qubit


def parse_strength(value: str) -> float:
    try:
        result = float(value)
    except ValueError:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match is None:
            raise ValueError(f"cannot parse strength {value!r}") from None
        result = float(match.group(0))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"invalid positive distortion strength: {value!r}")
    return result


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


def pauli_expectations(
    probabilities: np.ndarray, n_qubits: int
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    indices = np.arange(probabilities.size, dtype=np.int64)
    output = np.zeros(n_qubits, dtype=np.float64)
    for qubit in range(n_qubits):
        signs = np.where((indices & (1 << qubit)) == 0, 1.0, -1.0)
        output[qubit] = float(np.sum(signs * probabilities))
    return output


def total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return 0.5 * float(np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def rms(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(array * array))) if array.size else 0.0


def overlap_decomposition(
    clean: np.ndarray, distorted: np.ndarray, *, epsilon: float
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
    return {
        "population_component": population,
        "phase_component": phase,
        "total_overlap_loss": total,
        "dominance_log_ratio": float(
            math.log((phase + epsilon) / (population + epsilon))
        ),
        "state_infidelity": max(0.0, 1.0 - overlap * overlap),
        "decomposition_closure_error": abs(total - (population + phase)),
    }


def phenotype(
    decomposition: Mapping[str, float],
    *,
    negligible_floor: float,
    dominance_ratio: float,
) -> str:
    population = float(decomposition["population_component"])
    phase = float(decomposition["phase_component"])
    total = float(decomposition["total_overlap_loss"])
    if total <= negligible_floor:
        return "negligible"
    if phase >= dominance_ratio * population:
        return "phase_dominant"
    if population >= dominance_ratio * phase:
        return "population_dominant"
    return "mixed"


def load_manifest(root: Path) -> list[dict[str, Any]]:
    path = root / "manifests" / "item_manifest.parquet"
    rows = [dict(row) for row in pq.read_table(path).to_pylist()]
    if not rows:
        raise RuntimeError("empty source manifest")
    seen: set[str] = set()
    for row in rows:
        split = str(row["split"])
        if split not in {"train", "validation"}:
            raise RuntimeError(
                "matched identifiability audit refuses non-development split "
                f"{split!r} for {row.get('entity_id')}"
            )
        entity = str(row["entity_id"])
        if entity in seen:
            raise RuntimeError(f"duplicate entity_id: {entity}")
        seen.add(entity)
        axis_for_mechanism(str(row["raw_label"]))
    return rows


def source_artifact(root: Path, row: Mapping[str, Any]) -> Path:
    artifact = (root / str(row["artifact_ref"])).resolve()
    if root not in artifact.parents:
        raise RuntimeError(f"artifact escapes source product: {artifact}")
    return artifact


def inspect_source_example(
    root: Path, row: Mapping[str, Any]
) -> dict[str, Any]:
    artifact = source_artifact(root, row)
    with np.load(artifact, allow_pickle=False) as archive:
        required = {
            "c__clean_statevector_real",
            "c__clean_statevector_imag",
            "a__x_graph_gate_names",
            "audit__removed_distortion_gate_indices",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{artifact} lacks required arrays: {missing}")
        if "entity_id" in archive.files:
            found = str(scalar(archive["entity_id"]))
            if found != str(row["entity_id"]):
                raise RuntimeError(
                    f"entity mismatch: manifest={row['entity_id']} artifact={found}"
                )
        clean = normalized_state(
            archive["c__clean_statevector_real"],
            archive["c__clean_statevector_imag"],
        )
        gate_names = (
            np.asarray(archive["a__x_graph_gate_names"]).astype(str).reshape(-1)
        )
        removed = np.asarray(
            archive["audit__removed_distortion_gate_indices"], dtype=np.int64
        ).reshape(-1)

    n_qubits = int(round(math.log2(clean.size)))
    if int(row["n_qubits"]) != n_qubits:
        raise RuntimeError(
            f"n_qubits mismatch for {row['entity_id']}: "
            f"manifest={row['n_qubits']} state={n_qubits}"
        )
    if removed.size != 1:
        return {
            "valid": False,
            "reason": "removed_distortion_gate_count_not_one",
            "removed_count": int(removed.size),
            "entity_id": str(row["entity_id"]),
        }
    index = int(removed[0])
    if index < 0 or index >= gate_names.size:
        return {
            "valid": False,
            "reason": "removed_distortion_gate_index_out_of_range",
            "removed_index": index,
            "gate_count": int(gate_names.size),
            "entity_id": str(row["entity_id"]),
        }
    expected_axis = axis_for_mechanism(str(row["raw_label"]))
    removed_name = normalized_gate_name(gate_names[index])
    axis_match = removed_name == expected_axis
    terminal = index == gate_names.size - 1
    qubit = parse_affected_qubit(str(row["affected_qubit_signature"]), n_qubits)
    strength = parse_strength(str(row["strength_key"]))
    return {
        "valid": bool(axis_match and terminal),
        "reason": (
            "ok"
            if axis_match and terminal
            else "axis_mismatch"
            if not axis_match
            else "nonterminal_distortion"
        ),
        "entity_id": str(row["entity_id"]),
        "split_group_id": str(row["split_group_id"]),
        "family": str(row["family"]),
        "phase_sensitive_family": bool(row["phase_sensitive_family"]),
        "n_qubits": n_qubits,
        "strength_key": str(row["strength_key"]),
        "strength": strength,
        "affected_qubit_signature": str(row["affected_qubit_signature"]),
        "affected_qubit": qubit,
        "raw_label": str(row["raw_label"]),
        "expected_axis": expected_axis,
        "removed_gate_name": str(gate_names[index]),
        "removed_index": index,
        "gate_count": int(gate_names.size),
        "terminal": terminal,
        "axis_match": axis_match,
        "clean_state_hash": canonical_state_hash(clean),
        "clean_state": clean,
        "artifact": artifact,
    }


def context_key(info: Mapping[str, Any]) -> str:
    payload = {
        "clean_state_hash": info["clean_state_hash"],
        "family": info["family"],
        "n_qubits": info["n_qubits"],
        "affected_qubit": info["affected_qubit"],
        "strength_key": info["strength_key"],
        "phase_sensitive_family": info["phase_sensitive_family"],
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def evidence_for_state(
    clean: np.ndarray,
    distorted: np.ndarray,
    *,
    n_qubits: int,
) -> dict[str, Any]:
    clean_probs = {
        basis: measurement_probabilities(clean, basis) for basis in ("X", "Y", "Z")
    }
    distorted_probs = {
        basis: measurement_probabilities(distorted, basis)
        for basis in ("X", "Y", "Z")
    }
    clean_exp = {
        basis: pauli_expectations(clean_probs[basis], n_qubits)
        for basis in ("X", "Y", "Z")
    }
    distorted_exp = {
        basis: pauli_expectations(distorted_probs[basis], n_qubits)
        for basis in ("X", "Y", "Z")
    }
    tv = {
        basis: total_variation(clean_probs[basis], distorted_probs[basis])
        for basis in ("X", "Y", "Z")
    }
    exp_rms = {
        basis: rms(distorted_exp[basis] - clean_exp[basis])
        for basis in ("X", "Y", "Z")
    }
    signal_score = float(
        np.mean(
            [
                tv["X"],
                tv["Y"],
                tv["Z"],
                0.5 * exp_rms["X"],
                0.5 * exp_rms["Y"],
                0.5 * exp_rms["Z"],
            ]
        )
    )
    return {
        "clean_probs": clean_probs,
        "distorted_probs": distorted_probs,
        "clean_exp": clean_exp,
        "distorted_exp": distorted_exp,
        "tv": tv,
        "exp_rms": exp_rms,
        "signal_score": signal_score,
    }


def pair_separation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    epsilon: float,
    raw_minimum: float,
    relative_minimum: float,
    collision_maximum: float,
) -> dict[str, Any]:
    tv = {
        basis: total_variation(
            left["distorted_probs"][basis], right["distorted_probs"][basis]
        )
        for basis in ("X", "Y", "Z")
    }
    exp_rms = {
        basis: rms(
            np.asarray(left["distorted_exp"][basis])
            - np.asarray(right["distorted_exp"][basis])
        )
        for basis in ("X", "Y", "Z")
    }
    score = float(
        np.mean(
            [
                tv["X"],
                tv["Y"],
                tv["Z"],
                0.5 * exp_rms["X"],
                0.5 * exp_rms["Y"],
                0.5 * exp_rms["Z"],
            ]
        )
    )
    denominator = float(left["signal_score"]) + float(right["signal_score"]) + epsilon
    relative = 2.0 * score / denominator
    return {
        "pair_separation_score": score,
        "relative_separation": relative,
        "numerical_collision": score <= collision_maximum,
        "strong_pair": bool(score >= raw_minimum and relative >= relative_minimum),
        "pair_tv_x": tv["X"],
        "pair_tv_y": tv["Y"],
        "pair_tv_z": tv["Z"],
        "pair_expectation_rms_x": exp_rms["X"],
        "pair_expectation_rms_y": exp_rms["Y"],
        "pair_expectation_rms_z": exp_rms["Z"],
    }


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.mean(array)) if array.size else float("nan")


def summarize_pairs(
    rows: Sequence[Mapping[str, Any]], *, stratum_type: str, stratum_value: str
) -> dict[str, Any]:
    different = [row for row in rows if bool(row["phenotype_differs"])]
    return {
        "stratum_type": stratum_type,
        "stratum_value": stratum_value,
        "pair_count": len(rows),
        "strong_pair_fraction": finite_mean(
            float(bool(row["strong_pair"])) for row in rows
        ),
        "numerical_collision_fraction": finite_mean(
            float(bool(row["numerical_collision"])) for row in rows
        ),
        "median_pair_separation_score": (
            float(np.median([float(row["pair_separation_score"]) for row in rows]))
            if rows
            else float("nan")
        ),
        "median_relative_separation": (
            float(np.median([float(row["relative_separation"]) for row in rows]))
            if rows
            else float("nan")
        ),
        "different_phenotype_pair_count": len(different),
        "different_phenotype_strong_fraction": finite_mean(
            float(bool(row["strong_pair"])) for row in different
        ),
    }


def stratified_pair_summaries(
    pairs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = [summarize_pairs(pairs, stratum_type="overall", stratum_value="all")]
    dimensions = (
        "pair_type",
        "family",
        "n_qubits",
        "strength_key",
        "phase_sensitive_family",
        "affected_qubit_signature",
    )
    for dimension in dimensions:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairs:
            grouped[str(row[dimension])].append(row)
        for value in sorted(grouped):
            output.append(
                summarize_pairs(
                    grouped[value], stratum_type=dimension, stratum_value=value
                )
            )
    return output


def group_bootstrap_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    group_key: str,
    repeats: int,
    seed: int,
    filter_key: str | None = None,
) -> tuple[float, float]:
    if filter_key is not None:
        rows = [row for row in rows if bool(row[filter_key])]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    keys = sorted(groups)
    if not keys:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample: list[Mapping[str, Any]] = []
        for key in chosen:
            sample.extend(groups[str(key)])
        values[repeat] = np.mean(
            [float(bool(row[value_key])) for row in sample]
        )
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def group_bootstrap_negligible(
    rows: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["context_id"])].append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        sample: list[Mapping[str, Any]] = []
        for key in chosen:
            sample.extend(groups[str(key)])
        values[repeat] = np.mean(
            [float(row["phenotype"] == "negligible") for row in sample]
        )
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def decide(
    pair_rows: Sequence[Mapping[str, Any]],
    counterfactual_rows: Sequence[Mapping[str, Any]],
    strata: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = config["decision_policy"]
    stable = policy["IDENTIFIABLE"]
    context = policy["CONTEXT_DEPENDENT"]
    nonid = policy["NON_IDENTIFIABLE_REGIMES"]
    insufficient = policy["INSUFFICIENT_EFFECT"]
    min_pairs = int(policy["minimum_eligible_stratum_pairs"])

    overall_strong = finite_mean(
        float(bool(row["strong_pair"])) for row in pair_rows
    )
    collisions = finite_mean(
        float(bool(row["numerical_collision"])) for row in pair_rows
    )
    different = [row for row in pair_rows if bool(row["phenotype_differs"])]
    different_strong = finite_mean(
        float(bool(row["strong_pair"])) for row in different
    )
    negligible = finite_mean(
        float(row["phenotype"] == "negligible") for row in counterfactual_rows
    )

    eligible = [
        row
        for row in strata
        if row["stratum_type"] != "overall" and int(row["pair_count"]) >= min_pairs
    ]
    bad = [
        row
        for row in eligible
        if float(row["strong_pair_fraction"])
        < float(stable["eligible_stratum_strong_pair_fraction_min"])
    ]
    severe = [
        row
        for row in eligible
        if float(row["strong_pair_fraction"])
        < float(nonid["eligible_stratum_strong_pair_fraction_below"])
    ]

    if negligible > float(insufficient["negligible_counterfactual_fraction_above"]):
        status = "INSUFFICIENT_EFFECT"
    elif (
        overall_strong < float(nonid["overall_strong_pair_fraction_below"])
        or collisions > float(nonid["numerical_collision_fraction_above"])
        or severe
    ):
        status = "NON_IDENTIFIABLE_REGIMES"
    elif (
        overall_strong >= float(stable["overall_strong_pair_fraction_min"])
        and different_strong
        >= float(stable["different_phenotype_pair_strong_fraction_min"])
        and not bad
        and negligible <= float(stable["negligible_counterfactual_fraction_max"])
        and collisions <= float(stable["numerical_collision_fraction_max"])
    ):
        status = "IDENTIFIABLE"
    elif (
        overall_strong >= float(context["overall_strong_pair_fraction_min"])
        and different_strong
        >= float(context["different_phenotype_pair_strong_fraction_min"])
    ):
        status = "CONTEXT_DEPENDENT"
    else:
        status = "NON_IDENTIFIABLE_REGIMES"

    repeats = int(policy["bootstrap_repeats"])
    seed = int(policy["bootstrap_seed"])
    overall_ci = group_bootstrap_fraction(
        pair_rows,
        value_key="strong_pair",
        group_key="context_id",
        repeats=repeats,
        seed=seed,
    )
    different_ci = group_bootstrap_fraction(
        pair_rows,
        value_key="strong_pair",
        group_key="context_id",
        repeats=repeats,
        seed=seed + 1,
        filter_key="phenotype_differs",
    )
    negligible_ci = group_bootstrap_negligible(
        counterfactual_rows, repeats=repeats, seed=seed + 2
    )

    return {
        "status": status,
        "overall_strong_pair_fraction": overall_strong,
        "overall_strong_pair_fraction_ci95": list(overall_ci),
        "numerical_collision_fraction": collisions,
        "different_phenotype_pair_count": len(different),
        "different_phenotype_strong_fraction": different_strong,
        "different_phenotype_strong_fraction_ci95": list(different_ci),
        "negligible_counterfactual_fraction": negligible,
        "negligible_counterfactual_fraction_ci95": list(negligible_ci),
        "eligible_stratum_count": len(eligible),
        "bad_eligible_strata": [
            {
                "stratum_type": row["stratum_type"],
                "stratum_value": row["stratum_value"],
                "pair_count": row["pair_count"],
                "strong_pair_fraction": row["strong_pair_fraction"],
            }
            for row in bad
        ],
        "severe_nonidentifiable_strata": [
            {
                "stratum_type": row["stratum_type"],
                "stratum_value": row["stratum_value"],
                "pair_count": row["pair_count"],
                "strong_pair_fraction": row["strong_pair_fraction"],
            }
            for row in severe
        ],
    }


def main() -> None:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema") != "triqto.v0_2.matched_bdelta_identifiability_audit.v1":
        raise RuntimeError("unexpected audit config schema")

    root = resolve_product(arguments)
    manifest_path = root / "manifests" / "item_manifest.parquet"
    rows = load_manifest(root)

    script_path = Path(__file__).resolve()
    identity = {
        "config_sha256": sha256_file(config_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_generation_complete_sha256": sha256_file(
            root / "generation_complete.json"
        ),
        "runner_sha256": sha256_file(script_path),
    }
    audit_id = "audit_" + hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = arguments.output_parent.expanduser().resolve()
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing audit: {output_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    preflight_rows: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            info = inspect_source_example(root, row)
            preflight_rows.append(
                {key: value for key, value in info.items() if key not in {"clean_state", "artifact"}}
            )
            if info["valid"]:
                infos.append(info)
            if arguments.progress_every > 0 and index % arguments.progress_every == 0:
                print(f"Preflight inspected {index}/{len(rows)} examples", flush=True)

        atomic_csv(staging / "source_preflight.csv", preflight_rows)
        failures = [row for row in preflight_rows if not bool(row["valid"])]
        preflight = {
            "schema": SCHEMA,
            "status": "PASS" if not failures else "REFUSED",
            "source_example_count": len(rows),
            "valid_terminal_axis_matched_count": len(rows) - len(failures),
            "failure_count": len(failures),
            "failure_reasons": dict(
                sorted(
                    (
                        reason,
                        sum(1 for row in failures if row["reason"] == reason),
                    )
                    for reason in {str(row["reason"]) for row in failures}
                )
            ),
            "scientific_gate": (
                "State-only matched counterfactual generation is exact only when the "
                "stored distortion is one terminal single-qubit rotation with matching axis."
            ),
        }
        atomic_json(staging / "preflight.json", preflight)
        if failures:
            complete = {
                "schema": SCHEMA,
                "status": "REFUSED_INSERTION_POLICY",
                "audit_id": audit_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root),
                "identity": identity,
                "preflight": preflight,
                "classifier_trained": False,
                "historical_v0_1_test_accessed": False,
                "spent_confirmatory_cohort_accessed": False,
            }
            atomic_json(staging / "audit_complete.json", complete)
            os.replace(staging, output_root)
            raise RuntimeError(
                "matched B_delta audit refused because the source distortion "
                f"insertion policy is not uniformly terminal/axis-matched; see {output_root}"
            )

        contexts: dict[str, dict[str, Any]] = {}
        for info in infos:
            key = context_key(info)
            if key not in contexts:
                contexts[key] = {
                    "context_id": key,
                    "clean_state": info["clean_state"],
                    "family": info["family"],
                    "phase_sensitive_family": info["phase_sensitive_family"],
                    "n_qubits": info["n_qubits"],
                    "strength_key": info["strength_key"],
                    "strength": info["strength"],
                    "affected_qubit_signature": info["affected_qubit_signature"],
                    "affected_qubit": info["affected_qubit"],
                    "source_entity_ids": [info["entity_id"]],
                    "source_split_group_ids": [info["split_group_id"]],
                }
            else:
                contexts[key]["source_entity_ids"].append(info["entity_id"])
                contexts[key]["source_split_group_ids"].append(info["split_group_id"])

        epsilon = float(config["phenomenology"]["epsilon"])
        negligible_floor = float(
            config["phenomenology"]["negligible_overlap_loss_floor"]
        )
        dominance_ratio = float(config["phenomenology"]["strong_dominance_ratio"])
        pair_config = config["pairwise_identifiability"]
        raw_minimum = float(pair_config["minimum_raw_separation"])
        relative_minimum = float(pair_config["minimum_relative_separation"])
        collision_maximum = float(pair_config["numerical_collision_score_max"])

        counterfactual_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []

        for context_index, (context_id, context) in enumerate(
            sorted(contexts.items()), start=1
        ):
            clean = np.asarray(context["clean_state"], dtype=np.complex128)
            n_qubits = int(context["n_qubits"])
            qubit = int(context["affected_qubit"])
            strength = float(context["strength"])
            generated: dict[str, dict[str, Any]] = {}

            for mechanism in MECHANISMS:
                axis = axis_for_mechanism(mechanism)
                distorted = apply_single_qubit_matrix(
                    clean, qubit, rotation_matrix(axis, strength)
                )
                distorted /= np.linalg.norm(distorted)
                decomposition = overlap_decomposition(
                    clean, distorted, epsilon=epsilon
                )
                label = phenotype(
                    decomposition,
                    negligible_floor=negligible_floor,
                    dominance_ratio=dominance_ratio,
                )
                evidence = evidence_for_state(
                    clean, distorted, n_qubits=n_qubits
                )
                generated[mechanism] = {
                    "state": distorted,
                    "decomposition": decomposition,
                    "phenotype": label,
                    "evidence": evidence,
                }
                counterfactual_rows.append(
                    {
                        "context_id": context_id,
                        "source_entity_count": len(context["source_entity_ids"]),
                        "source_entity_ids": "|".join(
                            sorted(set(context["source_entity_ids"]))
                        ),
                        "family": context["family"],
                        "phase_sensitive_family": context["phase_sensitive_family"],
                        "n_qubits": n_qubits,
                        "strength_key": context["strength_key"],
                        "strength": strength,
                        "affected_qubit_signature": context[
                            "affected_qubit_signature"
                        ],
                        "affected_qubit": qubit,
                        "mechanism": mechanism,
                        "axis": axis,
                        "phenotype": label,
                        "signal_score": evidence["signal_score"],
                        "tv_x": evidence["tv"]["X"],
                        "tv_y": evidence["tv"]["Y"],
                        "tv_z": evidence["tv"]["Z"],
                        "expectation_rms_x": evidence["exp_rms"]["X"],
                        "expectation_rms_y": evidence["exp_rms"]["Y"],
                        "expectation_rms_z": evidence["exp_rms"]["Z"],
                        **decomposition,
                    }
                )

            for left_name, right_name, pair_type in PAIR_TYPES:
                left = generated[left_name]
                right = generated[right_name]
                separation = pair_separation(
                    left["evidence"],
                    right["evidence"],
                    epsilon=epsilon,
                    raw_minimum=raw_minimum,
                    relative_minimum=relative_minimum,
                    collision_maximum=collision_maximum,
                )
                pair_rows.append(
                    {
                        "context_id": context_id,
                        "family": context["family"],
                        "phase_sensitive_family": context[
                            "phase_sensitive_family"
                        ],
                        "n_qubits": n_qubits,
                        "strength_key": context["strength_key"],
                        "affected_qubit_signature": context[
                            "affected_qubit_signature"
                        ],
                        "pair_type": pair_type,
                        "left_mechanism": left_name,
                        "right_mechanism": right_name,
                        "left_phenotype": left["phenotype"],
                        "right_phenotype": right["phenotype"],
                        "phenotype_differs": left["phenotype"] != right["phenotype"],
                        "left_signal_score": left["evidence"]["signal_score"],
                        "right_signal_score": right["evidence"]["signal_score"],
                        "left_dominance_log_ratio": left["decomposition"][
                            "dominance_log_ratio"
                        ],
                        "right_dominance_log_ratio": right["decomposition"][
                            "dominance_log_ratio"
                        ],
                        **separation,
                    }
                )

            if (
                arguments.progress_every > 0
                and context_index % arguments.progress_every == 0
            ):
                print(
                    f"Audited {context_index}/{len(contexts)} matched contexts",
                    flush=True,
                )

        strata = stratified_pair_summaries(pair_rows)
        decision = decide(pair_rows, counterfactual_rows, strata, config)
        decision.update(
            {
                "schema": SCHEMA,
                "audit_id": audit_id,
                "context_count": len(contexts),
                "source_example_count": len(rows),
                "counterfactual_count": len(counterfactual_rows),
                "pair_count": len(pair_rows),
                "classifier_trained": False,
                "historical_v0_1_test_accessed": False,
                "spent_confirmatory_cohort_accessed": False,
            }
        )

        atomic_csv(staging / "counterfactual_metrics.csv", counterfactual_rows)
        atomic_csv(staging / "pairwise_metrics.csv", pair_rows)
        atomic_csv(staging / "stratified_metrics.csv", strata)
        atomic_json(staging / "decision.json", decision)

        file_hashes = {
            name: sha256_file(staging / name)
            for name in (
                "source_preflight.csv",
                "preflight.json",
                "counterfactual_metrics.csv",
                "pairwise_metrics.csv",
                "stratified_metrics.csv",
                "decision.json",
            )
        }
        complete = {
            "schema": SCHEMA,
            "status": "AUDIT_COMPLETE",
            "audit_id": audit_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(root),
            "identity": identity,
            "file_hashes": file_hashes,
            "decision_status": decision["status"],
            "source_example_count": len(rows),
            "deduplicated_context_count": len(contexts),
            "counterfactual_count": len(counterfactual_rows),
            "pair_count": len(pair_rows),
            "classifier_trained": False,
            "historical_v0_1_test_accessed": False,
            "spent_confirmatory_cohort_accessed": False,
            "scientific_boundary": (
                "Exact simulator identifiability only. Finite-shot and hardware "
                "deployability are explicitly deferred to Step 4."
            ),
        }
        atomic_json(staging / "audit_complete.json", complete)
        os.replace(staging, output_root)

    except Exception:
        if staging.exists():
            if not output_root.exists() and (staging / "audit_complete.json").is_file():
                os.replace(staging, output_root)
            elif staging.exists():
                for path in sorted(staging.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                staging.rmdir()
        raise

    print()
    print("TRIQTO MATCHED B_DELTA IDENTIFIABILITY AUDIT COMPLETE")
    print()
    print(f"Decision: {decision['status']}")
    print(
        "Strong mechanism-pair fraction: "
        f"{decision['overall_strong_pair_fraction']:.4f} "
        f"(95% context-bootstrap CI "
        f"{decision['overall_strong_pair_fraction_ci95'][0]:.4f}.."
        f"{decision['overall_strong_pair_fraction_ci95'][1]:.4f})"
    )
    print(
        "Different-phenotype pair strong fraction: "
        f"{decision['different_phenotype_strong_fraction']:.4f} "
        f"(95% CI {decision['different_phenotype_strong_fraction_ci95'][0]:.4f}.."
        f"{decision['different_phenotype_strong_fraction_ci95'][1]:.4f})"
    )
    print(
        "Numerical collision fraction: "
        f"{decision['numerical_collision_fraction']:.4f}"
    )
    print(
        "Negligible counterfactual fraction: "
        f"{decision['negligible_counterfactual_fraction']:.4f}"
    )
    print(
        f"Bad eligible context strata: {len(decision['bad_eligible_strata'])}"
    )
    print(
        "Historical v0.1 test accessed: NO\n"
        "Spent confirmatory cohort accessed: NO\n"
        "Classifier trained: NO"
    )
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
