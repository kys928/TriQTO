#!/usr/bin/env python3
"""Exact circuit-replay matched B_delta identifiability audit (Step 3 v2).

The merged Step 3 v1 audit correctly refused all 280 development examples because
all archived RX/RY/RZ distortions were nonterminal. This v2 runner preserves that
result and replaces the invalid final-state shortcut with exact ordered circuit
replay at the audited distortion index.

For every source example the archived distorted graph sequence is reconstructed,
then validated twice against privileged stored statevectors:

1. replay the full sequence -> archived distorted state;
2. remove the audited distortion gate -> archived clean state.

Both comparisons are global-phase invariant. No matched counterfactual enters the
scientific audit unless every source example passes the frozen replay tolerances.
Matched RZ/RX/RY counterfactuals then replace only the audited gate at the exact
same sequence index, qubit and archived angle before the entire circuit is replayed.

No classifier is trained. The historical v0.1 test and spent confirmatory cohort
are never accessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "audit_matched_bdelta_identifiability.py"
BASE_MODULE_NAME = "triqto_v0_2_matched_bdelta_v1"
SCHEMA = "triqto.v0_2.matched_bdelta_replay_identifiability_audit_result.v1"
DEFAULT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_pilot_v2/data/"
    "v0_2_phase_amplitude_identifiability_pilot"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/workspace/triqto-data/phase15_6_matched_bdelta_replay_identifiability"
)
MECHANISMS = ("rz_drift", "rx_overrotation", "ry_overrotation")
PAIR_TYPES = (
    ("rz_drift", "rx_overrotation", "rz_vs_rx"),
    ("rz_drift", "ry_overrotation", "rz_vs_ry"),
    ("rx_overrotation", "ry_overrotation", "rx_vs_ry"),
)


def load_base():
    spec = importlib.util.spec_from_file_location(BASE_MODULE_NAME, BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Step 3 v1 helpers from {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


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
            / "configs/v0_2/matched_bdelta_replay_identifiability_audit.json"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def resolve_product(arguments: argparse.Namespace) -> Path:
    if arguments.product_dir is not None:
        root = arguments.product_dir.expanduser().resolve()
    else:
        pointer = arguments.product_parent.expanduser().resolve() / "current_product.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"missing product pointer: {pointer}")
        root = Path(str(BASE.read_json(pointer)["product_dir"])).expanduser().resolve()
    marker = root / "generation_complete.json"
    manifest = root / "manifests" / "item_manifest.parquet"
    if not marker.is_file() or not manifest.is_file():
        raise RuntimeError(f"incomplete source product: {root}")
    return root


def scalar(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("expected scalar")
    item = array.reshape(-1)[0]
    return item.item() if isinstance(item, np.generic) else item


def gate_records(archive: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    required = {
        "a__x_graph_gate_names",
        "a__x_graph_gate_qubit_ptr",
        "a__x_graph_gate_qubit_indices",
        "a__x_graph_gate_parameter_ptr",
        "a__x_graph_gate_parameter_sin",
        "a__x_graph_gate_parameter_cos",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise RuntimeError(f"artifact lacks exact replay arrays: {missing}")

    names = np.asarray(archive["a__x_graph_gate_names"]).astype(str).reshape(-1)
    qptr = np.asarray(archive["a__x_graph_gate_qubit_ptr"], dtype=np.int64).reshape(-1)
    qidx = np.asarray(archive["a__x_graph_gate_qubit_indices"], dtype=np.int64).reshape(-1)
    pptr = np.asarray(archive["a__x_graph_gate_parameter_ptr"], dtype=np.int64).reshape(-1)
    psin = np.asarray(archive["a__x_graph_gate_parameter_sin"], dtype=np.float64).reshape(-1)
    pcos = np.asarray(archive["a__x_graph_gate_parameter_cos"], dtype=np.float64).reshape(-1)

    if qptr.size != names.size + 1 or pptr.size != names.size + 1:
        raise ValueError("gate pointer length mismatch")
    if qptr[0] != 0 or qptr[-1] != qidx.size or np.any(np.diff(qptr) < 0):
        raise ValueError("invalid gate-qubit pointer")
    if pptr[0] != 0 or pptr[-1] != psin.size or np.any(np.diff(pptr) < 0):
        raise ValueError("invalid gate-parameter pointer")
    if psin.shape != pcos.shape:
        raise ValueError("parameter sine/cosine shape mismatch")
    if not np.isfinite(psin).all() or not np.isfinite(pcos).all():
        raise ValueError("non-finite gate parameter encoding")
    unit_error = np.max(np.abs(psin * psin + pcos * pcos - 1.0)) if psin.size else 0.0
    if unit_error > 2e-5:
        raise ValueError(f"gate parameter sin/cos unit error {unit_error:.3e}")

    records: list[dict[str, Any]] = []
    for index, name in enumerate(names.tolist()):
        qubits = [int(value) for value in qidx[qptr[index] : qptr[index + 1]]]
        params = [
            float(math.atan2(float(s), float(c)))
            for s, c in zip(
                psin[pptr[index] : pptr[index + 1]],
                pcos[pptr[index] : pptr[index + 1]],
                strict=True,
            )
        ]
        records.append(
            {
                "index": index,
                "name": str(name).lower(),
                "qubits": qubits,
                "params": params,
            }
        )
    return records


def append_gate(circuit: QuantumCircuit, record: Mapping[str, Any]) -> None:
    name = str(record["name"]).lower()
    qubits = [int(value) for value in record["qubits"]]
    params = [float(value) for value in record["params"]]

    aliases = {"cnot": "cx", "i": "id", "phase": "p"}
    name = aliases.get(name, name)

    expected: dict[str, tuple[int, int]] = {
        "id": (1, 0), "x": (1, 0), "y": (1, 0), "z": (1, 0),
        "h": (1, 0), "s": (1, 0), "sdg": (1, 0), "t": (1, 0),
        "tdg": (1, 0), "sx": (1, 0), "sxdg": (1, 0),
        "rx": (1, 1), "ry": (1, 1), "rz": (1, 1), "p": (1, 1),
        "r": (1, 2), "u": (1, 3), "u1": (1, 1), "u2": (1, 2),
        "u3": (1, 3),
        "cx": (2, 0), "cy": (2, 0), "cz": (2, 0), "ch": (2, 0),
        "swap": (2, 0), "iswap": (2, 0), "dcx": (2, 0), "ecr": (2, 0),
        "cp": (2, 1), "crx": (2, 1), "cry": (2, 1), "crz": (2, 1),
        "rxx": (2, 1), "ryy": (2, 1), "rzz": (2, 1), "rzx": (2, 1),
        "cu": (2, 4),
        "ccx": (3, 0), "cswap": (3, 0),
        "barrier": (-1, 0),
    }
    if name not in expected:
        raise ValueError(f"unsupported replay gate {name!r}")
    q_expected, p_expected = expected[name]
    if q_expected >= 0 and len(qubits) != q_expected:
        raise ValueError(f"gate {name} has {len(qubits)} qubits; expected {q_expected}")
    if len(params) != p_expected:
        raise ValueError(f"gate {name} has {len(params)} params; expected {p_expected}")

    if name == "barrier":
        circuit.barrier(*qubits)
    elif name == "u1":
        circuit.p(params[0], qubits[0])
    elif name == "u2":
        circuit.u(math.pi / 2.0, params[0], params[1], qubits[0])
    elif name == "u3":
        circuit.u(params[0], params[1], params[2], qubits[0])
    else:
        method = getattr(circuit, name, None)
        if method is None:
            raise ValueError(f"Qiskit QuantumCircuit has no method for replay gate {name!r}")
        method(*params, *qubits)


def simulate_records(
    records: Sequence[Mapping[str, Any]],
    n_qubits: int,
    *,
    removed_index: int | None = None,
    replacement_axis: str | None = None,
    replacement_angle: float | None = None,
) -> np.ndarray:
    circuit = QuantumCircuit(n_qubits)
    for record in records:
        index = int(record["index"])
        if removed_index is not None and index == removed_index:
            if replacement_axis is None:
                continue
            qubits = list(record["qubits"])
            if len(qubits) != 1 or replacement_angle is None:
                raise ValueError("matched replacement requires one qubit and one angle")
            getattr(circuit, replacement_axis)(float(replacement_angle), int(qubits[0]))
            continue
        append_gate(circuit, record)
    state = np.asarray(Statevector.from_instruction(circuit).data, dtype=np.complex128)
    state /= np.linalg.norm(state)
    return state


def replay_error(reference: np.ndarray, replay: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    replay = np.asarray(replay, dtype=np.complex128).reshape(-1)
    if reference.shape != replay.shape:
        raise ValueError("state shape mismatch")
    overlap_complex = np.vdot(replay, reference)
    overlap_abs = float(abs(overlap_complex))
    if overlap_abs > 0.0:
        aligned = replay * np.exp(1j * np.angle(overlap_complex))
    else:
        aligned = replay
    return {
        "overlap_abs": overlap_abs,
        "overlap_loss": max(0.0, 1.0 - overlap_abs),
        "aligned_max_abs_error": float(np.max(np.abs(reference - aligned))),
    }


def clean_sequence_hash(records: Sequence[Mapping[str, Any]], removed_index: int) -> str:
    payload = []
    for record in records:
        if int(record["index"]) == removed_index:
            continue
        payload.append(
            {
                "name": str(record["name"]),
                "qubits": list(record["qubits"]),
                "params": [round(float(value), 12) for value in record["params"]],
            }
        )
    return BASE.sha256_bytes(BASE.canonical_json(payload).encode("utf-8"))


def insertion_bin(index: int, gate_count: int) -> tuple[float, str]:
    fraction = float(index) / max(1.0, float(gate_count - 1))
    if fraction < 0.25:
        label = "early_0_25"
    elif fraction < 0.50:
        label = "mid_25_50"
    elif fraction < 0.75:
        label = "mid_50_75"
    else:
        label = "late_75_100"
    return fraction, label


def load_rows(root: Path, expected: int) -> list[dict[str, Any]]:
    rows = [dict(row) for row in pq.read_table(root / "manifests" / "item_manifest.parquet").to_pylist()]
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} source examples, found {len(rows)}")
    for row in rows:
        if str(row["split"]) not in {"train", "validation"}:
            raise RuntimeError(f"forbidden split {row['split']!r}")
        BASE.axis_for_mechanism(str(row["raw_label"]))
    return rows


def replay_source(
    root: Path,
    row: Mapping[str, Any],
    *,
    overlap_loss_max: float,
    amplitude_error_max: float,
) -> dict[str, Any]:
    artifact = BASE.source_artifact(root, row)
    with np.load(artifact, allow_pickle=False) as archive_obj:
        archive = {name: archive_obj[name] for name in archive_obj.files}

    required = {
        "entity_id",
        "c__clean_statevector_real", "c__clean_statevector_imag",
        "c__distorted_statevector_real", "c__distorted_statevector_imag",
        "audit__removed_distortion_gate_indices",
    }
    missing = sorted(required - set(archive))
    if missing:
        raise RuntimeError(f"artifact lacks replay validation arrays: {missing}")
    if str(scalar(archive["entity_id"])) != str(row["entity_id"]):
        raise RuntimeError("entity id mismatch")

    clean_ref = BASE.normalized_state(
        archive["c__clean_statevector_real"], archive["c__clean_statevector_imag"]
    )
    distorted_ref = BASE.normalized_state(
        archive["c__distorted_statevector_real"], archive["c__distorted_statevector_imag"]
    )
    n_qubits = int(row["n_qubits"])
    records = gate_records(archive)
    removed = np.asarray(archive["audit__removed_distortion_gate_indices"], dtype=np.int64).reshape(-1)
    if removed.size != 1:
        raise ValueError(f"expected one removed distortion gate, found {removed.size}")
    removed_index = int(removed[0])
    if removed_index < 0 or removed_index >= len(records):
        raise ValueError("removed distortion index out of range")
    distortion = records[removed_index]
    expected_axis = BASE.axis_for_mechanism(str(row["raw_label"]))
    if str(distortion["name"]).lower() != expected_axis:
        raise ValueError(
            f"distortion axis mismatch: raw={row['raw_label']} gate={distortion['name']}"
        )
    if len(distortion["qubits"]) != 1 or len(distortion["params"]) != 1:
        raise ValueError("distortion gate is not a one-qubit one-parameter rotation")
    affected_qubit = BASE.parse_affected_qubit(
        str(row["affected_qubit_signature"]), n_qubits
    )
    if int(distortion["qubits"][0]) != affected_qubit:
        raise ValueError("distortion gate qubit disagrees with affected-qubit signature")
    angle = float(distortion["params"][0])

    clean_replay = simulate_records(records, n_qubits, removed_index=removed_index)
    distorted_replay = simulate_records(records, n_qubits)
    clean_error = replay_error(clean_ref, clean_replay)
    distorted_error = replay_error(distorted_ref, distorted_replay)
    valid = (
        clean_error["overlap_loss"] <= overlap_loss_max
        and distorted_error["overlap_loss"] <= overlap_loss_max
        and clean_error["aligned_max_abs_error"] <= amplitude_error_max
        and distorted_error["aligned_max_abs_error"] <= amplitude_error_max
    )
    fraction, depth_bin = insertion_bin(removed_index, len(records))
    return {
        "valid": bool(valid),
        "entity_id": str(row["entity_id"]),
        "split_group_id": str(row["split_group_id"]),
        "family": str(row["family"]),
        "phase_sensitive_family": bool(row["phase_sensitive_family"]),
        "n_qubits": n_qubits,
        "strength_key": str(row["strength_key"]),
        "affected_qubit_signature": str(row["affected_qubit_signature"]),
        "affected_qubit": affected_qubit,
        "raw_label": str(row["raw_label"]),
        "original_axis": expected_axis,
        "archived_angle": angle,
        "removed_index": removed_index,
        "gate_count": len(records),
        "insertion_depth_fraction": fraction,
        "insertion_depth_fraction_bin": depth_bin,
        "clean_sequence_hash": clean_sequence_hash(records, removed_index),
        "clean_replay_overlap_loss": clean_error["overlap_loss"],
        "clean_replay_aligned_max_abs_error": clean_error["aligned_max_abs_error"],
        "distorted_replay_overlap_loss": distorted_error["overlap_loss"],
        "distorted_replay_aligned_max_abs_error": distorted_error[
            "aligned_max_abs_error"
        ],
        "clean_state": clean_replay,
        "records": records,
    }


def context_key(info: Mapping[str, Any]) -> str:
    payload = {
        "clean_sequence_hash": info["clean_sequence_hash"],
        "removed_index": int(info["removed_index"]),
        "affected_qubit": int(info["affected_qubit"]),
        "archived_angle": round(float(info["archived_angle"]), 12),
        "n_qubits": int(info["n_qubits"]),
    }
    return BASE.sha256_bytes(BASE.canonical_json(payload).encode("utf-8"))


def add_insertion_strata(
    pairs: Sequence[Mapping[str, Any]], strata: list[dict[str, Any]]
) -> None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["insertion_depth_fraction_bin"])].append(row)
    for value in sorted(grouped):
        strata.append(
            BASE.summarize_pairs(
                grouped[value],
                stratum_type="insertion_depth_fraction_bin",
                stratum_value=value,
            )
        )


def main() -> None:
    arguments = parse_args()
    config_path = arguments.config.expanduser().resolve()
    config = BASE.read_json(config_path)
    if config.get("schema") != "triqto.v0_2.matched_bdelta_replay_identifiability_audit.v1":
        raise RuntimeError("unexpected replay audit config schema")

    root = resolve_product(arguments)
    rows = load_rows(root, int(config["source"]["expected_source_examples"]))
    manifest_path = root / "manifests" / "item_manifest.parquet"
    script_path = Path(__file__).resolve()
    identity = {
        "config_sha256": BASE.sha256_file(config_path),
        "source_manifest_sha256": BASE.sha256_file(manifest_path),
        "source_generation_complete_sha256": BASE.sha256_file(root / "generation_complete.json"),
        "runner_sha256": BASE.sha256_file(script_path),
        "v1_runner_sha256": BASE.sha256_file(BASE_PATH),
    }
    audit_id = "audit_" + hashlib.sha256(
        BASE.canonical_json(identity).encode("utf-8")
    ).hexdigest()[:24]
    output_parent = arguments.output_parent.expanduser().resolve()
    output_root = output_parent / audit_id
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing audit: {output_root}")
    output_parent.mkdir(parents=True, exist_ok=True)
    staging = output_parent / f".{audit_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    replay_contract = config["replay_contract"]
    overlap_loss_max = float(replay_contract["state_overlap_loss_max"])
    amplitude_error_max = float(
        replay_contract["aligned_amplitude_max_abs_error_max"]
    )

    infos: list[dict[str, Any]] = []
    preflight_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            try:
                info = replay_source(
                    root,
                    row,
                    overlap_loss_max=overlap_loss_max,
                    amplitude_error_max=amplitude_error_max,
                )
                public = {
                    key: value
                    for key, value in info.items()
                    if key not in {"clean_state", "records"}
                }
                preflight_rows.append(public)
                if info["valid"]:
                    infos.append(info)
                else:
                    failure_rows.append({**public, "reason": "state_replay_mismatch"})
            except Exception as exc:
                failure = {
                    "valid": False,
                    "entity_id": str(row.get("entity_id", "")),
                    "raw_label": str(row.get("raw_label", "")),
                    "family": str(row.get("family", "")),
                    "n_qubits": int(row.get("n_qubits", 0)),
                    "reason": type(exc).__name__,
                    "detail": str(exc),
                }
                preflight_rows.append(failure)
                failure_rows.append(failure)
            if arguments.progress_every > 0 and index % arguments.progress_every == 0:
                print(f"Replay preflight validated {index}/{len(rows)} examples", flush=True)

        BASE.atomic_csv(staging / "replay_preflight.csv", preflight_rows)
        preflight = {
            "schema": SCHEMA,
            "status": "PASS" if not failure_rows else "REFUSED",
            "source_example_count": len(rows),
            "replay_valid_count": len(infos),
            "failure_count": len(failure_rows),
            "failure_reasons": dict(
                sorted(
                    (
                        reason,
                        sum(1 for row in failure_rows if row["reason"] == reason),
                    )
                    for reason in {str(row["reason"]) for row in failure_rows}
                )
            ),
            "max_clean_replay_overlap_loss": max(
                (float(row.get("clean_replay_overlap_loss", 0.0)) for row in preflight_rows),
                default=0.0,
            ),
            "max_clean_replay_aligned_max_abs_error": max(
                (float(row.get("clean_replay_aligned_max_abs_error", 0.0)) for row in preflight_rows),
                default=0.0,
            ),
            "max_distorted_replay_overlap_loss": max(
                (float(row.get("distorted_replay_overlap_loss", 0.0)) for row in preflight_rows),
                default=0.0,
            ),
            "max_distorted_replay_aligned_max_abs_error": max(
                (float(row.get("distorted_replay_aligned_max_abs_error", 0.0)) for row in preflight_rows),
                default=0.0,
            ),
            "scientific_gate": (
                "All archived clean and distorted statevectors must be exactly reproduced "
                "up to global phase before matched counterfactual generation."
            ),
        }
        BASE.atomic_json(staging / "replay_preflight.json", preflight)
        if failure_rows:
            complete = {
                "schema": SCHEMA,
                "status": "REFUSED_REPLAY_VALIDATION",
                "audit_id": audit_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(root),
                "identity": identity,
                "preflight": preflight,
                "classifier_trained": False,
                "historical_v0_1_test_accessed": False,
                "spent_confirmatory_cohort_accessed": False,
            }
            BASE.atomic_json(staging / "audit_complete.json", complete)
            os.replace(staging, output_root)
            raise RuntimeError(
                "matched B_delta replay audit refused because archived circuit replay "
                f"did not validate for every source example; see {output_root}"
            )

        contexts: dict[str, dict[str, Any]] = {}
        for info in infos:
            key = context_key(info)
            if key not in contexts:
                contexts[key] = {
                    "context_id": key,
                    "clean_state": info["clean_state"],
                    "records": info["records"],
                    "family": info["family"],
                    "phase_sensitive_family": info["phase_sensitive_family"],
                    "n_qubits": info["n_qubits"],
                    "strength_key": info["strength_key"],
                    "archived_angle": info["archived_angle"],
                    "affected_qubit_signature": info["affected_qubit_signature"],
                    "affected_qubit": info["affected_qubit"],
                    "removed_index": info["removed_index"],
                    "insertion_depth_fraction": info["insertion_depth_fraction"],
                    "insertion_depth_fraction_bin": info["insertion_depth_fraction_bin"],
                    "source_entity_ids": [info["entity_id"]],
                    "source_split_group_ids": [info["split_group_id"]],
                }
            else:
                contexts[key]["source_entity_ids"].append(info["entity_id"])
                contexts[key]["source_split_group_ids"].append(info["split_group_id"])

        epsilon = float(config["phenomenology"]["epsilon"])
        negligible_floor = float(config["phenomenology"]["negligible_overlap_loss_floor"])
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
            records = context["records"]
            removed_index = int(context["removed_index"])
            angle = float(context["archived_angle"])
            generated: dict[str, dict[str, Any]] = {}

            for mechanism in MECHANISMS:
                axis = BASE.axis_for_mechanism(mechanism)
                distorted = simulate_records(
                    records,
                    n_qubits,
                    removed_index=removed_index,
                    replacement_axis=axis,
                    replacement_angle=angle,
                )
                decomposition = BASE.overlap_decomposition(clean, distorted, epsilon=epsilon)
                label = BASE.phenotype(
                    decomposition,
                    negligible_floor=negligible_floor,
                    dominance_ratio=dominance_ratio,
                )
                evidence = BASE.evidence_for_state(clean, distorted, n_qubits=n_qubits)
                generated[mechanism] = {
                    "phenotype": label,
                    "decomposition": decomposition,
                    "evidence": evidence,
                }
                counterfactual_rows.append(
                    {
                        "context_id": context_id,
                        "source_entity_count": len(set(context["source_entity_ids"])),
                        "source_entity_ids": "|".join(sorted(set(context["source_entity_ids"]))),
                        "family": context["family"],
                        "phase_sensitive_family": context["phase_sensitive_family"],
                        "n_qubits": n_qubits,
                        "strength_key": context["strength_key"],
                        "archived_angle": angle,
                        "affected_qubit_signature": context["affected_qubit_signature"],
                        "affected_qubit": context["affected_qubit"],
                        "removed_index": removed_index,
                        "insertion_depth_fraction": context["insertion_depth_fraction"],
                        "insertion_depth_fraction_bin": context[
                            "insertion_depth_fraction_bin"
                        ],
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
                separation = BASE.pair_separation(
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
                        "phase_sensitive_family": context["phase_sensitive_family"],
                        "n_qubits": n_qubits,
                        "strength_key": context["strength_key"],
                        "affected_qubit_signature": context["affected_qubit_signature"],
                        "removed_index": removed_index,
                        "insertion_depth_fraction": context["insertion_depth_fraction"],
                        "insertion_depth_fraction_bin": context[
                            "insertion_depth_fraction_bin"
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

            if arguments.progress_every > 0 and context_index % arguments.progress_every == 0:
                print(
                    f"Audited {context_index}/{len(contexts)} exact-replay matched contexts",
                    flush=True,
                )

        strata = BASE.stratified_pair_summaries(pair_rows)
        add_insertion_strata(pair_rows, strata)
        decision = BASE.decide(pair_rows, counterfactual_rows, strata, config)
        decision.update(
            {
                "schema": SCHEMA,
                "audit_id": audit_id,
                "source_example_count": len(rows),
                "deduplicated_context_count": len(contexts),
                "counterfactual_count": len(counterfactual_rows),
                "pair_count": len(pair_rows),
                "classifier_trained": False,
                "historical_v0_1_test_accessed": False,
                "spent_confirmatory_cohort_accessed": False,
            }
        )

        BASE.atomic_csv(staging / "counterfactual_metrics.csv", counterfactual_rows)
        BASE.atomic_csv(staging / "pairwise_metrics.csv", pair_rows)
        BASE.atomic_csv(staging / "stratified_metrics.csv", strata)
        BASE.atomic_json(staging / "decision.json", decision)
        file_hashes = {
            name: BASE.sha256_file(staging / name)
            for name in (
                "replay_preflight.csv",
                "replay_preflight.json",
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
                "Exact replay and exact-probability simulator identifiability only. "
                "Finite-shot and hardware deployability remain Step 4."
            ),
        }
        BASE.atomic_json(staging / "audit_complete.json", complete)
        os.replace(staging, output_root)

    except Exception:
        if staging.exists():
            if not output_root.exists() and (staging / "audit_complete.json").is_file():
                os.replace(staging, output_root)
            else:
                for path in sorted(staging.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                staging.rmdir()
        raise

    print()
    print("TRIQTO MATCHED B_DELTA EXACT-REPLAY IDENTIFIABILITY AUDIT COMPLETE")
    print()
    print(f"Decision: {decision['status']}")
    print(f"Replay-valid source examples: {preflight['replay_valid_count']}/{len(rows)}")
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
    print(f"Numerical collision fraction: {decision['numerical_collision_fraction']:.4f}")
    print(
        "Negligible counterfactual fraction: "
        f"{decision['negligible_counterfactual_fraction']:.4f}"
    )
    print(f"Bad eligible context strata: {len(decision['bad_eligible_strata'])}")
    print(
        "Historical v0.1 test accessed: NO\n"
        "Spent confirmatory cohort accessed: NO\n"
        "Classifier trained: NO"
    )
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
